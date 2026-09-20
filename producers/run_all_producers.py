#!/usr/bin/env python3
"""
UrbanStream – Unified Producer
Runs all 4 producers (traffic, pollution, weather, worker) as threads.
Usage:
  python3 producers/run_all_producers.py

Environment variables (all optional, sensible defaults):
  KAFKA_BROKER      – e.g. localhost:9092  (default: localhost:9092)
  TRAFFIC_SPEED     – rows/sec             (default: 100)
  POLLUTION_SPEED   – rows/sec             (default: 50)
  WEATHER_SPEED     – rows/sec             (default: 10)
  DATA_DIR          – path to CSV folder   (default: data/)
  NUM_WORKERS       – simulated workers    (default: 50)
  MOVE_INTERVAL     – worker move secs     (default: 30)
"""

import csv
import io
import json
import logging
import math
import os
import random
import sys
import threading
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-18s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Global config ────────────────────────────────────────────────────────────
KAFKA_BROKER     = os.getenv("KAFKA_BROKER",     "localhost:9092")
DATA_DIR         = Path(os.getenv("DATA_DIR",     "data"))
TRAFFIC_SPEED    = int(os.getenv("TRAFFIC_SPEED",   "500"))
POLLUTION_SPEED  = int(os.getenv("POLLUTION_SPEED",  "200"))
WEATHER_SPEED    = int(os.getenv("WEATHER_SPEED",    "10"))
NUM_WORKERS      = int(os.getenv("NUM_WORKERS",      "50"))
MOVE_INTERVAL    = float(os.getenv("MOVE_INTERVAL",  "30"))

TOPICS = {
    "traffic_stream":   3,
    "pollution_stream": 3,
    "weather_stream":   3,
    "worker_stream":    3,
}

# ─── Zone / borough constants (shared by all producers) ──────────────────────
ALL_ZONES = (
    [f"MN-{i:02d}" for i in range(1, 7)] +
    [f"BK-{i:02d}" for i in range(1, 7)] +
    [f"QN-{i:02d}" for i in range(1, 7)] +
    [f"BX-{i:02d}" for i in range(1, 7)] +
    [f"SI-{i:02d}" for i in range(1, 7)]
)

BOROUGH_BOUNDS = {
    "MN": (40.70, 40.88, -74.02, -73.91),
    "BK": (40.57, 40.74, -74.04, -73.84),
    "QN": (40.54, 40.80, -73.97, -73.70),
    "BX": (40.80, 40.92, -73.94, -73.74),
    "SI": (40.48, 40.65, -74.26, -74.03),
}
BOROUGH_NAMES = {
    "MN": "Manhattan", "BK": "Brooklyn",
    "QN": "Queens",    "BX": "Bronx",
    "SI": "Staten Island",
}
BOROUGH_ZONES = {p: [f"{p}-{i:02d}" for i in range(1, 7)] for p in BOROUGH_BOUNDS}

# Centroids for lat/lon lookups
BOROUGH_CENTROIDS = {
    "MANHATTAN":     (40.7831, -73.9712),
    "BROOKLYN":      (40.6782, -73.9442),
    "QUEENS":        (40.7282, -73.7949),
    "BRONX":         (40.8448, -73.8648),
    "STATEN ISLAND": (40.5795, -74.1502),
}

# Kafka helpers

# called once at startup and makes sure all 4 topics exist before producers start sending
def create_topics(broker: str) -> None:
    """Create all Kafka topics once at startup."""
    log = logging.getLogger("topic-init")
    try:
        admin = AdminClient({"bootstrap.servers": broker, "socket.timeout.ms": 10000}) # Connects as an admin
        meta  = admin.list_topics(timeout=15)
        new_topics = [
            NewTopic(t, num_partitions=p, replication_factor=1)
            for t, p in TOPICS.items()
            if t not in meta.topics # only creates the missing topics
        ]
        if new_topics:
            results = admin.create_topics(new_topics)
            for t, fut in results.items():
                try:
                    fut.result()
                    log.info("Created topic: %s", t)
                except Exception as e:
                    log.warning("Topic %s: %s", t, e)
        else:
            log.info("All Kafka topics already exist.")
    except Exception as exc:
        log.warning("Could not pre-create topics (auto-create may handle it): %s", exc)

# Called by producers and connects to Kafka
def make_producer(client_id: str) -> Producer:
    return Producer({
        "bootstrap.servers":            KAFKA_BROKER,
        "client.id":                    client_id,
        "queue.buffering.max.messages": 100_000,
        "batch.num.messages":           1_000,
        "linger.ms":                    5,
        "compression.type":             "snappy",
    })


def delivery_report(err, msg):
    if err:
        logging.getLogger("kafka").error("Delivery failed [%s]: %s", msg.topic(), err)


# ─── AQI calculator (for pollution producer) ─────────────────────────────────

AQI_BP_PM25 = [
    (0.0,   12.0,   0,  50), (12.1,  35.4,  51, 100),
    (35.5,  55.4, 101, 150), (55.5, 150.4, 151, 200),
    (150.5,250.4, 201, 300), (250.5,350.4, 301, 400),
    (350.5,500.4, 401, 500),
]
AQI_BP_NO2 = [
    (0,   53,   0,  50), (54,  100,  51, 100),
    (101, 360, 101, 150), (361, 649, 151, 200),
    (650,1249, 201, 300), (1250,1649,301, 400),
    (1650,2049,401, 500),
]

def _aqi(conc: float, bp: list) -> float:
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= conc <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    return 500.0

def compute_aqi(pm25: float, no2: float) -> float:
    return round(max(_aqi(pm25, AQI_BP_PM25), _aqi(no2, AQI_BP_NO2)), 1)


def lat_lon_to_zone(lat: float, lon: float, idx: int) -> str:
    for prefix, (lat_lo, lat_hi, lon_lo, lon_hi) in BOROUGH_BOUNDS.items():
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            zones = BOROUGH_ZONES[prefix]
            return zones[idx % len(zones)]
    return f"MN-{(idx % 6) + 1:02d}"


# Reads CSV row by row and sends each row to Kafka
def run_traffic_producer():
    log      = logging.getLogger("traffic-producer")
    topic    = "traffic_stream"
    interval = 1.0 / TRAFFIC_SPEED
    data_file = DATA_DIR / "nyc_traffic.csv"

    if not data_file.exists():
        log.error("Missing %s – skipping traffic producer.", data_file)
        log.error("Download: https://data.cityofnewyork.us/Transportation/Traffic-Speed/4h9m-uh3q")
        return

    producer    = make_producer("traffic-prod")
    total_sent  = 0
    start_time  = time.time()
    w_sent      = 0
    w_start     = time.time()

    log.info("Starting at %d rows/sec → %s", TRAFFIC_SPEED, topic)

    while True:
        with open(data_file, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                try:
                    borough_raw = (row.get("BOROUGH") or "MANHATTAN").strip().upper()
                    borough_key = borough_raw if borough_raw in [
                        "MANHATTAN","BROOKLYN","QUEENS","BRONX","STATEN ISLAND"
                    ] else "MANHATTAN"
                    borough_short = {
                        "MANHATTAN":"MN","BROOKLYN":"BK","QUEENS":"QN",
                        "BRONX":"BX","STATEN ISLAND":"SI"
                    }.get(borough_key, "MN")
                    zone_id = BOROUGH_ZONES[borough_short][idx % 6]

                    speed  = float(row.get("SPEED") or row.get("speed") or "30")
                    lat, lon = BOROUGH_CENTROIDS.get(borough_key, (40.7128, -74.006))
                    lat += (hash(row.get("LINK_ID", str(idx))) % 1000) / 100_000
                    lon += (hash(row.get("LINK_ID", str(idx))) % 1000) / 100_000

                    msg = {
                        "segment_id": row.get("LINK_ID") or f"SEG-{idx:06d}",
                        "speed_kmh":  round(speed * 1.60934, 2),
                        "speed_mph":  round(speed, 2),
                        "borough":    borough_key.title(),
                        "zone_id":    zone_id,
                        "timestamp":  datetime.now(timezone.utc).isoformat(),
                        "lat":        round(lat, 6),
                        "lon":        round(lon, 6),
                    }
                    producer.produce(topic, key=msg["segment_id"].encode(),
                                     value=json.dumps(msg).encode(), callback=delivery_report)
                    producer.poll(0)
                    total_sent += 1
                    w_sent     += 1
                except Exception as e:
                    logging.getLogger("traffic-producer").debug("Row %d skipped: %s", idx, e)

                elapsed = time.time() - w_start
                if elapsed >= 5.0:
                    log.info("%.1f rows/sec | total: %d | uptime: %.0fs",
                             w_sent / elapsed, total_sent, time.time() - start_time)
                    w_sent, w_start = 0, time.time()

                time.sleep(interval)

        producer.flush()
        log.info("CSV loop complete – rewinding (total: %d)", total_sent)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCER 2 – POLLUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_pollution_producer():
    log       = logging.getLogger("pollution-prod ")
    topic     = "pollution_stream"
    data_file = DATA_DIR / "openaq_nyc.csv"
    OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")

    producer   = make_producer("pollution-prod")
    total_sent = 0
    start_time = time.time()

    def pm25_to_aqi(pm):
        for lo, hi, ilo, ihi in AQI_BP_PM25:
            if lo <= pm <= hi:
                return round((ihi-ilo)/(hi-lo)*(pm-lo)+ilo, 1)
        return min(500.0, round(pm * 2.1, 1))

    def nearest_zone(lat, lon):
        best, dist = ALL_ZONES[0], float("inf")
        for z in ALL_ZONES:
            p = z[:2]; a,b,c,d = BOROUGH_BOUNDS[p]; i = int(z[3:])-1
            zlat = round(a+(i//2+.5)*(b-a)/3, 5)
            zlon = round(c+(i%2+.5)*(d-c)/2,  5)
            dd = (lat-zlat)**2 + (lon-zlon)**2
            if dd < dist: dist=dd; best=z
        return best

    # ── LIVE MODE ────────────────────────────────────────────────────────────
    if OPENAQ_API_KEY:
        log.info("LIVE MODE — OpenAQ API every 30s → %s", topic)
        while True:
            try:
                resp = requests.get(
                    "https://api.openaq.org/v3/locations",
                    params={"bbox":"-74.26,40.48,-73.70,40.92",
                            "limit":100},
                    headers={"X-API-Key": OPENAQ_API_KEY},
                    timeout=10,
                )
                resp.raise_for_status()
                locations = resp.json().get("results", [])

                stations = {}
                for loc in locations:
                    lid    = str(loc.get("id",""))
                    coords = loc.get("coordinates") or {}
                    lat    = float(coords.get("latitude",  0) or 0)
                    lon    = float(coords.get("longitude", 0) or 0)
                    if lat == 0: continue
                    stations[lid] = {"lat":lat,"lon":lon,"pm25":None,"no2":None}
                    for sensor in loc.get("sensors", []):
                        param = (sensor.get("parameter") or {}).get("name","").lower()
                        val   = (sensor.get("latest")    or {}).get("value")
                        if val is None: continue
                        if param == "pm25": stations[lid]["pm25"] = float(val)
                        if param == "no2":  stations[lid]["no2"]  = float(val)

                sent = 0
                for lid, s in stations.items():
                    pm25 = s["pm25"] or 8.0
                    no2  = s["no2"]  or 15.0
                    zone = nearest_zone(s["lat"], s["lon"])
                    msg  = {
                        "station_id": f"OAQ-{lid}",
                        "zone_id":    zone,
                        "pm25":       round(pm25, 2),
                        "no2":        round(no2,  2),
                        "aqi":        pm25_to_aqi(pm25),
                        "timestamp":  datetime.now(timezone.utc).isoformat(),
                        "lat": s["lat"], "lon": s["lon"],
                        "source": "openaq_live",
                    }
                    producer.produce(topic, key=zone.encode(),
                                     value=json.dumps(msg).encode(), callback=delivery_report)
                    producer.poll(0)
                    sent += 1; total_sent += 1
                producer.flush()
                log.info("[LIVE] %d readings | total: %d | uptime: %.0fs",
                         sent, total_sent, time.time()-start_time)
            except Exception as e:
                log.error("OpenAQ fetch error: %s", e)
            time.sleep(30)

    # ── CSV FALLBACK MODE ─────────────────────────────────────────────────────
    else:
        log.info("CSV MODE (set OPENAQ_API_KEY for live data) → %s", topic)
        if not data_file.exists():
            log.error("Missing %s – skipping pollution producer.", data_file)
            return
        interval = 1.0 / POLLUTION_SPEED
        buffer: dict = {}
        w_sent = 0; w_start = time.time()
        while True:
            with open(data_file, newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    try:
                        station_id = (row.get("location") or f"NYC-{idx%30:03d}").strip()
                        parameter  = (row.get("parameter") or "").strip().lower()
                        value      = float(row.get("value") or "0")
                        lat        = float(row.get("latitude")  or "40.7128")
                        lon        = float(row.get("longitude") or "-74.006")
                        if station_id not in buffer:
                            buffer[station_id] = {"pm25":0.0,"no2":0.0,"lat":lat,"lon":lon}
                        if "pm25" in parameter: buffer[station_id]["pm25"] = max(0.0, value)
                        elif "no2" in parameter: buffer[station_id]["no2"] = max(0.0, value)
                        else:
                            time.sleep(interval); continue
                        e    = buffer[station_id]
                        zone = nearest_zone(lat, lon)
                        msg  = {"station_id":station_id,"zone_id":zone,
                                "pm25":round(e["pm25"],2),"no2":round(e["no2"],2),
                                "aqi":compute_aqi(e["pm25"],e["no2"]),
                                "timestamp":datetime.now(timezone.utc).isoformat(),
                                "lat":round(lat,6),"lon":round(lon,6),"source":"csv_replay"}
                        producer.produce(topic, key=station_id.encode(),
                                         value=json.dumps(msg).encode(), callback=delivery_report)
                        producer.poll(0)
                        total_sent += 1; w_sent += 1
                    except Exception as e:
                        logging.getLogger("pollution-prod").debug("Row %d skipped: %s", idx, e)
                    elapsed = time.time() - w_start
                    if elapsed >= 5.0:
                        log.info("%.1f rows/sec | total: %d | uptime: %.0fs",
                                 w_sent/elapsed, total_sent, time.time()-start_time)
                        w_sent, w_start = 0, time.time()
                    time.sleep(interval)
            producer.flush()
            log.info("CSV loop complete – rewinding (total: %d)", total_sent)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCER 3 – WEATHER
# ══════════════════════════════════════════════════════════════════════════════

def run_weather_producer():
    log       = logging.getLogger("weather-prod  ")
    topic     = "weather_stream"
    interval  = 1.0 / WEATHER_SPEED
    data_file = DATA_DIR / "weather_nyc.csv"

    if not data_file.exists():
        log.error("Missing %s – skipping weather producer.", data_file)
        log.error("Download from: https://open-meteo.com/ (NYC, hourly)")
        return

    producer   = make_producer("weather-prod")
    total_sent = 0
    start_time = time.time()
    w_sent     = 0
    w_start    = time.time()

    log.info("Starting at %d rows/sec → %s", WEATHER_SPEED, topic)

    TEMP_KEYS  = ["temperature_2m", "temperature_2m (°C)", "temperature", "temp"]
    HUMID_KEYS = ["relativehumidity_2m", "relativehumidity_2m (%)", "humidity", "relative_humidity"]
    WIND_KEYS  = ["windspeed_10m", "windspeed_10m (km/h)", "windspeed", "wind_speed"]
    TIME_KEYS  = ["time", "date", "datetime", "timestamp"]

    def get_val(row, keys):
        for k in keys:
            v = row.get(k, "").strip()
            if v:
                try: return float(v)
                except ValueError: pass
        return None

    while True:
        with open(data_file, newline="", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        # Skip metadata header rows (Open-Meteo sometimes adds 2-3 info rows)
        header_idx = 0
        for i, line in enumerate(lines):
            low = line.lower()
            if "time" in low or "temperature" in low:
                header_idx = i
                break

        reader = csv.DictReader(io.StringIO("".join(lines[header_idx:])))
        for idx, row in enumerate(reader):
            try:
                temp      = get_val(row, TEMP_KEYS)
                humidity  = get_val(row, HUMID_KEYS)
                windspeed = get_val(row, WIND_KEYS)
                if temp is None:
                    time.sleep(interval)
                    continue

                condition = "Clear"
                if humidity  and humidity  > 80: condition = "Humid"
                if windspeed and windspeed > 30: condition = "Windy"
                if temp < 5:                     condition = "Cold"
                if temp > 30:                    condition = "Hot"

                source_ts = next((row.get(k,"") for k in TIME_KEYS if row.get(k)), "")

                msg = {
                    "temperature": round(temp,      1),
                    "humidity":    round(humidity,  1) if humidity  is not None else None,
                    "windspeed":   round(windspeed, 1) if windspeed is not None else None,
                    "condition":   condition,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "source_ts":   source_ts,
                    "lat": 40.7128, "lon": -74.006,
                }
                producer.produce(topic, key=b"nyc-weather",
                                 value=json.dumps(msg).encode(), callback=delivery_report)
                producer.poll(0)
                total_sent += 1
                w_sent     += 1
            except Exception as e:
                logging.getLogger("weather-prod").debug("Row %d skipped: %s", idx, e)

            elapsed = time.time() - w_start
            if elapsed >= 5.0:
                log.info("%.1f rows/sec | total: %d | uptime: %.0fs",
                         w_sent / elapsed, total_sent, time.time() - start_time)
                w_sent, w_start = 0, time.time()

            time.sleep(interval)

        producer.flush()
        log.info("CSV loop complete – rewinding (total: %d)", total_sent)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCER 4 – WORKERS
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compute zone lat/lon centroids
ZONE_LAT: dict[str, float] = {}
ZONE_LON: dict[str, float] = {}
ZONE_BOROUGH_NAME: dict[str, str] = {}
ZONE_NEIGHBORS: dict[str, list[str]] = {}

for _zone in ALL_ZONES:
    _p = _zone[:2]
    _lat_lo, _lat_hi, _lon_lo, _lon_hi = BOROUGH_BOUNDS[_p]
    _idx = int(_zone[3:]) - 1
    _lat_step = (_lat_hi - _lat_lo) / 3
    _lon_step = (_lon_hi - _lon_lo) / 2
    ZONE_LAT[_zone]          = round(_lat_lo + (_idx // 2 + 0.5) * _lat_step, 6)
    ZONE_LON[_zone]          = round(_lon_lo + (_idx  % 2 + 0.5) * _lon_step, 6)
    ZONE_BOROUGH_NAME[_zone] = BOROUGH_NAMES[_p]

for _zone in ALL_ZONES:
    _p = _zone[:2]
    _same  = [z for z in ALL_ZONES if z.startswith(_p) and z != _zone]
    _adj_p = {"MN":"BK","BK":"MN","QN":"MN","BX":"MN","SI":"BK"}.get(_p, _p)
    _adj   = [z for z in ALL_ZONES if z.startswith(_adj_p)][:2]
    ZONE_NEIGHBORS[_zone] = _same + _adj


class GigWorker:
    def __init__(self, wid: str):
        self.worker_id    = wid
        self.zone         = random.choice(ALL_ZONES)
        self.last_move    = time.time()
        self.mob          = random.choice(["high", "medium", "low"])
        self.move_interval = {"high": MOVE_INTERVAL*0.5,
                               "medium": MOVE_INTERVAL,
                               "low": MOVE_INTERVAL*2.0}[self.mob]

    def tick(self) -> bool:
        if time.time() - self.last_move >= self.move_interval:
            self.zone      = random.choice(ZONE_NEIGHBORS[self.zone])
            self.last_move = time.time()
            return True
        return False

    def to_msg(self) -> dict:
        z = self.zone
        return {
            "worker_id":     self.worker_id,
            "zone_id":       z,
            "borough":       ZONE_BOROUGH_NAME[z],
            "lat":           ZONE_LAT[z]  + random.uniform(-0.001, 0.001),
            "lon":           ZONE_LON[z]  + random.uniform(-0.001, 0.001),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "mobility_type": self.mob,
        }


def run_worker_producer():
    log       = logging.getLogger("worker-prod   ")
    topic     = "worker_stream"
    producer  = make_producer("worker-prod")
    workers   = [GigWorker(f"W-{i:02d}") for i in range(1, NUM_WORKERS + 1)]

    total_sent  = 0
    start_time  = time.time()
    w_sent      = 0
    w_start     = time.time()
    HEARTBEAT   = 5.0   # send all workers every 5 seconds

    log.info("Starting %d workers (heartbeat=%.0fs) → %s", NUM_WORKERS, HEARTBEAT, topic)

    while True:
        batch_start = time.time()

        for worker in workers:
            worker.tick()
            msg = worker.to_msg()
            producer.produce(topic, key=worker.worker_id.encode(),
                             value=json.dumps(msg).encode(), callback=delivery_report)
            producer.poll(0)
            total_sent += 1
            w_sent     += 1

        elapsed = time.time() - w_start
        if elapsed >= 5.0:
            zone_counts: dict[str, int] = {}
            for w in workers:
                zone_counts[w.zone] = zone_counts.get(w.zone, 0) + 1
            top = sorted(zone_counts.items(), key=lambda x: -x[1])[:3]
            log.info("%.0f msgs/sec | total: %d | top zones: %s | uptime: %.0fs",
                     w_sent / elapsed, total_sent,
                     ", ".join(f"{z}({c})" for z, c in top),
                     time.time() - start_time)
            w_sent, w_start = 0, time.time()

        elapsed_batch = time.time() - batch_start
        time.sleep(max(0.0, HEARTBEAT - elapsed_batch))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN – start all 4 producers as daemon threads
# ══════════════════════════════════════════════════════════════════════════════

PRODUCERS = [
    ("traffic",   run_traffic_producer),
    ("pollution", run_pollution_producer),
    ("weather",   run_weather_producer),
    ("workers",   run_worker_producer),
]


def main():
    log = logging.getLogger("main")

    log.info("=" * 60)
    log.info("  UrbanStream – Unified Producer")
    log.info("=" * 60)
    log.info("  Broker   : %s", KAFKA_BROKER)
    log.info("  Data dir : %s", DATA_DIR.resolve())
    log.info("  Traffic  : %d rows/sec", TRAFFIC_SPEED)
    log.info("  Pollution: %d rows/sec", POLLUTION_SPEED)
    log.info("  Weather  : %d rows/sec", WEATHER_SPEED)
    log.info("  Workers  : %d simulated", NUM_WORKERS)
    log.info("=" * 60)

    # Check what data files exist
    for fname in ["nyc_traffic.csv", "openaq_nyc.csv", "weather_nyc.csv"]:
        path = DATA_DIR / fname
        status = "✓ found" if path.exists() else "✗ MISSING"
        log.info("  %s  ← %s", status, fname)
    log.info("=" * 60)

    # Wait for Kafka to be reachable (retry for up to 60 seconds)
    log.info("Waiting for Kafka broker at %s ...", KAFKA_BROKER)
    for attempt in range(12):
        try:
            admin = AdminClient({"bootstrap.servers": KAFKA_BROKER, "socket.timeout.ms": 5000})
            admin.list_topics(timeout=5)
            log.info("Kafka broker reachable.")
            break
        except Exception as exc:
            log.warning("Attempt %d/12 – broker not ready: %s", attempt + 1, exc)
            time.sleep(5)
    else:
        log.error("Could not connect to Kafka broker after 60s. Exiting.")
        sys.exit(1)

    # Create topics
    create_topics(KAFKA_BROKER)

    # Launch all 4 producers as parallel threads
    threads = []
    for name, func in PRODUCERS:
        t = threading.Thread(target=func, name=name, daemon=True)
        t.start()
        threads.append(t)
        log.info("Thread started: %s", name)
        time.sleep(0.3)   # slight stagger to avoid thundering-herd on Kafka

    log.info("All producers running. Press Ctrl+C to stop.")

    # Keep main thread alive; restart any thread that dies
    try:
        while True:
            time.sleep(10)
            for i, (name, func) in enumerate(PRODUCERS):
                if not threads[i].is_alive():
                    log.warning("Thread '%s' died – restarting...", name)
                    t = threading.Thread(target=func, name=name, daemon=True)
                    t.start()
                    threads[i] = t
    except KeyboardInterrupt:
        log.info("Shutdown requested – stopping all producers.")


if __name__ == "__main__":
    main()
