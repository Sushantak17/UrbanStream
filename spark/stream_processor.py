"""
UrbanStream – Spark Structured Streaming Processor

Each topic is aggregated independently in its own streaming query.
Zone scoring happens inside foreachBatch using a static join on collected rows.
This avoids the stream-stream join watermark delay that caused empty batches.

"""

import json, logging, math, os, random
from collections import defaultdict
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BROKER    = os.getenv("KAFKA_BROKER",    "redpanda:9092")
CHECKPOINT_BASE = os.getenv("CHECKPOINT_BASE", "s3a://checkpoints/urbanstream")
DATALAKE_BASE   = os.getenv("DATALAKE_BASE",   "s3a://datalake/urbanstream")
WATERMARK       = os.getenv("WATERMARK",       "2 minutes")
REDIS_HOST      = os.getenv("REDIS_HOST",      "redis")
BASELINE_SPEED  = 40.0
K               = 4      # number of clusters

# Exposure thresholds — tune these to control demo speed
HIGH_AQI  = float(os.getenv("HIGH_AQI",   "100.0")) # WHO threshold — unhealthy for sensitive groups
WARN_HRS  = float(os.getenv("WARN_HRS",   "1.5"))   # hours in high-AQI → WARNING
CRIT_HRS  = float(os.getenv("CRIT_HRS",   "3.0"))   # hours in high-AQI → CRITICAL

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [STREAM-PROCESSOR] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# connects to Redis and pings it even before spark starts
try:
    import redis as _rl
    _redis = _rl.Redis(host=REDIS_HOST, port=6379, db=0,
                       socket_timeout=2, decode_responses=True)
    _redis.ping()
    log.info("Redis connected at %s:6379", REDIS_HOST)
except Exception as e:
    _redis = None
    log.warning("Redis unavailable: %s", e)

# ── Schemas ───────────────────────────────────────────────────────────────────
TRAFFIC_SCHEMA = StructType([
    StructField("segment_id", StringType()),
    StructField("speed_kmh",  DoubleType()),
    StructField("borough",    StringType()),
    StructField("zone_id",    StringType()),
    StructField("timestamp",  StringType()),
    StructField("lat",        DoubleType()),
    StructField("lon",        DoubleType()),
])
POLLUTION_SCHEMA = StructType([
    StructField("station_id", StringType()),
    StructField("zone_id",    StringType()),
    StructField("pm25",       DoubleType()),
    StructField("no2",        DoubleType()),
    StructField("aqi",        DoubleType()),
    StructField("timestamp",  StringType()),
    StructField("lat",        DoubleType()),
    StructField("lon",        DoubleType()),
])
WEATHER_SCHEMA = StructType([
    StructField("temperature", DoubleType()),
    StructField("humidity",    DoubleType()),
    StructField("windspeed",   DoubleType()),
    StructField("condition",   StringType()),
    StructField("timestamp",   StringType()),
])
WORKER_SCHEMA = StructType([
    StructField("worker_id",     StringType()),
    StructField("zone_id",       StringType()),
    StructField("borough",       StringType()),
    StructField("timestamp",     StringType()),
    StructField("mobility_type", StringType()),
])

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN","BK","QN","BX","SI"] for i in range(1,7)]

# ── In-memory streaming state ─────────────────────────────────────────────────
_traffic   = {}   # zone_id → {avg_speed, borough}
_pollution = {}   # zone_id → {avg_aqi, avg_pm25, avg_no2}
_exposure  = {}   # worker_id → {hours_in_high_aqi, daily_avg_aqi, count, ...}
# Load previous exposure from disk if it exists (survives Spark restarts)
_EXPOSURE_FILE = "/tmp/urbanstream/worker_exposure.json"
try:
    if os.path.exists(_EXPOSURE_FILE):
        with open(_EXPOSURE_FILE) as _f:
            _exposure = json.load(_f)
        log.info("Loaded exposure state for %d workers from disk", len(_exposure))
except Exception as _e:
    log.warning("Could not load exposure file: %s", _e)
_total     = [0]

# ═════════════════════════════════════════════════════════════════════════════
# STREAMING K-MEANS  (MiniBatchKMeans-style, pure Python, no numpy)
# ═════════════════════════════════════════════════════════════════════════════
#
# Design mirrors sklearn's MiniBatchKMeans.partial_fit:
#   • Centroids are maintained across batches (persistent state)
#   • Each batch does ONE pass of mini-batch gradient descent on centroids
#   • Learning rate decays as 1/(n_batches+1) so early batches have more
#     influence than later ones — good for non-stationary streams
#   • 4 features per zone: avg_speed, avg_aqi, avg_pm25, borough_density
#   • Inertia (sum of squared distances to nearest centroid) is computed
#     and logged so you can watch the model converge over time

from streaming_kmeans import StreamingKMeans, CLUSTER_NAMES  # testable without Spark

# Anomaly detection (Isolation Forest)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ml.anomaly_detector import AnomalyDetector
    _anomaly_detector = AnomalyDetector()
    log.info("Anomaly detector: %s", "ready" if _anomaly_detector.is_ready else "model not found")
except Exception as e:
    _anomaly_detector = None
    log.warning("Anomaly detector import failed: %s", e)

# Borough density proxy (population density index 0-1)
BOROUGH_DENSITY = {"MN": 1.0, "BK": 0.75, "QN": 0.60, "BX": 0.55, "SI": 0.30}

# Singleton model — persists across all foreachBatch calls in this JVM process
_skm = StreamingKMeans(k=K)

# ── Centroid persistence helpers ──────────────────────────────────────────────
_SKM_CENTROID_KEY = "skm_centroids"

# loads previous K-Means model state
def _restore_centroids_from_redis() -> None:
    """Reload centroid state saved by a previous Spark run so partial_fit
    continues from where it left off instead of reinitialising."""
    if not _redis:
        return
    try:
        raw = _redis.get(_SKM_CENTROID_KEY)
        if not raw:
            return
        state = json.loads(raw)
        _skm.centroids  = state["centroids"]
        _skm.counts     = state["counts"]
        _skm.n_batches  = state["n_batches"]
        _skm._feat_min  = state["feat_min"]
        _skm._feat_max  = state["feat_max"]
        log.info("[SKM] Restored centroids from Redis (batch #%d)", _skm.n_batches)
    except Exception as e:
        log.warning("[SKM] Centroid restore skipped: %s", e)

def _save_centroids_to_redis() -> None:
    """Persist current centroid state to Redis so restarts are warm."""
    if not _redis or _skm.centroids is None:
        return
    try:
        state = {
            "centroids": _skm.centroids,
            "counts":    _skm.counts,
            "n_batches": _skm.n_batches,
            "feat_min":  _skm._feat_min,
            "feat_max":  _skm._feat_max,
        }
        _redis.set(_SKM_CENTROID_KEY, json.dumps(state), ex=86400)
    except Exception as e:
        log.warning("[SKM] Centroid save skipped: %s", e)

_restore_centroids_from_redis()

def _warmstart_clusters_from_file() -> None:
    """Push offline KMeans labels to Redis on startup so dashboard
    has real cluster data from batch 0, not synthetic fallback."""
    path = "/opt/spark/jobs/cluster_labels.json"
    if not _redis or not os.path.exists(path):
        return
    try:
        with open(path) as f:
            labels = json.load(f).get("zone_labels", {})
        pipe = _redis.pipeline()
        for zone, name in labels.items():
            pipe.set(f"cluster:{zone}", name, ex=3600)
        pipe.execute()
        log.info("[SKM] Warm-started %d cluster labels from offline KMeans", len(labels))
    except Exception as e:
        log.warning("[SKM] Warm-start failed: %s", e)

_warmstart_clusters_from_file()


def _retrain_and_push_clusters(batch_id: int) -> None:
    """
    Called after every traffic or pollution batch.
    Builds feature matrix from current in-memory state,
    runs partial_fit, pushes updated cluster labels to Redis.
    """
    zones = [z for z in ALL_ZONES if z in _traffic or z in _pollution]
    if len(zones) < K:
        log.debug("[SKM] Not enough zones yet (%d < %d)", len(zones), K)
        return

    # Build 4-feature matrix: [avg_speed, avg_aqi, avg_pm25, density]
    X, zone_ids = [], []
    for z in ALL_ZONES:   # use ALL_ZONES so ordering is stable
        t = _traffic.get(z,   {"avg_speed": 30.0})
        p = _pollution.get(z, {"avg_aqi": 50.0, "avg_pm25": 10.0})
        X.append([
            t["avg_speed"],
            p["avg_aqi"],
            p["avg_pm25"],
            BOROUGH_DENSITY.get(z[:2], 0.5),
        ])
        zone_ids.append(z)

    # ── PARTIAL FIT — this is the streaming retraining every batch ────────────
    labels = _skm.partial_fit(X)

    # Assign semantic names from centroid hazard ranking
    zone_clusters = _skm.semantic_labels(zone_ids, labels)

    # push all cluster labels to Redis
    if _redis:
        try:
            pipe = _redis.pipeline()
            for z, name in zone_clusters.items():
                pipe.set(f"cluster:{z}", name, ex=300)
            _inertia = round(_skm.inertia, 4) if _skm.inertia != float("inf") else 0.0
            pipe.set("skm_batch_count", _skm.n_batches, ex=86400)
            pipe.set("skm_inertia",     _inertia,        ex=86400)
            pipe.execute()
            _save_centroids_to_redis()   # persist so restarts are warm
        except Exception as e:
            log.warning("[SKM] Redis push failed: %s", e)

    log.info(
        "[SKM] Batch #%d | partial_fit #%d | %d zones reclassified | inertia=%.4f",
        batch_id, _skm.n_batches, len(zone_ids), _skm.inertia
    )

    # Log cluster composition for visibility
    by_cluster = defaultdict(list)
    for z, name in zone_clusters.items():
        by_cluster[name].append(z)
    for name, zlist in sorted(by_cluster.items()):
        log.info("  %-28s → %d zones: %s", name, len(zlist),
                 ", ".join(sorted(zlist)[:4]) + ("…" if len(zlist) > 4 else ""))



def _push_zone_scores(batch_id: int) -> None:
    zones = set(list(_traffic.keys()) + list(_pollution.keys()))
    if not zones:
        return
    scores = []
    for zid in zones:
        t   = _traffic.get(zid,   {"avg_speed": BASELINE_SPEED, "borough": ""})
        p   = _pollution.get(zid, {"avg_aqi": 50.0, "avg_pm25": 10.0, "avg_no2": 20.0})
        spd = t["avg_speed"]; aqi = p["avg_aqi"]
        ss  = min(1.0, spd / BASELINE_SPEED)
        aq  = 1.0 - min(1.0, aqi / 200.0)   # inverted: higher AQI → lower score
        zs  = round(ss * 0.5 + aq * 0.5, 3)
        # Thresholds tuned to real data ranges (speed 30-57 km/h, AQI 33-50)
        cong = spd < BASELINE_SPEED * 0.85   # < 34 km/h  → congestion
        poll = aqi > 38                       # > 38 AQI   → pollution alert
        evt  = ("COMPOUND_EVENT"   if cong and poll else
                "CONGESTION_EVENT" if cong else
                "POLLUTION_ALERT"  if poll else "NORMAL")
        scores.append({"zone_id": zid, "avg_speed": round(spd,1),
                        "avg_aqi": round(aqi,1), "avg_pm25": round(p["avg_pm25"],1),
                        "zone_score": zs, "event_type": evt,
                        "borough": t["borough"], "batch_id": batch_id,
                        "ts": datetime.now().isoformat(timespec="seconds")})
        
    # all 30 zones in one pipeline call in redis
    if _redis:
        try:
            pipe = _redis.pipeline()
            for s in scores:
                pipe.set(f"zone_score:{s['zone_id']}", json.dumps(s), ex=300)
            pipe.execute()
        except Exception as e:
            log.warning("[REDIS] zone push failed: %s", e)

    evts = [s for s in scores if s["event_type"] != "NORMAL"]
    if evts:
        log.warning("[EVENTS] Batch #%d → %d threshold events: %s",
                    batch_id, len(evts),
                    ", ".join(f"{e['zone_id']}:{e['event_type']}" for e in evts[:5]))

    # ── Anomaly detection (Isolation Forest) ──────────────────────────────────
    if _anomaly_detector and _anomaly_detector.is_ready:
        zone_features = {s["zone_id"]: s for s in scores}
        anomalies = _anomaly_detector.detect_anomalies(zone_features)
        if anomalies and _redis:
            try:
                pipe = _redis.pipeline()
                for a in anomalies:
                    a["batch_id"] = batch_id
                    a["ts"] = datetime.now().isoformat(timespec="seconds")
                    pipe.set(f"anomaly:{a['zone_id']}", json.dumps(a), ex=300)
                pipe.execute()
                log.warning("[ANOMALY] Batch #%d → %d anomalies: %s",
                            batch_id, len(anomalies),
                            ", ".join(f"{a['zone_id']}[{a['severity']}]" for a in anomalies[:5]))
            except Exception as e:
                log.warning("[REDIS] anomaly push failed: %s", e)

    if _redis:
        log.info("[ZONE_SCORES] Batch #%d | %d zones | pushed to Redis", batch_id, len(scores))
    else:
        log.warning("[ZONE_SCORES] Batch #%d | %d zones | Redis unavailable — data NOT pushed", batch_id, len(scores))

    # write permanently as JSON file in MINIO
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark and scores:
            path = f"{DATALAKE_BASE}/zone_scores/batch_{batch_id:06d}"
            (spark.createDataFrame(scores)
                  .coalesce(1)
                  .write.mode("overwrite")
                  .json(path))
            log.info("[DATALAKE] Batch #%d written to %s", batch_id, path)
    except Exception as e:
        log.warning("[DATALAKE] Write failed: %s", e)


# Creates a streaming DataFrame that represents an infinite table of Kafka messages. 
def read_kafka(spark, topic, schema):
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", KAFKA_BROKER)
           .option("subscribe", topic)
           .option("startingOffsets", "latest")
           .option("failOnDataLoss", "false")
           .option("maxOffsetsPerTrigger", 10000)
           .load())
    return (raw.select(F.from_json(F.col("value").cast("string"), schema).alias("d"),
                       F.col("timestamp").alias("kafka_ts"))
            .select("d.*", "kafka_ts")
            .withColumn("event_time",
                F.coalesce(F.to_timestamp("timestamp"), F.col("kafka_ts"))))


# processes one 30 second batch of traffic data
def handle_traffic(df: DataFrame, batch_id: int):
    rows = df.collect()
    if not rows:
        return
    zone_data = {}
    for r in rows:
        zid = r["zone_id"] or "MN-01"
        spd = float(r["speed_kmh"] or 30)
        zone_data.setdefault(zid, {"speeds": [], "borough": r["borough"] or ""})["speeds"].append(spd)
    for zid, d in zone_data.items():
        _traffic[zid] = {"avg_speed": sum(d["speeds"]) / len(d["speeds"]),
                         "borough": d["borough"]}
    _total[0] += len(rows)
    log.info("[TRAFFIC] Batch #%d | %d rows | %d zones | total %d",
             batch_id, len(rows), len(zone_data), _total[0])
    if _redis:
        try: _redis.set("perf_total_records", _total[0], ex=86400)
        except Exception: pass
    # ── retrain streaming KMeans every traffic batch ──────────────────────────
    _retrain_and_push_clusters(batch_id)
    _push_zone_scores(batch_id)


def handle_pollution(df: DataFrame, batch_id: int):
    rows = df.collect()
    if not rows:
        return
    zone_data = {}
    for r in rows:
        zid = r["zone_id"] or "MN-01"
        zone_data.setdefault(zid, {"aqis": [], "pm25s": [], "no2s": []})
        zone_data[zid]["aqis"].append(float(r["aqi"] or 50))
        zone_data[zid]["pm25s"].append(float(r["pm25"] or 10))
        zone_data[zid]["no2s"].append(float(r["no2"] or 20))
    for zid, d in zone_data.items():
        _pollution[zid] = {"avg_aqi":  sum(d["aqis"])  / len(d["aqis"]),
                           "avg_pm25": sum(d["pm25s"]) / len(d["pm25s"]),
                           "avg_no2":  sum(d["no2s"])  / len(d["no2s"])}
    _total[0] += len(rows)
    log.info("[POLLUTION] Batch #%d | %d rows | %d zones", batch_id, len(rows), len(zone_data))
    # ── retrain streaming KMeans every pollution batch ────────────────────────
    _retrain_and_push_clusters(batch_id)
    _push_zone_scores(batch_id)


def handle_workers(df: DataFrame, batch_id: int):
    rows = df.collect()
    if not rows:
        return
    updated = []
    for r in rows:
        wid  = r["worker_id"]
        zid  = r["zone_id"] or "MN-01"
        aqi  = _pollution.get(zid, {}).get("avg_aqi", 75.0)
        prev = _exposure.get(wid, {"hours_in_high_aqi": 0.0, "daily_avg_aqi": 0.0, "count": 0})
        new_high_mins = prev["hours_in_high_aqi"] * 60 + (1.0 if aqi > HIGH_AQI else 0)
        new_count     = prev["count"] + 1
        new_avg_aqi   = (prev["daily_avg_aqi"] * prev["count"] + aqi) / new_count
        hrs_high      = new_high_mins / 60.0
        status        = ("CRITICAL" if hrs_high > CRIT_HRS else
                         "WARNING"  if hrs_high > WARN_HRS else "SAFE")
        _exposure[wid] = {"worker_id": wid, "zone_id": zid,
                          "hours_in_high_aqi": round(hrs_high, 3),
                          "daily_avg_aqi":     round(new_avg_aqi, 1),
                          "exposure_status":   status, "count": new_count}
        updated.append(_exposure[wid])

    # Persist to disk so restarts don't lose accumulated exposure
    try:
        os.makedirs("/tmp/urbanstream", exist_ok=True)
        with open(_EXPOSURE_FILE, "w") as f:
            json.dump(_exposure, f)
    except Exception as e:
        log.warning("Could not save exposure file: %s", e)
    if _redis:
        try:
            pipe = _redis.pipeline()
            for w in updated:
                rec = {**w,
                       "high_aqi_minutes": round(w["hours_in_high_aqi"] * 60, 2),
                       "total_minutes":    round(w["count"] * 1.0, 1),
                       "aqi_sum":          round(w["daily_avg_aqi"] * w["count"], 1)}
                pipe.set(f"exposure:{w['worker_id']}", json.dumps(rec), ex=60)
                pipe.set(f"worker_zone:{w['worker_id']}", w["zone_id"], ex=300)
            pipe.execute()
        except Exception as e:
            log.warning("[REDIS] worker push failed: %s", e)
    crit = sum(1 for w in updated if w["exposure_status"] == "CRITICAL")
    warn = sum(1 for w in updated if w["exposure_status"] == "WARNING")
    log.info("[WORKERS] Batch #%d | %d workers | Crit:%d Warn:%d",
             batch_id, len(updated), crit, warn)


def handle_weather(df: DataFrame, batch_id: int):
    rows = df.collect()
    if not rows:
        return
    r = rows[-1]
    log.info("[WEATHER] Batch #%d | %.1f°C  %.0f%%  %.1f km/h",
             batch_id, float(r["temperature"] or 0),
             float(r["humidity"] or 0), float(r["windspeed"] or 0))
    if _redis:
        try:
            _redis.set("weather_current", json.dumps({
                "temperature": float(r["temperature"] or 0),
                "humidity":    float(r["humidity"]    or 0),
                "windspeed":   float(r["windspeed"]   or 0),
                "condition":   str(r["condition"]     or "Clear"),
                "ts":          datetime.now().isoformat(timespec="seconds")}), ex=300)
        except Exception:
            pass


def handle_metrics(df: DataFrame, batch_id: int):
    count = df.count()
    _total[0] += count
    log.info("[METRICS] Batch #%d | %d records | running total: %d",
             batch_id, count, _total[0])
    if _redis:
        try:
            _redis.set("perf_total_records", _total[0], ex=86400)
            entry = json.dumps({"ts":         datetime.now().strftime("%H:%M"),
                                "rps":        count,
                                "lag_ms":     50,
                                "storage_mb": _total[0] / 1000})
            _redis.rpush("perf_history", entry)
            _redis.ltrim("perf_history", -20, -1)
            _redis.expire("perf_history", 3600)
        except Exception:
            pass


# SparkSession + Main
# Creates the SparkSession with all configs
def create_spark():
    minio_endpoint = os.getenv("MINIO_ENDPOINT",       "http://minio:9000")
    minio_user     = os.getenv("MINIO_ROOT_USER",      "minioadmin")
    minio_password = os.getenv("MINIO_ROOT_PASSWORD",  "minioadmin")

    return (SparkSession.builder
            .appName("UrbanStream-v4")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.streaming.stopGracefullyOnShutdown", "true")
            # ── S3A / MinIO ──────────────────────────────────────────────────
            .config("spark.hadoop.fs.s3a.endpoint",               minio_endpoint)
            .config("spark.hadoop.fs.s3a.access.key",             minio_user)
            .config("spark.hadoop.fs.s3a.secret.key",             minio_password)
            .config("spark.hadoop.fs.s3a.path.style.access",      "true")
            .config("spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .getOrCreate())


def main():
    log.info("UrbanStream Spark Processor v4")
    log.info("  Streaming KMeans: partial_fit every batch, k=%d", K)
    log.info("  Features: [avg_speed, avg_aqi, avg_pm25, borough_density]")
    log.info("  Broker=%s  Redis=%s", KAFKA_BROKER, REDIS_HOST)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info("Datalake  → %s", DATALAKE_BASE)
    log.info("Checkpoints → %s", CHECKPOINT_BASE)

    # creates 4 streaming DataFrames, one per Kafka topic
    traffic_raw   = read_kafka(spark, "traffic_stream",   TRAFFIC_SCHEMA)
    pollution_raw = read_kafka(spark, "pollution_stream", POLLUTION_SCHEMA)
    weather_raw   = read_kafka(spark, "weather_stream",   WEATHER_SCHEMA)
    worker_raw    = read_kafka(spark, "worker_stream",    WORKER_SCHEMA)

    def _qs(df, cp, fn):
        return (df.withWatermark("event_time", WATERMARK)
                  .writeStream.outputMode("append")
                  .option("checkpointLocation", cp)
                  .trigger(processingTime="30 seconds")
                  .foreachBatch(fn).start())
    
    # starts all 5 in parallel and they never stop until Spark shuts down  
    q1 = _qs(traffic_raw,   f"{CHECKPOINT_BASE}/traffic",   handle_traffic)
    q2 = _qs(pollution_raw, f"{CHECKPOINT_BASE}/pollution",  handle_pollution)
    q3 = _qs(weather_raw,   f"{CHECKPOINT_BASE}/weather",    handle_weather)
    q4 = _qs(worker_raw,    f"{CHECKPOINT_BASE}/workers",    handle_workers)
    q5 = (traffic_raw.writeStream.outputMode("append")
          .option("checkpointLocation", f"{CHECKPOINT_BASE}/metrics")
          .trigger(processingTime="60 seconds")
          .foreachBatch(handle_metrics).start())

    log.info("All 5 queries running — first batch + cluster update in ~30s")
    log.info("  traffic=%s  pollution=%s  weather=%s  workers=%s  metrics=%s",
             q1.id, q2.id, q3.id, q4.id, q5.id)
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()