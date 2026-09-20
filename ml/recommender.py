#!/usr/bin/env python3
"""
UrbanStream Recommender v5
- LinUCB contextual bandit for SAFE workers (learns optimal routing)
- Distance-based zone recommendations for CRITICAL/WARNING workers
- LSTM-powered AQI forecasting with AR(3) fallback
- Accumulates worker exposure every 30s using Redis as persistent state
"""
import json, logging, math, os, random, time
import numpy as np
from datetime import datetime, timezone
from collections import deque
import redis

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [RECOMMENDER] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REDIS_HOST   = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))
RUN_INTERVAL = int(os.getenv("RUN_INTERVAL", "30"))
NUM_WORKERS   = 50
HIGH_AQI      = 40.0   # WHO AQI threshold — unhealthy for sensitive groups
WHO_DAILY_HRS = 8.0     # WHO: no more than 8h/day above AQI 100
WARN_HRS      = 1.5     # WARNING after 1.5h in high-AQI zone
CRIT_HRS      = 3.0     # CRITICAL after 3h  in high-AQI zone

# Average walking/biking speed for gig workers (km/h) — used for travel time estimate
WORKER_SPEED_KMH = 15.0
# Distance decay factor: how much to penalise distant zones in scoring
DISTANCE_DECAY   = 0.3   # score / (1 + decay * distance_km)

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN","BK","QN","BX","SI"] for i in range(1,7)]

CLUSTER_WEIGHTS = {
    "Permanently Hazardous": 0.1,
    "Peak Hour Hazardous":   0.5,
    "Weather Sensitive":     0.7,
    "Safe Corridor":         1.0,
}

# ─── Zone coordinates (centroids computed from borough bounding boxes) ────────
BOROUGH_BOUNDS = {
    "MN": (40.70, 40.88, -74.02, -73.91),
    "BK": (40.57, 40.74, -74.04, -73.84),
    "QN": (40.54, 40.80, -73.97, -73.70),
    "BX": (40.80, 40.92, -73.94, -73.74),
    "SI": (40.48, 40.65, -74.26, -74.03),
}
ZONE_COORDS: dict[str, tuple[float, float]] = {}
for _z in ALL_ZONES:
    _p = _z[:2]
    _a, _b, _c, _d = BOROUGH_BOUNDS[_p]
    _i = int(_z[3:]) - 1
    ZONE_COORDS[_z] = (
        round(_a + (_i // 2 + 0.5) * (_b - _a) / 3, 6),
        round(_c + (_i % 2  + 0.5) * (_d - _c) / 2, 6),
    )


# ─── Haversine distance (km) ─────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two lat/lon points in kilometres.
    Uses the Haversine formula — accurate for short distances within a city.
    """
    R = 6371.0  # Earth radius in km
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def zone_distance_km(zone_a: str, zone_b: str) -> float:
    """Distance between two zone centroids in km."""
    if zone_a == zone_b:
        return 0.0
    lat1, lon1 = ZONE_COORDS[zone_a]
    lat2, lon2 = ZONE_COORDS[zone_b]
    return round(haversine_km(lat1, lon1, lat2, lon2), 2)


def travel_time_min(distance_km: float) -> float:
    """Estimated travel time in minutes at WORKER_SPEED_KMH."""
    return round(distance_km / WORKER_SPEED_KMH * 60, 1)


# ─── AQI Forecast (LSTM with AR(3) fallback) ─────────────────────────────────
# ForecastEngine handles its own sliding window internally.
# Falls back to AR(3) if LSTM model not trained yet.
try:
    from ml.aqi_forecaster import ForecastEngine
    _forecast_engine = ForecastEngine()
    _using_lstm = _forecast_engine.is_lstm_active
    log.info("Forecast engine: %s", "LSTM" if _using_lstm else "AR(3) fallback")
except Exception as e:
    _forecast_engine = None
    _using_lstm = False
    log.warning("ForecastEngine import failed (%s) — using inline AR(3)", e)

# ─── LinUCB Contextual Bandit ─────────────────────────────────────────────────
try:
    from ml.bandit_router import LinUCBRouter
    _bandit_router = LinUCBRouter()
    _bandit_router.load_state()  # warm restart from saved A/b matrices
    log.info("LinUCB bandit: loaded (n_obs=%d, alpha=%.4f)",
             _bandit_router.n_obs, _bandit_router.alpha)
except Exception as e:
    _bandit_router = None
    log.warning("LinUCB import failed (%s) — using distance-based heuristic only", e)

# Legacy AR(3) history for fallback
_aqi_history: dict = {z: deque(maxlen=12) for z in ALL_ZONES}

def _forecast_aqi(zone: str, current_aqi: float) -> float:
    """
    Predict next AQI for a zone. Uses LSTM if available, AR(3) otherwise.
    """
    # Try LSTM forecast engine first
    if _forecast_engine is not None:
        return _forecast_engine.forecast(zone, current_aqi)

    # Inline AR(3) fallback
    hist = _aqi_history[zone]
    hist.append(current_aqi)
    n = len(hist)
    if n < 4:
        return current_aqi

    y = np.array(list(hist), dtype=float)
    p = 3
    X = np.column_stack([y[i:n-p+i] for i in range(p)])
    y_target = y[p:]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
        forecast = float(np.dot(coeffs, y[-p:]))
        return round(max(0.0, min(500.0, forecast)), 1)
    except Exception:
        return current_aqi


# ─── Redis ────────────────────────────────────────────────────────────────────
def get_r():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                       socket_timeout=5, decode_responses=True)


# Stable deterministic zone score (doesn't change randomly each cycle)
def _synthetic_score(zone):
    base = {"MN":0.38,"BK":0.52,"QN":0.64,"BX":0.44,"SI":0.76}.get(zone[:2], 0.5)
    rng  = random.Random(hash(zone) & 0xFFFF)
    # AQI range 50–200 so ~half of zones cross HIGH_AQI=100 and exposure accumulates
    aqi  = rng.uniform(50, 200)
    spd  = rng.uniform(12, 68)
    ss   = min(1.0, spd / 40.0)
    aq   = 1.0 - min(1.0, aqi / 200.0)   # inverted: higher AQI → lower score
    evts = ["NORMAL","NORMAL","NORMAL","CONGESTION_EVENT","POLLUTION_ALERT"]
    return {"zone_id":zone,"zone_score":round((ss*0.5)+(aq*0.5),3),
            "avg_aqi":round(aqi,1),"avg_speed":round(spd,1),
            "event_type":rng.choice(evts)}

def _synthetic_cluster(zone):
    rng = random.Random(hash(zone) & 0xFFFF)
    p = zone[:2]
    w = {"MN":[20,30,30,20],"BK":[30,30,25,15],
         "QN":[40,30,20,10],"BX":[20,25,30,25],"SI":[60,25,10,5]}.get(p,[25,25,25,25])
    return rng.choices(["Safe Corridor","Weather Sensitive",
                        "Peak Hour Hazardous","Permanently Hazardous"],weights=w)[0]

# Reads all 30 zone scores in one call and returns a list of JSON strings that gets parsed into dicts.
def read_zone_scores(r):
    pipe = r.pipeline()
    for z in ALL_ZONES: pipe.get(f"zone_score:{z}")
    results = pipe.execute()
    scores, real = {}, 0
    for z, raw in zip(ALL_ZONES, results):
        if raw:
            try: scores[z] = json.loads(raw); real += 1; continue
            except Exception: pass
        scores[z] = _synthetic_score(z)
    log.info("Zone scores: %d from Spark + %d synthetic", real, len(ALL_ZONES)-real)
    return scores

def _load_kmeans_labels():
    """Load cluster labels from saved KMeans model file."""
    labels_path = os.path.join(os.path.dirname(__file__), "models", "cluster_labels.json")
    try:
        with open(labels_path) as f:
            return json.load(f).get("zone_labels", {})
    except Exception:
        return {}

_KMEANS_LABELS = _load_kmeans_labels()

def read_cluster_labels(r):
    pipe = r.pipeline()
    for z in ALL_ZONES: pipe.get(f"cluster:{z}")
    results = pipe.execute()
    return {z: (lbl or _KMEANS_LABELS.get(z) or _synthetic_cluster(z))
            for z, lbl in zip(ALL_ZONES, results)}

def update_exposure(r, zone_scores):
    """Add RUN_INTERVAL seconds of exposure for every worker. Persist in Redis."""
    mins_this_cycle = RUN_INTERVAL / 60.0 * 5   # 5x speed: each 30s = 2.5 min exposure

    # Read current worker zones — try Spark key first, then recommender fallback
    pipe = r.pipeline()
    for i in range(1, NUM_WORKERS+1): pipe.get(f"worker_zone:W-{i:02d}")
    zone_results = pipe.execute()

    worker_zones = {}
    for i, z in enumerate(zone_results):
        wid = f"W-{i+1:02d}"
        # stable zone assignment: changes every 5 minutes but is deterministic
        slot = int(time.time() / 300)
        fallback = ALL_ZONES[(hash(wid + str(slot))) % len(ALL_ZONES)]
        worker_zones[wid] = z if (z and z in ALL_ZONES) else fallback

    # Read existing exposure — use rec_exposure: (recommender-owned) key
    # This is separate from exposure: written by Spark so they never collide
    pipe = r.pipeline()
    for i in range(1, NUM_WORKERS+1): pipe.get(f"rec_exposure:W-{i:02d}")
    exp_results = pipe.execute()

    updated = {}
    pipe = r.pipeline()
    for i, raw in enumerate(exp_results):
        wid = f"W-{i+1:02d}"
        prev = {}
        if raw:
            try: prev = json.loads(raw)
            except Exception: pass

        zone    = worker_zones[wid]
        current_aqi  = float(zone_scores.get(zone, {}).get("avg_aqi", 50.0))
        forecast_aqi = _forecast_aqi(zone, current_aqi)
        # Use the higher of current vs forecast — protective routing
        aqi = max(current_aqi, forecast_aqi)
        # Clean accumulation — prev is always our own rec_exposure: key
        h_mins   = float(prev.get("high_aqi_minutes") or 0.0)
        t_mins   = float(prev.get("total_minutes")    or 0.0)
        aqi_sum  = float(prev.get("aqi_sum")          or 0.0)
        n_cycles = int(prev.get("n_cycles")           or 0)
        h_mins += (mins_this_cycle if aqi > HIGH_AQI else 0.0)
        t_mins  += mins_this_cycle
        aqi_sum += aqi
        n_cycles += 1

        hrs_high = h_mins / 60.0
        avg_aqi  = aqi_sum / n_cycles
        status    = ("CRITICAL" if hrs_high > CRIT_HRS
                     else "WARNING" if hrs_high > WARN_HRS else "SAFE")

        rec = {
            "worker_id":         wid,
            "zone_id":           zone,
            "high_aqi_minutes":  round(h_mins, 3),
            "total_minutes":     round(t_mins, 3),
            "aqi_sum":           round(aqi_sum, 1),
            "n_cycles":          n_cycles,
            "hours_in_high_aqi": round(hrs_high, 3),
            "daily_avg_aqi":     round(avg_aqi, 1),
            "exposure_status":   status,
            "who_pct":           round(min(100, hrs_high / WHO_DAILY_HRS * 100), 1),
            "forecast_aqi":      forecast_aqi,
        }
        updated[wid] = rec
        pipe.set(f"rec_exposure:{wid}", json.dumps(rec), ex=86400)
    pipe.execute()
    return updated, worker_zones


# ═══════════════════════════════════════════════════════════════════════════════
# DISTANCE-BASED RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# For each worker, compute a distance-weighted score for every candidate zone:
#   adjusted_score = zone_safety_score / (1 + DISTANCE_DECAY * distance_km)
#
# Then filter by exposure status:
#   CRITICAL → must go to nearest Safe Corridor
#   WARNING  → nearest non-hazardous zone (avoids "Permanently Hazardous")
#   SAFE     → stay if current zone is good, else nearest better zone
#

def recommend(wid: str, exp: dict, final_scores: dict,
              cluster_labels: dict, current_zone: str,
              zone_scores: dict = None) -> dict:
    """
    Recommend the nearest good zone for a worker based on:
    - Zone safety score (from Spark)
    - Cluster label (from KMeans)
    - Distance from worker's current zone (Haversine)
    - Worker's exposure status (SAFE/WARNING/CRITICAL)
    """
    status = exp["exposure_status"]
    hours  = exp["hours_in_high_aqi"]

    # Pre-compute distances from current zone to all other zones
    distances = {z: zone_distance_km(current_zone, z) for z in ALL_ZONES}

    # Compute distance-weighted scores for each candidate zone
    weighted_scores = {}
    for z in ALL_ZONES:
        raw_score = final_scores.get(z, 0.0)
        dist = distances[z]
        # Penalty increases with distance: nearby good zones preferred
        weighted_scores[z] = raw_score / (1.0 + DISTANCE_DECAY * dist)

    # Sort by weighted score (highest first)
    ranked_by_weighted = sorted(weighted_scores.items(), key=lambda x: -x[1])

    if status == "CRITICAL":
        # CRITICAL: must go to nearest Safe Corridor — safety is non-negotiable
        safe_zones = [(z, distances[z]) for z in ALL_ZONES
                      if cluster_labels.get(z) == "Safe Corridor"]
        if safe_zones:
            # Pick the NEAREST Safe Corridor zone
            safe_zones.sort(key=lambda x: x[1])
            rz = safe_zones[0][0]
        else:
            # No Safe Corridors available — pick nearest non-hazardous
            non_haz = [(z, distances[z]) for z in ALL_ZONES
                       if cluster_labels.get(z) != "Permanently Hazardous"]
            non_haz.sort(key=lambda x: x[1])
            rz = non_haz[0][0] if non_haz else ranked_by_weighted[0][0]
        dist_km = distances[rz]
        reason = (f"CRITICAL ({hours:.1f}h) → nearest Safe Corridor "
                  f"({dist_km:.1f}km, ~{travel_time_min(dist_km):.0f}min)")

    elif status == "WARNING":
        # WARNING: go to nearest zone that isn't Permanently Hazardous
        candidates = [(z, ws) for z, ws in ranked_by_weighted
                      if cluster_labels.get(z) != "Permanently Hazardous"]
        if not candidates:
            candidates = ranked_by_weighted
        rz = candidates[0][0]
        dist_km = distances[rz]
        reason = (f"WARNING ({hours:.1f}h) → nearest safe zone "
                  f"({dist_km:.1f}km, ~{travel_time_min(dist_km):.0f}min)")

    else:
        # SAFE: try LinUCB bandit first, fall back to distance-based heuristic
        bandit_used = False
        if _bandit_router and _bandit_router.is_ready and zone_scores is not None:
            try:
                contexts = {}
                for z in ALL_ZONES:
                    zdata = zone_scores.get(z, {"avg_aqi": 50.0, "avg_speed": 30.0, "zone_id": z})
                    zdata["zone_id"] = z
                    ctx = _bandit_router.build_context(
                        worker_exp=exp, zone_data=zdata,
                        distance_km=distances[z],
                        cluster_label=cluster_labels.get(z, "Safe Corridor"),
                    )
                    contexts[z] = ctx
                rz, ucb_score = _bandit_router.select_arm(contexts)
                dist_km = distances[rz]
                reason = (f"LinUCB bandit (UCB={ucb_score:.3f}, "
                          f"{dist_km:.1f}km, ~{travel_time_min(dist_km):.0f}min)")
                bandit_used = True
            except Exception as e:
                log.debug("Bandit failed for %s: %s — falling back to heuristic", wid, e)

        if not bandit_used:
            current_score = weighted_scores.get(current_zone, 0.0)
            current_label = cluster_labels.get(current_zone, "")
            if (current_label in ("Safe Corridor", "Weather Sensitive")
                    and current_score > 0.3):
                rz = current_zone
                reason = f"Current zone is safe ({current_label}, score {current_score:.2f})"
            else:
                rz = ranked_by_weighted[0][0]
                dist_km = distances[rz]
                reason = (f"Better zone nearby "
                          f"({dist_km:.1f}km, ~{travel_time_min(dist_km):.0f}min)")

    dist_km = distances[rz]
    return {
        "worker_id":         wid,
        "zone_id":           current_zone,
        "current_zone":      current_zone,
        "rec_zone":          rz,
        "rec_label":         cluster_labels.get(rz, ""),
        "rec_score":         round(final_scores.get(rz, 0.0), 4),
        "distance_km":       round(dist_km, 2),
        "estimated_travel_min": travel_time_min(dist_km),
        "status":            status,
        "exposure_status":   status,
        "reason":            reason,
        "routing_method":    "bandit" if (status == "SAFE" and bandit_used) else "heuristic",
        "hours_in_high_aqi": round(hours, 2),
        "daily_avg_aqi":     round(exp.get("daily_avg_aqi", 50), 1),
        "who_pct":           round(min(100, hours / WHO_DAILY_HRS * 100), 1),
        "ts":                datetime.now(timezone.utc).isoformat(),
    }


def run_once(r):
    zone_scores    = read_zone_scores(r)
    cluster_labels = read_cluster_labels(r)

    # ── Forecast: update per-zone history, replace raw AQI with LSTM/AR(3)
    for zone, data in zone_scores.items():
        data["avg_aqi"] = _forecast_aqi(zone, data.get("avg_aqi", 50.0))

    worker_exp, worker_zones = update_exposure(r, zone_scores)

    final_scores = {z: round(zone_scores[z]["zone_score"] *
                              CLUSTER_WEIGHTS.get(cluster_labels[z], 1.0), 4)
                    for z in ALL_ZONES}
    ranked = sorted(final_scores.items(), key=lambda x: -x[1])

    pipe = r.pipeline()
    pipe.delete("zone_rankings")
    for z, s in ranked: pipe.zadd("zone_rankings", {z: s})
    pipe.expire("zone_rankings", 300)

    counts = {"SAFE":0,"WARNING":0,"CRITICAL":0}
    dist_stats = []
    bandit_count = 0
    for wid, exp in worker_exp.items():
        s = exp["exposure_status"]
        counts[s] += 1
        rec = recommend(wid, exp, final_scores, cluster_labels,
                        worker_zones[wid], zone_scores=zone_scores)
        dist_stats.append(rec["distance_km"])
        if rec.get("routing_method") == "bandit":
            bandit_count += 1
        pipe.set(f"rec:{wid}", json.dumps(rec), ex=300)

        # ── Bandit update: compute reward for previous recommendation ────
        if _bandit_router and s == "SAFE":
            try:
                rz = rec["rec_zone"]
                zdata = zone_scores.get(rz, {"avg_aqi": 50, "avg_speed": 30, "zone_id": rz})
                zdata["zone_id"] = rz
                ctx = _bandit_router.build_context(
                    worker_exp=exp, zone_data=zdata,
                    distance_km=rec["distance_km"],
                    cluster_label=cluster_labels.get(rz, "Safe Corridor"),
                )
                reward = _bandit_router.compute_reward(
                    zdata, rec["distance_km"], exp)
                _bandit_router.update(rz, ctx, reward)
            except Exception:
                pass

    # Save bandit stats to Redis
    if _bandit_router:
        bandit_stats = _bandit_router.get_stats()
        pipe.set("bandit_stats", json.dumps(bandit_stats), ex=300)
        # Periodically save state to disk
        if _bandit_router.n_obs % 50 == 0 and _bandit_router.n_obs > 0:
            _bandit_router.save_state()

    pipe.set("worker_status_counts", json.dumps(counts), ex=300)
    pipe.set("last_rec_ts", datetime.now().isoformat(timespec="seconds"), ex=300)
    pipe.execute()

    avg_hrs = sum(e["hours_in_high_aqi"] for e in worker_exp.values()) / NUM_WORKERS
    avg_dist = sum(dist_stats) / len(dist_stats) if dist_stats else 0.0
    log.info("Safe=%d Warning=%d Critical=%d | bandit=%d | avg_exposure=%.3fh | avg_dist=%.1fkm | top=%s(%.3f)",
             counts["SAFE"], counts["WARNING"], counts["CRITICAL"], bandit_count,
             avg_hrs, avg_dist, ranked[0][0], ranked[0][1])

def main():
    log.info("UrbanStream Recommender v5 | Redis=%s | interval=%ds | bandit=%s",
             REDIS_HOST, RUN_INTERVAL,
             "active" if _bandit_router else "disabled")
    while True:
        try:
            run_once(get_r())
        except redis.ConnectionError as e:
            log.error("Redis error: %s", e)
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
        time.sleep(RUN_INTERVAL)

if __name__ == "__main__":
    main()
