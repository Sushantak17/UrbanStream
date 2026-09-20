#!/usr/bin/env python3
"""
UrbanStream — Router Agent

Assigns workers to optimal zones using the LinUCB contextual bandit
(Phase 5) combined with distance-based heuristics (Phase 2).

Perceives: zone scores, forecasts, anomalies, worker positions
Uses:      LinUCBRouter (bandit) + distance-weighted scoring
Decides:   optimal zone for each worker with reasoning
Acts:      sets rec:{worker_id} keys, publishes to agent:router:assignments
Interval:  30 seconds
"""

import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN", "BK", "QN", "BX", "SI"] for i in range(1, 7)]
NUM_WORKERS = 50

CLUSTER_WEIGHTS = {
    "Permanently Hazardous": 0.1,
    "Peak Hour Hazardous":   0.5,
    "Weather Sensitive":     0.7,
    "Safe Corridor":         1.0,
}

# ── Zone coordinates ──────────────────────────────────────────────────────────
BOROUGH_BOUNDS = {
    "MN": (40.70, 40.88, -74.02, -73.91),
    "BK": (40.57, 40.74, -74.04, -73.84),
    "QN": (40.54, 40.80, -73.97, -73.70),
    "BX": (40.80, 40.92, -73.94, -73.74),
    "SI": (40.48, 40.65, -74.26, -74.03),
}
ZONE_COORDS = {}
for _z in ALL_ZONES:
    _p = _z[:2]; _a, _b, _c, _d = BOROUGH_BOUNDS[_p]; _i = int(_z[3:]) - 1
    ZONE_COORDS[_z] = (
        round(_a + (_i // 2 + 0.5) * (_b - _a) / 3, 6),
        round(_c + (_i % 2  + 0.5) * (_d - _c) / 2, 6),
    )

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RouterAgent(BaseAgent):
    """Worker routing agent — uses LinUCB bandit + distance heuristics."""

    def __init__(self, **kwargs):
        super().__init__(
            name="router",
            role="Worker-to-zone routing with contextual bandit",
            run_interval_sec=30.0,
            **kwargs,
        )
        self._bandit = None
        try:
            from ml.bandit_router import LinUCBRouter
            self._bandit = LinUCBRouter()
            self._bandit.load_state()
            log.info("[router] LinUCB bandit loaded (n_obs=%d)", self._bandit.n_obs)
        except Exception as e:
            log.warning("[router] Bandit not available: %s", e)

    def perceive(self, r) -> dict:
        """Read zone scores, forecasts, anomalies, and worker positions."""
        pipe = r.pipeline()
        # Zone scores
        for z in ALL_ZONES:
            pipe.get(f"zone_score:{z}")
        # Cluster labels
        for z in ALL_ZONES:
            pipe.get(f"cluster:{z}")
        # Anomalies from monitor agent
        for z in ALL_ZONES:
            pipe.get(f"anomaly:{z}")
        # Forecasts from forecaster agent
        for z in ALL_ZONES:
            pipe.get(f"forecast:{z}")
        # Worker zones
        for i in range(1, NUM_WORKERS + 1):
            pipe.get(f"worker_zone:W-{i:02d}")
        # Worker exposure
        for i in range(1, NUM_WORKERS + 1):
            pipe.get(f"rec_exposure:W-{i:02d}")

        results = pipe.execute()
        idx = 0

        # Parse zone scores
        zone_scores = {}
        for z in ALL_ZONES:
            raw = results[idx]; idx += 1
            if raw:
                try: zone_scores[z] = json.loads(raw); continue
                except Exception: pass
            rng = random.Random(hash(z) & 0xFFFF)
            aqi = rng.uniform(30, 175); spd = rng.uniform(12, 68)
            zone_scores[z] = {"zone_id": z, "avg_aqi": round(aqi, 1),
                              "avg_speed": round(spd, 1),
                              "zone_score": round((min(1, spd/40)*0.5 +
                                                   (1-min(1, aqi/200))*0.5), 3)}

        # Parse clusters
        clusters = {}
        for z in ALL_ZONES:
            raw = results[idx]; idx += 1
            clusters[z] = raw or "Safe Corridor"

        # Parse anomalies
        anomalous_zones = set()
        for z in ALL_ZONES:
            raw = results[idx]; idx += 1
            if raw:
                anomalous_zones.add(z)

        # Parse forecasts
        forecasts = {}
        for z in ALL_ZONES:
            raw = results[idx]; idx += 1
            if raw:
                try: forecasts[z] = json.loads(raw)
                except Exception: pass

        # Parse worker zones + exposure
        worker_zones = {}
        for i in range(1, NUM_WORKERS + 1):
            raw = results[idx]; idx += 1
            wid = f"W-{i:02d}"
            slot = int(time.time() / 300)
            fallback = ALL_ZONES[(hash(wid + str(slot))) % len(ALL_ZONES)]
            worker_zones[wid] = raw if (raw and raw in ALL_ZONES) else fallback

        worker_exposure = {}
        for i in range(1, NUM_WORKERS + 1):
            raw = results[idx]; idx += 1
            wid = f"W-{i:02d}"
            if raw:
                try: worker_exposure[wid] = json.loads(raw); continue
                except Exception: pass
            worker_exposure[wid] = {
                "worker_id": wid, "hours_in_high_aqi": 0.0,
                "daily_avg_aqi": 50.0, "exposure_status": "SAFE",
            }

        return {
            "zone_scores": zone_scores,
            "clusters": clusters,
            "anomalous_zones": anomalous_zones,
            "forecasts": forecasts,
            "worker_zones": worker_zones,
            "worker_exposure": worker_exposure,
        }

    def decide(self, obs: dict) -> dict:
        """Compute optimal zone for each worker."""
        zone_scores = obs["zone_scores"]
        clusters = obs["clusters"]
        anomalous = obs["anomalous_zones"]
        forecasts = obs["forecasts"]
        worker_zones = obs["worker_zones"]
        worker_exp = obs["worker_exposure"]

        # Score each zone: base score × cluster weight, penalize anomalous
        final_scores = {}
        for z in ALL_ZONES:
            score = zone_scores[z].get("zone_score", 0.5)
            cw = CLUSTER_WEIGHTS.get(clusters[z], 1.0)
            final = score * cw

            # Reduce score for anomalous zones
            if z in anomalous:
                final *= 0.3

            # Reduce score for zones with rising AQI forecast
            fc = forecasts.get(z)
            if fc and fc.get("trend") == "rising":
                final *= 0.8

            final_scores[z] = round(final, 4)

        assignments = {}
        bandit_count = 0
        for wid, exposure in worker_exp.items():
            current_zone = worker_zones.get(wid, "MN-01")
            status = exposure.get("exposure_status", "SAFE")
            hours = float(exposure.get("hours_in_high_aqi", 0.0))

            # Compute distances
            clat, clon = ZONE_COORDS.get(current_zone, (40.75, -73.98))
            distances = {}
            for z in ALL_ZONES:
                zlat, zlon = ZONE_COORDS[z]
                distances[z] = round(haversine_km(clat, clon, zlat, zlon), 2)

            if status == "CRITICAL":
                # Must go to nearest Safe Corridor
                safe = [(z, distances[z]) for z in ALL_ZONES
                        if clusters.get(z) == "Safe Corridor" and z not in anomalous]
                if safe:
                    safe.sort(key=lambda x: x[1])
                    rz = safe[0][0]
                else:
                    rz = max(final_scores, key=final_scores.get)
                reason = f"CRITICAL → nearest safe zone ({distances[rz]:.1f}km)"

            elif status == "WARNING":
                # Nearest non-hazardous
                candidates = [(z, final_scores[z] / (1 + 0.3 * distances[z]))
                              for z in ALL_ZONES
                              if clusters.get(z) != "Permanently Hazardous"
                              and z not in anomalous]
                candidates.sort(key=lambda x: -x[1])
                rz = candidates[0][0] if candidates else current_zone
                reason = f"WARNING → best nearby ({distances[rz]:.1f}km)"

            else:
                # SAFE: use bandit if ready
                rz = current_zone
                reason = "Staying in current zone"
                current_score = final_scores.get(current_zone, 0)

                if current_score < 0.3 or current_zone in anomalous:
                    # Need to move — use bandit or heuristic
                    if self._bandit and self._bandit.is_ready:
                        try:
                            from ml.bandit_router import LinUCBRouter
                            contexts = {}
                            for z in ALL_ZONES:
                                if z in anomalous:
                                    continue
                                zdata = zone_scores.get(z, {"avg_aqi": 50, "avg_speed": 30})
                                zdata["zone_id"] = z
                                ctx = self._bandit.build_context(
                                    worker_exp=exposure, zone_data=zdata,
                                    distance_km=distances[z],
                                    cluster_label=clusters.get(z, "Safe Corridor"),
                                )
                                contexts[z] = ctx
                            if contexts:
                                rz, ucb = self._bandit.select_arm(contexts)
                                reason = f"LinUCB (UCB={ucb:.3f}, {distances[rz]:.1f}km)"
                                bandit_count += 1
                        except Exception:
                            pass

                    if rz == current_zone:
                        # Heuristic fallback
                        weighted = [(z, final_scores[z] / (1 + 0.3 * distances[z]))
                                    for z in ALL_ZONES if z not in anomalous]
                        weighted.sort(key=lambda x: -x[1])
                        if weighted:
                            rz = weighted[0][0]
                            reason = f"Better zone ({distances[rz]:.1f}km)"

            assignments[wid] = {
                "worker_id": wid,
                "current_zone": current_zone,
                "rec_zone": rz,
                "distance_km": distances.get(rz, 0.0),
                "reason": reason,
                "status": status,
                "routing_method": "bandit" if "LinUCB" in reason else "heuristic",
            }

        self.log_reasoning(
            f"Routed {len(assignments)} workers: {bandit_count} via bandit, "
            f"{len(anomalous)} zones avoided",
            {"bandit_count": bandit_count, "anomalous_avoided": list(anomalous)},
        )

        return {"assignments": assignments, "final_scores": final_scores}

    def act(self, r, decisions: dict) -> None:
        """Publish assignments to Redis."""
        pipe = r.pipeline()
        for wid, assignment in decisions["assignments"].items():
            pipe.set(f"rec:{wid}", json.dumps(assignment), ex=300)

        # Publish summary
        assignments = decisions["assignments"]
        summary = {
            "n_workers": len(assignments),
            "bandit_routed": sum(1 for a in assignments.values()
                                if a["routing_method"] == "bandit"),
            "critical": sum(1 for a in assignments.values()
                           if a["status"] == "CRITICAL"),
        }
        pipe.publish("agent:router:assignments", json.dumps(summary))

        # Save bandit state periodically
        if self._bandit and self._bandit.n_obs > 0 and self._bandit.n_obs % 50 == 0:
            self._bandit.save_state()

        pipe.execute()
