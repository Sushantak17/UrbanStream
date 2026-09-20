#!/usr/bin/env python3
"""
UrbanStream — Monitor Agent

Watches all 30 zones for anomalies and threshold violations using
the Isolation Forest model from Phase 4.

Perceives: zone_score:{z} keys from Redis
Uses:      AnomalyDetector (Isolation Forest)
Decides:   which zones are anomalous, severity level
Acts:      publishes to agent:monitor:alerts, sets anomaly:{zone_id} keys
Interval:  15 seconds
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN", "BK", "QN", "BX", "SI"] for i in range(1, 7)]

# AQI thresholds for rule-based alerts (complement anomaly detection)
AQI_WARNING  = 100.0
AQI_CRITICAL = 150.0
SPEED_LOW    = 15.0


class MonitorAgent(BaseAgent):
    """Anomaly detection agent — watches zone conditions for outliers."""

    def __init__(self, **kwargs):
        super().__init__(
            name="monitor",
            role="Zone anomaly detection and threshold monitoring",
            run_interval_sec=15.0,
            **kwargs,
        )
        # Load the anomaly detector
        self._detector = None
        try:
            from ml.anomaly_detector import AnomalyDetector
            self._detector = AnomalyDetector()
            if self._detector.is_ready:
                log.info("[monitor] Anomaly detector loaded (Isolation Forest)")
            else:
                log.warning("[monitor] Anomaly detector model not trained yet")
        except Exception as e:
            log.warning("[monitor] Failed to load anomaly detector: %s", e)

    def perceive(self, r) -> dict:
        """Read all 30 zone scores from Redis."""
        pipe = r.pipeline()
        for z in ALL_ZONES:
            pipe.get(f"zone_score:{z}")
        results = pipe.execute()

        zones = {}
        for z, raw in zip(ALL_ZONES, results):
            if raw:
                try:
                    zones[z] = json.loads(raw)
                    continue
                except Exception:
                    pass
            # Synthetic fallback
            import random
            rng = random.Random(hash(z) & 0xFFFF)
            aqi = rng.uniform(30, 175)
            spd = rng.uniform(12, 68)
            zones[z] = {
                "zone_id": z, "avg_aqi": round(aqi, 1),
                "avg_speed": round(spd, 1), "avg_pm25": round(aqi * 0.24, 1),
                "zone_score": round((min(1, spd / 40) * 0.5 +
                                     (1 - min(1, aqi / 200)) * 0.5), 3),
            }
        return {"zones": zones}

    def decide(self, observations: dict) -> dict:
        """Run anomaly detection + threshold checks."""
        zones = observations["zones"]
        anomalies = []
        threshold_alerts = []

        # 1. ML-based anomaly detection (Isolation Forest)
        if self._detector and self._detector.is_ready:
            zone_features = {
                z: {"avg_speed": d.get("avg_speed", 30),
                    "avg_aqi": d.get("avg_aqi", 50),
                    "avg_pm25": d.get("avg_pm25", 12)}
                for z, d in zones.items()
            }
            anomalies = self._detector.detect_anomalies(zone_features)

        # 2. Rule-based threshold alerts
        for z, data in zones.items():
            aqi = data.get("avg_aqi", 0)
            speed = data.get("avg_speed", 50)
            if aqi > AQI_CRITICAL:
                threshold_alerts.append({
                    "zone_id": z, "type": "AQI_CRITICAL",
                    "value": aqi, "threshold": AQI_CRITICAL,
                    "message": f"{z}: AQI {aqi:.0f} exceeds critical threshold {AQI_CRITICAL:.0f}",
                })
            elif aqi > AQI_WARNING:
                threshold_alerts.append({
                    "zone_id": z, "type": "AQI_WARNING",
                    "value": aqi, "threshold": AQI_WARNING,
                    "message": f"{z}: AQI {aqi:.0f} above warning level",
                })
            if speed < SPEED_LOW:
                threshold_alerts.append({
                    "zone_id": z, "type": "SPEED_LOW",
                    "value": speed, "threshold": SPEED_LOW,
                    "message": f"{z}: Speed {speed:.1f} km/h is dangerously low",
                })

        n_anom = len(anomalies)
        n_thresh = len(threshold_alerts)
        self.log_reasoning(
            f"Scanned {len(zones)} zones: {n_anom} ML anomalies, {n_thresh} threshold alerts",
            {"anomaly_zones": [a["zone_id"] for a in anomalies],
             "threshold_zones": [a["zone_id"] for a in threshold_alerts]},
        )

        return {"anomalies": anomalies, "threshold_alerts": threshold_alerts}

    def act(self, r, decisions: dict) -> None:
        """Publish anomaly alerts to Redis."""
        pipe = r.pipeline()

        # Set individual anomaly keys
        all_alerts = decisions["anomalies"] + decisions["threshold_alerts"]
        alerted_zones = set()
        for alert in all_alerts:
            zid = alert["zone_id"]
            if zid not in alerted_zones:
                pipe.set(f"anomaly:{zid}", json.dumps(alert), ex=60)
                alerted_zones.add(zid)

        # Clear anomaly keys for zones that are now normal
        for z in ALL_ZONES:
            if z not in alerted_zones:
                pipe.delete(f"anomaly:{z}")

        # Publish summary to pub/sub channel
        summary = {
            "n_anomalies": len(decisions["anomalies"]),
            "n_threshold_alerts": len(decisions["threshold_alerts"]),
            "affected_zones": list(alerted_zones),
        }
        pipe.publish("agent:monitor:alerts", json.dumps(summary))
        pipe.execute()
