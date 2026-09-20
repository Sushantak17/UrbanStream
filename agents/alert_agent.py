#!/usr/bin/env python3
"""
UrbanStream — Alert Agent

Generates human-readable safety briefings and escalation alerts
for workers based on routing decisions and exposure levels.

Perceives: router assignments, worker exposure, monitor alerts
Decides:   which workers need immediate attention, briefing content
Acts:      sets briefing:{worker_id} keys, publishes to agent:alert:briefings
Interval:  30 seconds
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

NUM_WORKERS = 50
ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN", "BK", "QN", "BX", "SI"] for i in range(1, 7)]
BOROUGH = {"MN": "Manhattan", "BK": "Brooklyn", "QN": "Queens", "BX": "Bronx", "SI": "Staten Island"}


class AlertAgent(BaseAgent):
    """Safety briefing agent — generates human-readable alerts for workers."""

    def __init__(self, **kwargs):
        super().__init__(
            name="alert",
            role="Safety briefings and escalation alerts",
            run_interval_sec=30.0,
            **kwargs,
        )

    def perceive(self, r) -> dict:
        """Read routing assignments and worker exposure."""
        pipe = r.pipeline()
        # Read recommendations
        for i in range(1, NUM_WORKERS + 1):
            pipe.get(f"rec:W-{i:02d}")
        # Read exposure
        for i in range(1, NUM_WORKERS + 1):
            pipe.get(f"rec_exposure:W-{i:02d}")
        results = pipe.execute()

        recs = {}
        for i in range(NUM_WORKERS):
            wid = f"W-{i+1:02d}"
            raw = results[i]
            if raw:
                try: recs[wid] = json.loads(raw)
                except Exception: pass

        exposures = {}
        for i in range(NUM_WORKERS):
            wid = f"W-{i+1:02d}"
            raw = results[NUM_WORKERS + i]
            if raw:
                try: exposures[wid] = json.loads(raw)
                except Exception: pass

        return {"recs": recs, "exposures": exposures}

    def decide(self, observations: dict) -> dict:
        """Generate briefings for workers based on severity."""
        recs = observations["recs"]
        exposures = observations["exposures"]
        briefings = {}
        urgent_count = 0

        for wid in [f"W-{i:02d}" for i in range(1, NUM_WORKERS + 1)]:
            rec = recs.get(wid, {})
            exp = exposures.get(wid, {})

            status = rec.get("status", exp.get("exposure_status", "SAFE"))
            hours = float(exp.get("hours_in_high_aqi", rec.get("hours_in_high_aqi", 0.0)))
            current = rec.get("current_zone", exp.get("zone_id", "MN-01"))
            rec_zone = rec.get("rec_zone", current)
            dist = rec.get("distance_km", 0.0)
            reason = rec.get("reason", "No recommendation available")

            borough = BOROUGH.get(rec_zone[:2], "NYC")

            # Generate briefing based on severity
            if status == "CRITICAL":
                icon = "🚨"
                priority = "CRITICAL"
                message = (f"{icon} {wid}: {hours:.1f}h in high-AQI zones — "
                          f"MOVE to {rec_zone} ({borough}, {dist:.1f}km away). "
                          f"Exceeds 3h safety limit. {reason}")
                urgent_count += 1

            elif status == "WARNING":
                icon = "⚠️"
                priority = "WARNING"
                remaining = max(0, 3.0 - hours)
                message = (f"{icon} {wid}: {hours:.1f}h exposure, "
                          f"{remaining:.1f}h until critical. "
                          f"Recommended: {rec_zone} ({borough}). {reason}")
                urgent_count += 1

            else:
                icon = "✅"
                priority = "INFO"
                if current != rec_zone:
                    message = (f"{icon} {wid}: Safe ({hours:.1f}h). "
                              f"Consider moving to {rec_zone} for better conditions.")
                else:
                    message = f"{icon} {wid}: Safe in {current} ({borough}). No action needed."

            briefings[wid] = {
                "worker_id": wid,
                "priority": priority,
                "message": message,
                "current_zone": current,
                "rec_zone": rec_zone,
                "hours_exposed": round(hours, 2),
                "ts": datetime.now(timezone.utc).isoformat(),
            }

        self.log_reasoning(
            f"Generated {len(briefings)} briefings: {urgent_count} urgent",
            {"critical": sum(1 for b in briefings.values() if b["priority"] == "CRITICAL"),
             "warning": sum(1 for b in briefings.values() if b["priority"] == "WARNING")},
        )

        return {"briefings": briefings}

    def act(self, r, decisions: dict) -> None:
        """Publish briefings to Redis."""
        pipe = r.pipeline()
        for wid, briefing in decisions["briefings"].items():
            pipe.set(f"briefing:{wid}", json.dumps(briefing), ex=120)

        # Publish summary
        briefings = decisions["briefings"]
        summary = {
            "total": len(briefings),
            "critical": sum(1 for b in briefings.values() if b["priority"] == "CRITICAL"),
            "warning": sum(1 for b in briefings.values() if b["priority"] == "WARNING"),
        }
        pipe.publish("agent:alert:briefings", json.dumps(summary))
        pipe.execute()
