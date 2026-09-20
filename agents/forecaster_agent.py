#!/usr/bin/env python3
"""
UrbanStream — Forecaster Agent

Predicts future AQI conditions for all 30 zones using the LSTM model
from Phase 3 (with AR(3) fallback).

Perceives: current AQI readings from zone_score:{z} keys
Uses:      ForecastEngine (LSTM from Phase 3)
Decides:   predicted AQI for next 1-3 steps per zone
Acts:      sets forecast:{zone_id} keys, publishes to agent:forecaster:predictions
Interval:  30 seconds
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN", "BK", "QN", "BX", "SI"] for i in range(1, 7)]


class ForecasterAgent(BaseAgent):
    """AQI forecasting agent — predicts future conditions using LSTM."""

    def __init__(self, **kwargs):
        super().__init__(
            name="forecaster",
            role="AQI forecasting with LSTM neural network",
            run_interval_sec=30.0,
            **kwargs,
        )
        self._engine = None
        self._method = "none"
        try:
            from ml.aqi_forecaster import ForecastEngine
            self._engine = ForecastEngine()
            self._method = "LSTM" if self._engine.is_lstm_active else "AR(3)"
            log.info("[forecaster] Engine loaded: %s", self._method)
        except Exception as e:
            log.warning("[forecaster] Failed to load ForecastEngine: %s", e)

    def perceive(self, r) -> dict:
        """Read current AQI for all zones."""
        pipe = r.pipeline()
        for z in ALL_ZONES:
            pipe.get(f"zone_score:{z}")
        results = pipe.execute()

        current_aqi = {}
        for z, raw in zip(ALL_ZONES, results):
            aqi = 50.0  # default
            if raw:
                try:
                    data = json.loads(raw)
                    aqi = float(data.get("avg_aqi", 50.0))
                except Exception:
                    pass
            else:
                import random
                aqi = random.Random(hash(z) & 0xFFFF).uniform(30, 175)
            current_aqi[z] = round(aqi, 1)

        return {"current_aqi": current_aqi}

    def decide(self, observations: dict) -> dict:
        """Forecast AQI for each zone."""
        current_aqi = observations["current_aqi"]
        forecasts = {}

        for z, aqi in current_aqi.items():
            if self._engine:
                predicted = self._engine.forecast(z, aqi)
            else:
                # Simple persistence fallback
                predicted = aqi

            trend = "rising" if predicted > aqi * 1.05 else (
                    "falling" if predicted < aqi * 0.95 else "stable")

            forecasts[z] = {
                "zone_id": z,
                "current_aqi": aqi,
                "predicted_aqi": round(predicted, 1),
                "trend": trend,
                "method": self._method,
            }

        rising = sum(1 for f in forecasts.values() if f["trend"] == "rising")
        falling = sum(1 for f in forecasts.values() if f["trend"] == "falling")
        self.log_reasoning(
            f"Forecasted {len(forecasts)} zones: {rising} rising, {falling} falling",
            {"rising_zones": [z for z, f in forecasts.items() if f["trend"] == "rising"]},
        )

        return {"forecasts": forecasts}

    def act(self, r, decisions: dict) -> None:
        """Write forecasts to Redis."""
        pipe = r.pipeline()
        for z, forecast in decisions["forecasts"].items():
            pipe.set(f"forecast:{z}", json.dumps(forecast), ex=120)

        # Publish summary
        summary = {
            "method": self._method,
            "n_zones": len(decisions["forecasts"]),
            "rising": sum(1 for f in decisions["forecasts"].values()
                         if f["trend"] == "rising"),
        }
        pipe.publish("agent:forecaster:predictions", json.dumps(summary))
        pipe.execute()
