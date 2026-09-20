#!/usr/bin/env python3
"""
UrbanStream — Anomaly Detection with Isolation Forest

Detects unusual zone states that simple thresholds miss:
  - "AQI normal but speed abnormally low for this borough"
  - "Unusual speed+AQI combination for time of day"

Usage:
    python3 ml/anomaly_detector.py              # train model on historical data
    python3 ml/anomaly_detector.py --test       # test on sample data
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [ANOMALY-DETECTOR] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "ml", "models")
TRAFFIC_CSV   = os.path.join(DATA_DIR, "nyc_traffic.csv")
POLLUTION_CSV = os.path.join(DATA_DIR, "openaq_nyc.csv")
WEATHER_CSV   = os.path.join(DATA_DIR, "weather_nyc.csv")
MODEL_PATH    = os.path.join(MODEL_DIR, "anomaly_iforest.joblib")
SCALER_PATH   = os.path.join(MODEL_DIR, "anomaly_scaler.joblib")
METRICS_PATH  = os.path.join(MODEL_DIR, "anomaly_metrics.json")

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN","BK","QN","BX","SI"] for i in range(1,7)]

BOROUGH_MAP = {
    "MN": "Manhattan", "BK": "Brooklyn",
    "QN": "Queens",    "BX": "Bronx", "SI": "Staten Island",
}

# Borough-level baseline expectations (for explanation generation)
BOROUGH_BASELINES = {
    "MN": {"speed": 22.0, "aqi": 55.0, "label": "Manhattan (dense, slow, higher AQI)"},
    "BK": {"speed": 30.0, "aqi": 45.0, "label": "Brooklyn (moderate traffic)"},
    "QN": {"speed": 35.0, "aqi": 40.0, "label": "Queens (suburban, faster)"},
    "BX": {"speed": 28.0, "aqi": 50.0, "label": "Bronx (industrial pockets)"},
    "SI": {"speed": 40.0, "aqi": 35.0, "label": "Staten Island (low density)"},
}

# Feature names used by the model
FEATURE_NAMES = ["avg_speed", "avg_aqi", "avg_pm25", "density", "speed_aqi_ratio"]


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for zone conditions.

    Detects unusual combinations of speed, AQI, PM2.5, and borough density
    that simple threshold rules miss.
    """

    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self._load_model(model_path, scaler_path)

    def _load_model(self, model_path: str, scaler_path: str):
        """Load trained Isolation Forest model from disk."""
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                log.info("Anomaly model loaded from %s", model_path)
            except Exception as e:
                log.warning("Failed to load anomaly model: %s", e)
        else:
            log.warning("Anomaly model not found at %s — run anomaly_detector.py first",
                         model_path)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def _build_features(self, zone_data: dict) -> np.ndarray:
        """
        Build feature vector from zone data dict.
        Expected keys: avg_speed, avg_aqi, avg_pm25, zone_id
        """
        zone_id = zone_data.get("zone_id", "MN-01")
        density = {"MN": 1.0, "BK": 0.75, "QN": 0.60, "BX": 0.55, "SI": 0.30
                    }.get(zone_id[:2], 0.5)
        speed = float(zone_data.get("avg_speed", 30.0))
        aqi   = float(zone_data.get("avg_aqi", 50.0))
        pm25  = float(zone_data.get("avg_pm25", 10.0))
        # Derived feature: speed-to-AQI ratio (low speed + high AQI = trouble)
        speed_aqi_ratio = speed / max(aqi, 1.0)

        return np.array([speed, aqi, pm25, density, speed_aqi_ratio])

    def detect_anomalies(self, zone_features: dict[str, dict]) -> list[dict]:
        """
        Detect anomalous zones from current zone features.

        Args:
            zone_features: {zone_id: {"avg_speed": .., "avg_aqi": .., "avg_pm25": ..}}

        Returns:
            List of anomaly dicts with zone_id, anomaly_score, severity, explanation
        """
        if not self.is_ready:
            return []

        anomalies = []
        for zone_id, data in zone_features.items():
            data["zone_id"] = zone_id
            features = self._build_features(data).reshape(1, -1)
            scaled = self.scaler.transform(features)

            # score_samples returns negative values — more negative = more anomalous
            score = float(self.model.score_samples(scaled)[0])
            is_anomaly = self.model.predict(scaled)[0] == -1

            if is_anomaly:
                severity = "HIGH" if score < -0.6 else "MEDIUM"
                explanation = self._explain(zone_id, data, score)
                anomalies.append({
                    "zone_id":       zone_id,
                    "anomaly_score": round(score, 4),
                    "severity":      severity,
                    "explanation":   explanation,
                    "avg_speed":     data.get("avg_speed", 0),
                    "avg_aqi":       data.get("avg_aqi", 0),
                })

        return anomalies

    def _explain(self, zone_id: str, data: dict, score: float) -> str:
        """Generate human-readable explanation for why a zone is anomalous."""
        borough = zone_id[:2]
        baseline = BOROUGH_BASELINES.get(borough, {"speed": 30, "aqi": 50})
        speed = float(data.get("avg_speed", 30))
        aqi   = float(data.get("avg_aqi", 50))

        parts = []
        speed_pct = (speed - baseline["speed"]) / max(baseline["speed"], 1) * 100
        aqi_pct   = (aqi - baseline["aqi"]) / max(baseline["aqi"], 1) * 100

        if speed_pct < -30:
            parts.append(f"Speed {speed:.0f} km/h is {abs(speed_pct):.0f}% below "
                         f"{BOROUGH_MAP.get(borough, 'borough')} baseline ({baseline['speed']:.0f})")
        elif speed_pct > 40:
            parts.append(f"Speed {speed:.0f} km/h is unusually high for "
                         f"{BOROUGH_MAP.get(borough, 'borough')}")

        if aqi_pct > 50:
            parts.append(f"AQI {aqi:.0f} is {aqi_pct:.0f}% above normal "
                         f"({baseline['aqi']:.0f})")
        elif aqi_pct < -40 and speed < baseline["speed"] * 0.7:
            parts.append(f"AQI normal ({aqi:.0f}) but speed unusually low — "
                         f"possible incident or road closure")

        if not parts:
            parts.append(f"Unusual combination of speed ({speed:.0f}) and "
                         f"AQI ({aqi:.0f}) for {BOROUGH_MAP.get(borough, 'this area')}")

        return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _build_training_data() -> pd.DataFrame:
    """Build zone-level feature dataset from historical CSVs."""
    log.info("Building training data from historical CSVs...")

    # Traffic data
    traffic_df = pd.read_csv(TRAFFIC_CSV)
    traffic_df.columns = [c.strip().upper() for c in traffic_df.columns]
    traffic_df = traffic_df[traffic_df["SPEED"] > 0].copy()

    prefix_map = {v: k for k, v in BOROUGH_MAP.items()}
    traffic_df["borough"] = traffic_df["BOROUGH"].map(prefix_map)
    traffic_df = traffic_df.dropna(subset=["borough"])

    # Borough-level speed stats
    traffic_agg = traffic_df.groupby("borough")["SPEED"].agg(
        avg_speed="mean", std_speed="std"
    ).fillna(0)

    # Pollution data
    pollution_df = pd.read_csv(POLLUTION_CSV)
    pollution_df = pollution_df[pollution_df["value"] > 0].copy()

    # Simulate zone-level variation by creating multiple feature rows
    density_map = {"MN": 1.0, "BK": 0.75, "QN": 0.60, "BX": 0.55, "SI": 0.30}

    rows = []
    for zone in ALL_ZONES:
        borough = zone[:2]
        if borough in traffic_agg.index:
            base_speed = traffic_agg.loc[borough, "avg_speed"]
            speed_std  = traffic_agg.loc[borough, "std_speed"]
        else:
            base_speed, speed_std = 30.0, 10.0

        borough_poll = pollution_df[
            pollution_df["parameter"] == "pm25"
        ]["value"]
        if len(borough_poll) > 0:
            base_pm25 = float(borough_poll.mean())
            pm25_std  = float(borough_poll.std())
        else:
            base_pm25, pm25_std = 12.0, 5.0

        density = density_map.get(borough, 0.5)

        # Generate multiple samples per zone to train the model
        rng = np.random.default_rng(hash(zone) & 0xFFFFFFFF)
        for _ in range(50):
            speed = max(1.0, rng.normal(base_speed, speed_std * 0.5))
            pm25  = max(0.0, rng.normal(base_pm25, pm25_std * 0.5))
            aqi   = pm25 * 4.167
            speed_aqi_ratio = speed / max(aqi, 1.0)
            rows.append({
                "zone_id": zone,
                "avg_speed": round(speed, 2),
                "avg_aqi": round(aqi, 2),
                "avg_pm25": round(pm25, 2),
                "density": density,
                "speed_aqi_ratio": round(speed_aqi_ratio, 4),
            })

    df = pd.DataFrame(rows)
    log.info("Training data: %d samples × %d features", len(df), len(FEATURE_NAMES))
    return df


def train_anomaly_detector():
    """Train Isolation Forest on historical zone features."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = _build_training_data()
    X = df[FEATURE_NAMES].values

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    # contamination=0.05 means ~5% of training data is expected to be anomalous
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Evaluate on training data
    predictions = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)
    n_anomalies = int((predictions == -1).sum())
    n_normal    = int((predictions == 1).sum())

    log.info("Training results:")
    log.info("  Normal:     %d (%.1f%%)", n_normal, n_normal / len(X) * 100)
    log.info("  Anomalies:  %d (%.1f%%)", n_anomalies, n_anomalies / len(X) * 100)
    log.info("  Score range: [%.4f, %.4f]", scores.min(), scores.max())
    log.info("  Score mean:  %.4f", scores.mean())

    # Save model
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "n_samples":    len(X),
        "n_anomalies":  n_anomalies,
        "n_normal":     n_normal,
        "contamination": 0.05,
        "n_estimators":  200,
        "score_min":    round(float(scores.min()), 4),
        "score_max":    round(float(scores.max()), 4),
        "score_mean":   round(float(scores.mean()), 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model saved to %s", MODEL_PATH)
    log.info("Metrics saved to %s", METRICS_PATH)
    return model, scaler


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UrbanStream Anomaly Detection")
    parser.add_argument("--test", action="store_true",
                        help="Test detector on sample data")
    args = parser.parse_args()

    if args.test:
        detector = AnomalyDetector()
        if not detector.is_ready:
            log.error("Model not found — run training first.")
            return
        # Test with known-normal and known-anomalous data
        test_zones = {
            "MN-01": {"avg_speed": 22.0, "avg_aqi": 55.0, "avg_pm25": 13.0},  # normal for Manhattan
            "MN-02": {"avg_speed": 5.0,  "avg_aqi": 180.0, "avg_pm25": 43.0}, # anomalous
            "SI-01": {"avg_speed": 40.0, "avg_aqi": 35.0, "avg_pm25": 8.0},   # normal for SI
            "SI-02": {"avg_speed": 8.0,  "avg_aqi": 30.0, "avg_pm25": 7.0},   # anomalous (low speed for SI)
        }
        results = detector.detect_anomalies(test_zones)
        log.info("Test results: %d anomalies detected", len(results))
        for r in results:
            log.info("  %s [%s] score=%.4f: %s",
                     r["zone_id"], r["severity"], r["anomaly_score"], r["explanation"])
    else:
        train_anomaly_detector()


if __name__ == "__main__":
    main()
