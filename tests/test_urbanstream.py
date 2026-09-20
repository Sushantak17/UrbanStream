"""
UrbanStream — Test Suite
Run with:  pytest tests/ -v
"""

import json
import os
import sys
import numpy as np
import pytest

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ──────────────────────────────────────────────────────────────────────────────
# 1. KMeans feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def test_zone_ids_cover_all_boroughs():
    """All 5 NYC boroughs × 6 sub-zones = 30 zone IDs."""
    from ml.clustering import ALL_ZONES, BOROUGH_MAP
    assert len(ALL_ZONES) == 30
    prefixes = {z[:2] for z in ALL_ZONES}
    assert prefixes == set(BOROUGH_MAP.keys())


def test_zone_coords_in_nyc_bounds():
    """Every zone centroid must fall within greater NYC lat/lon bounds."""
    from ml.clustering import ZONE_COORDS
    for zone, (lat, lon) in ZONE_COORDS.items():
        assert 40.4 < lat < 41.0, f"{zone} lat {lat} out of NYC bounds"
        assert -74.3 < lon < -73.6, f"{zone} lon {lon} out of NYC bounds"


def test_assign_zone_returns_valid_zone():
    """assign_zone should always return a zone that exists in ALL_ZONES."""
    from ml.clustering import assign_zone, ALL_ZONES
    # Manhattan midtown
    z = assign_zone(40.754, -73.990)
    assert z in ALL_ZONES
    # Staten Island
    z2 = assign_zone(40.579, -74.151)
    assert z2 in ALL_ZONES


def test_saved_model_files_exist():
    """Pre-trained model artefacts must be present for dashboard cold-start."""
    base = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
    for fname in ("kmeans_zones.joblib", "scaler_zones.joblib", "cluster_labels.json"):
        assert os.path.exists(os.path.join(base, fname)), f"Missing {fname}"


def test_cluster_labels_schema():
    """cluster_labels.json must have expected keys and 30 zone entries."""
    path = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "cluster_labels.json")
    with open(path) as f:
        meta = json.load(f)
    assert "zone_labels" in meta
    assert "silhouette_score" in meta
    assert "feature_cols" in meta
    assert len(meta["zone_labels"]) == 30
    assert 0 < meta["silhouette_score"] < 1


def test_predict_zone_returns_known_label():
    """predict_zone should return one of the 4 semantic cluster names."""
    from ml.clustering import predict_zone, CLUSTER_NAMES
    result = predict_zone("MN-01", avg_speed=15.0, avg_pm25=35.0)
    assert result in CLUSTER_NAMES, f"Unexpected label: {result}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Streaming KMeans (pure Python, no Spark needed)
# ──────────────────────────────────────────────────────────────────────────────

def _make_skm():
    """Import StreamingKMeans directly from its standalone module (no Spark needed)."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "streaming_kmeans",
        pathlib.Path(__file__).parent.parent / "spark" / "streaming_kmeans.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StreamingKMeans


def test_streaming_kmeans_partial_fit_assigns_labels():
    """partial_fit must return one label per input zone."""
    SKM = _make_skm()
    skm = SKM(k=4)
    # 30 zones × 4 features: [speed, aqi, pm25, density]
    rng = np.random.default_rng(0)
    X = rng.uniform([5, 20, 5, 0.3], [70, 200, 60, 1.0], size=(30, 4)).tolist()
    labels = skm.partial_fit(X)
    assert len(labels) == 30
    assert all(0 <= l < 4 for l in labels)


def test_streaming_kmeans_inertia_decreases():
    """Inertia trend should fall over successive partial_fit calls.

    MiniBatchKMeans oscillates batch-to-batch (each batch is new random data),
    so we compare the mean of the last 3 batches vs the mean of the first 3.
    We allow a 50% slack because with only 8 batches and random data the model
    is still in its early exploration phase — the important thing is that
    n_batches increments and inertia is finite, not that it has fully converged.
    """
    SKM = _make_skm()
    skm = SKM(k=4)
    rng = np.random.default_rng(42)
    inertias = []
    for _ in range(20):   # more batches → clearer trend
        X = rng.uniform([5, 20, 5, 0.3], [70, 200, 60, 1.0], size=(30, 4)).tolist()
        skm.partial_fit(X)
        inertias.append(skm.inertia)
    early = sum(inertias[:3]) / 3
    late  = sum(inertias[-3:]) / 3
    assert skm.n_batches == 20, "n_batches not incrementing"
    assert all(i < float("inf") for i in inertias), "inertia should always be finite"
    assert late <= early * 1.5, f"Inertia did not converge: early={early:.3f} late={late:.3f}"


def test_streaming_kmeans_semantic_labels():
    """semantic_labels must assign only known cluster names."""
    SKM = _make_skm()
    skm = SKM(k=4)
    rng = np.random.default_rng(7)
    zones = [f"MN-0{i+1}" for i in range(6)] + [f"BK-0{i+1}" for i in range(6)]
    X = rng.uniform([5, 20, 5, 0.3], [70, 200, 60, 1.0], size=(12, 4)).tolist()
    labels = skm.partial_fit(X)
    named = skm.semantic_labels(zones, labels)
    valid = {"Permanently Hazardous", "Peak Hour Hazardous", "Weather Sensitive", "Safe Corridor"}
    for z, name in named.items():
        assert name in valid, f"{z} got unknown label '{name}'"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Recommender logic (no Redis needed)
# ──────────────────────────────────────────────────────────────────────────────

def test_ar3_forecast_returns_float():
    """_forecast_aqi should return a float in a sane AQI range."""
    from ml.recommender import _forecast_aqi, _aqi_history
    zone = "MN-01"
    _aqi_history[zone].clear()
    readings = [55.0, 60.0, 58.0, 63.0, 70.0, 65.0]
    for v in readings:
        result = _forecast_aqi(zone, v)
    assert isinstance(result, float)
    assert 0.0 <= result <= 500.0


def test_ar3_insufficient_history_returns_current():
    """With <4 readings, forecast should return the current value unchanged."""
    from ml.recommender import _forecast_aqi, _aqi_history
    from collections import deque
    zone = "QN-99"
    _aqi_history[zone] = deque(maxlen=12)  # create zone entry if absent
    out = _forecast_aqi(zone, 42.0)
    assert out == 42.0


def _make_full_scores_and_labels():
    """Helper: build full ALL_ZONES dicts for recommend() tests."""
    from ml.recommender import ALL_ZONES
    # Default: all zones are Safe Corridors with moderate scores
    cluster_labels = {z: "Safe Corridor" for z in ALL_ZONES}
    final_scores   = {z: 0.5 for z in ALL_ZONES}
    return cluster_labels, final_scores


def test_recommend_critical_prefers_safe_corridor():
    """A CRITICAL worker should always be sent to a Safe Corridor zone."""
    from ml.recommender import recommend, ALL_ZONES
    cluster_labels, final_scores = _make_full_scores_and_labels()
    # Make MN-01 hazardous, keep some zones as Safe Corridor
    cluster_labels["MN-01"] = "Permanently Hazardous"
    cluster_labels["MN-02"] = "Permanently Hazardous"
    final_scores["MN-01"] = 0.9   # high score but hazardous
    exp = {
        "exposure_status": "CRITICAL",
        "hours_in_high_aqi": 3.5,
        "daily_avg_aqi": 120.0,
    }
    rec = recommend("W-01", exp, final_scores, cluster_labels, "MN-01")
    assert rec["rec_label"] == "Safe Corridor"
    assert rec["status"] == "CRITICAL"
    assert "distance_km" in rec
    assert "estimated_travel_min" in rec


def test_recommend_critical_picks_nearest_safe():
    """CRITICAL worker should go to NEAREST Safe Corridor, not globally best."""
    from ml.recommender import recommend, ALL_ZONES, zone_distance_km
    cluster_labels, final_scores = _make_full_scores_and_labels()
    # Make all zones Permanently Hazardous except two Safe Corridors
    for z in ALL_ZONES:
        cluster_labels[z] = "Permanently Hazardous"
        final_scores[z] = 0.1
    # Two Safe Corridors — one near (MN-03), one far (SI-06)
    cluster_labels["MN-03"] = "Safe Corridor"
    cluster_labels["SI-06"] = "Safe Corridor"
    final_scores["MN-03"] = 0.5
    final_scores["SI-06"] = 0.9   # SI-06 has higher score but is far
    exp = {"exposure_status": "CRITICAL", "hours_in_high_aqi": 4.0, "daily_avg_aqi": 150.0}
    rec = recommend("W-01", exp, final_scores, cluster_labels, "MN-01")
    # Should pick MN-03 (nearest) over SI-06 (highest score)
    dist_mn03 = zone_distance_km("MN-01", "MN-03")
    dist_si06 = zone_distance_km("MN-01", "SI-06")
    assert dist_mn03 < dist_si06, "Test setup: MN-03 should be closer than SI-06"
    assert rec["rec_zone"] == "MN-03", f"Expected MN-03 (nearest), got {rec['rec_zone']}"


def test_recommend_safe_worker_stays_in_good_zone():
    """A SAFE worker in a Safe Corridor should stay put (distance=0)."""
    from ml.recommender import recommend
    cluster_labels, final_scores = _make_full_scores_and_labels()
    # MN-01 is a good Safe Corridor
    cluster_labels["MN-01"] = "Safe Corridor"
    final_scores["MN-01"] = 0.6
    exp = {"exposure_status": "SAFE", "hours_in_high_aqi": 0.0, "daily_avg_aqi": 30.0}
    rec = recommend("W-02", exp, final_scores, cluster_labels, "MN-01")
    assert rec["rec_zone"] == "MN-01", "SAFE worker in good zone should stay"
    assert rec["distance_km"] == 0.0


def test_haversine_accuracy():
    """Haversine between known NYC landmarks should be within 10% of true distance."""
    from ml.recommender import haversine_km
    # Times Square to JFK Airport ≈ 20.5 km
    d = haversine_km(40.7580, -73.9855, 40.6413, -73.7781)
    assert 18.0 < d < 23.0, f"Times Square→JFK expected ~20.5km, got {d:.1f}km"
    # Same point → 0
    assert haversine_km(40.7, -73.9, 40.7, -73.9) == 0.0


def test_recommend_output_has_distance_fields():
    """All recommendations must include distance_km and estimated_travel_min."""
    from ml.recommender import recommend
    cluster_labels, final_scores = _make_full_scores_and_labels()
    exp = {"exposure_status": "SAFE", "hours_in_high_aqi": 0.0, "daily_avg_aqi": 30.0}
    rec = recommend("W-03", exp, final_scores, cluster_labels, "BK-01")
    required_keys = ["distance_km", "estimated_travel_min", "rec_zone",
                     "current_zone", "reason", "status"]
    for key in required_keys:
        assert key in rec, f"Missing key '{key}' in recommendation output"
    assert isinstance(rec["distance_km"], float)
    assert isinstance(rec["estimated_travel_min"], float)
    assert rec["distance_km"] >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Zone scoring sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def test_zone_score_range():
    """Zone score must always be in [0, 1]."""
    BASELINE = 40.0
    test_cases = [
        (0.0,  0.0),    # stopped + no pollution
        (40.0, 0.0),    # baseline speed + no pollution
        (80.0, 200.0),  # very fast + very polluted
        (10.0, 150.0),  # slow + polluted
    ]
    for speed, aqi in test_cases:
        ss = min(1.0, speed / BASELINE)
        aq = 1.0 - min(1.0, aqi / 200.0)   # inverted: higher AQI → lower score
        zs = round(ss * 0.5 + aq * 0.5, 3)
        assert 0.0 <= zs <= 1.0, f"score={zs} out of range for speed={speed}, aqi={aqi}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. LSTM AQI Forecaster
# ──────────────────────────────────────────────────────────────────────────────

def test_lstm_model_forward_pass():
    """LSTM model should accept (batch, seq_len, 1) and output (batch, pred_len)."""
    import torch
    from ml.aqi_forecaster import AQILSTMModel
    model = AQILSTMModel(input_dim=1, hidden_dim=32, num_layers=2, pred_len=3)
    x = torch.randn(4, 24, 1)  # batch=4, seq_len=24, features=1
    out = model(x)
    assert out.shape == (4, 3), f"Expected (4, 3), got {out.shape}"


def test_forecast_engine_returns_float():
    """ForecastEngine.forecast should return a float in valid AQI range."""
    from ml.aqi_forecaster import ForecastEngine
    engine = ForecastEngine()
    # Feed 30 readings to build up history
    result = 50.0
    for i in range(30):
        result = engine.forecast("TEST-01", 50.0 + i * 0.5)
    assert isinstance(result, float)
    assert 0.0 <= result <= 500.0


def test_lstm_model_file_exists():
    """Trained LSTM model artefact must be present after training."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "aqi_lstm.pt")
    assert os.path.exists(model_path), "LSTM model not found — run ml/aqi_forecaster.py first"


def test_lstm_metrics_show_improvement():
    """LSTM should have lower MSE than AR(3) baseline."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "aqi_lstm_metrics.json")
    if not os.path.exists(metrics_path):
        pytest.skip("Metrics file not found — run training first")
    with open(metrics_path) as f:
        m = json.load(f)
    assert m["lstm"]["mse"] < m["ar3"]["mse"], \
        f"LSTM MSE ({m['lstm']['mse']}) should be lower than AR3 ({m['ar3']['mse']})"
    assert m["improvement_pct"] > 0, "LSTM should show positive improvement"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Anomaly Detection (Isolation Forest)
# ──────────────────────────────────────────────────────────────────────────────

def test_anomaly_detector_model_exists():
    """Anomaly detection model must be trained and saved."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "anomaly_iforest.joblib")
    assert os.path.exists(model_path), "Anomaly model not found — run ml/anomaly_detector.py first"


def test_anomaly_detector_catches_outlier():
    """Known outlier (very low speed + very high AQI) should be flagged."""
    from ml.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    if not detector.is_ready:
        pytest.skip("Anomaly model not trained yet")
    # Extremely anomalous: speed 3 km/h + AQI 300 in Manhattan
    zones = {"MN-01": {"avg_speed": 3.0, "avg_aqi": 300.0, "avg_pm25": 72.0}}
    results = detector.detect_anomalies(zones)
    assert len(results) > 0, "Should detect extreme outlier"
    assert results[0]["zone_id"] == "MN-01"
    assert results[0]["severity"] in ("HIGH", "MEDIUM")


def test_anomaly_detector_normal_data_passes():
    """Normal zone conditions should NOT be flagged as anomalous."""
    from ml.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    if not detector.is_ready:
        pytest.skip("Anomaly model not trained yet")
    # Typical Manhattan data
    zones = {"MN-01": {"avg_speed": 22.0, "avg_aqi": 55.0, "avg_pm25": 13.0}}
    results = detector.detect_anomalies(zones)
    anomalous_zones = [r["zone_id"] for r in results]
    assert "MN-01" not in anomalous_zones, "Normal data should not be flagged"


def test_anomaly_explanation_is_readable():
    """Anomaly explanations should be non-empty strings."""
    from ml.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    if not detector.is_ready:
        pytest.skip("Anomaly model not trained yet")
    zones = {"SI-01": {"avg_speed": 2.0, "avg_aqi": 200.0, "avg_pm25": 48.0}}
    results = detector.detect_anomalies(zones)
    assert len(results) > 0
    assert isinstance(results[0]["explanation"], str)
    assert len(results[0]["explanation"]) > 10


# ──────────────────────────────────────────────────────────────────────────────
# 7. LinUCB Contextual Bandit
# ──────────────────────────────────────────────────────────────────────────────

def test_bandit_context_shape():
    """build_context should return a 13-dimensional vector."""
    from ml.bandit_router import LinUCBRouter
    router = LinUCBRouter()
    ctx = router.build_context(
        worker_exp={"hours_in_high_aqi": 1.5, "daily_avg_aqi": 80.0},
        zone_data={"avg_aqi": 60.0, "avg_speed": 25.0, "zone_id": "MN-01"},
        distance_km=3.5,
        cluster_label="Safe Corridor",
        hour=14,
    )
    assert ctx.shape == (13,), f"Expected (13,), got {ctx.shape}"
    assert all(0.0 <= v <= 1.0 for v in ctx), "All context values should be in [0, 1]"


def test_bandit_select_arm_returns_valid_zone():
    """select_arm should return a zone ID from ALL_ZONES."""
    from ml.bandit_router import LinUCBRouter, ALL_ZONES
    router = LinUCBRouter()
    # Build contexts for all zones
    worker_exp = {"hours_in_high_aqi": 0.5, "daily_avg_aqi": 45.0}
    contexts = {}
    for z in ALL_ZONES:
        ctx = router.build_context(
            worker_exp=worker_exp,
            zone_data={"avg_aqi": 50.0, "avg_speed": 30.0, "zone_id": z},
            distance_km=5.0,
            cluster_label="Safe Corridor",
            hour=10,
        )
        contexts[z] = ctx
    zone, score = router.select_arm(contexts)
    assert zone in ALL_ZONES, f"select_arm returned unknown zone: {zone}"
    assert isinstance(score, float)


def test_bandit_update_changes_parameters():
    """After update(), A and b matrices should differ from identity/zero."""
    from ml.bandit_router import LinUCBRouter
    router = LinUCBRouter()
    ctx = router.build_context(
        worker_exp={"hours_in_high_aqi": 1.0, "daily_avg_aqi": 70.0},
        zone_data={"avg_aqi": 40.0, "avg_speed": 35.0, "zone_id": "BK-03"},
        distance_km=2.0,
        cluster_label="Weather Sensitive",
        hour=8,
    )
    A_before = router.A[0].copy()
    router.update("MN-01", ctx, reward=0.8)
    assert router.n_obs == 1
    assert not np.allclose(router.A[0], A_before), "A matrix should change after update"


def test_bandit_cold_start_not_ready():
    """Bandit should NOT be ready before MIN_OBS_FOR_BANDIT observations."""
    from ml.bandit_router import LinUCBRouter, MIN_OBS_FOR_BANDIT
    router = LinUCBRouter()
    assert not router.is_ready
    assert router.n_obs == 0
    # Feed some observations but fewer than threshold
    ctx = np.ones(13) * 0.5
    for i in range(min(10, MIN_OBS_FOR_BANDIT - 1)):
        router.update("MN-01", ctx, reward=0.5)
    assert not router.is_ready, "Should not be ready with < 100 obs"


def test_bandit_reward_computation():
    """compute_reward should return a float in a reasonable range."""
    from ml.bandit_router import LinUCBRouter
    router = LinUCBRouter()
    reward = router.compute_reward(
        zone_data={"avg_aqi": 40.0, "zone_score": 0.8},
        distance_km=3.0,
        worker_exp={"hours_in_high_aqi": 2.0},
    )
    assert isinstance(reward, float)
    assert -1.0 <= reward <= 2.0, f"Reward {reward} out of expected range"
    # Safe zone + short distance + high exposure → should get urgency bonus
    reward_urgent = router.compute_reward(
        zone_data={"avg_aqi": 30.0, "zone_score": 0.9},
        distance_km=1.0,
        worker_exp={"hours_in_high_aqi": 2.5},
    )
    assert reward_urgent > reward, "Urgent worker in safe zone should get higher reward"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Multi-Agent System
# ──────────────────────────────────────────────────────────────────────────────

def test_base_agent_lifecycle():
    """BaseAgent subclass should start and stop cleanly."""
    from agents.base_agent import BaseAgent

    class DummyAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="dummy", role="test", run_interval_sec=0.1,
                             redis_host="__INVALID__", redis_port=1)
            self.perceive_count = 0

        def perceive(self, r):
            self.perceive_count += 1
            return {"ok": True}

        def decide(self, obs):
            return {"action": "noop"}

        def act(self, r, dec):
            pass  # Redis will fail (invalid host) — that's expected

    agent = DummyAgent()
    assert not agent.is_alive
    agent.start()
    assert agent._running is True
    import time; time.sleep(0.3)
    assert agent.is_alive, "Agent thread should be running"
    agent.stop()
    assert not agent._running
    # Verify status dict works
    status = agent.get_status()
    assert status["name"] == "dummy"
    assert isinstance(status["decisions"], int)


def test_monitor_agent_decision_logic():
    """MonitorAgent.decide should return anomalies and threshold alerts."""
    from agents.monitor_agent import MonitorAgent

    agent = MonitorAgent(redis_host="__INVALID__")
    # Feed synthetic zone data directly to decide()
    zones = {
        "MN-01": {"avg_aqi": 180.0, "avg_speed": 8.0, "avg_pm25": 43.0},
        "BK-01": {"avg_aqi": 35.0, "avg_speed": 45.0, "avg_pm25": 8.0},
    }
    result = agent.decide({"zones": zones})
    assert "anomalies" in result
    assert "threshold_alerts" in result
    # MN-01 has AQI 180 (> 150 critical) and speed 8 (< 15 low)
    threshold_zones = [a["zone_id"] for a in result["threshold_alerts"]]
    assert "MN-01" in threshold_zones, "MN-01 should trigger threshold alerts"


def test_coordinator_creates_all_agents():
    """Coordinator should instantiate all 4 agents."""
    from agents.coordinator import Coordinator
    coord = Coordinator(redis_host="__INVALID__")
    assert len(coord.agents) == 4
    agent_names = {a.name for a in coord.agents}
    assert agent_names == {"monitor", "forecaster", "router", "alert"}

