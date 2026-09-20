#!/usr/bin/env python3
"""
UrbanStream — LinUCB Contextual Bandit for Worker Routing

Implements a Linear Upper Confidence Bound (LinUCB) bandit that learns
optimal zone assignments from experience:
  - Context: worker exposure, zone features, distance, time-of-day
  - Arms: 30 zones (one per arm)
  - Reward: safety_score - distance_penalty

The bandit starts in exploration mode and gradually learns which zones
are best for each worker context. Falls back to distance-based heuristic
during cold start (<100 observations).

Reference: Li et al., "A Contextual-Bandit Approach to Personalized
           News Article Recommendation", WWW 2010.
"""

import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [BANDIT-ROUTER] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

ALL_ZONES = [f"{p}-{i:02d}" for p in ["MN","BK","QN","BX","SI"] for i in range(1,7)]
N_ARMS    = len(ALL_ZONES)  # 30 zones = 30 arms
ZONE_IDX  = {z: i for i, z in enumerate(ALL_ZONES)}

# LinUCB hyperparameters
ALPHA     = 1.0    # exploration parameter (higher = more exploration)
ALPHA_MIN = 0.1    # minimum alpha after decay
ALPHA_DECAY = 0.999  # decay per observation
MIN_OBS_FOR_BANDIT = 100  # cold start threshold

# Context feature dimension
# [exposure_hours, current_aqi, zone_aqi, zone_speed, distance_km,
#  density, hour_sin, hour_cos, is_rush, cluster_safe, cluster_weather,
#  cluster_peak, cluster_hazardous]
N_FEATURES = 13

# Cluster one-hot mapping
CLUSTER_TO_IDX = {
    "Safe Corridor":          0,
    "Weather Sensitive":      1,
    "Peak Hour Hazardous":    2,
    "Permanently Hazardous":  3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LinUCB ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class LinUCBRouter:
    """
    LinUCB contextual bandit for worker-to-zone routing.

    Maintains per-arm (per-zone) A and b matrices:
      A_a: (d×d) matrix for arm a
      b_a: (d×1) vector for arm a

    At decision time:
      θ_a = A_a^{-1} b_a
      p_a = θ_a^T x + α √(x^T A_a^{-1} x)
      Choose arm with highest p_a (UCB score)
    """

    def __init__(self, n_features: int = N_FEATURES, n_arms: int = N_ARMS,
                 alpha: float = ALPHA):
        self.n_features = n_features
        self.n_arms     = n_arms
        self.alpha      = alpha
        self.n_obs      = 0
        self._cumulative_reward = 0.0

        # Per-arm parameters
        self.A = np.array([np.eye(n_features) for _ in range(n_arms)])  # (n_arms, d, d)
        self.b = np.zeros((n_arms, n_features))                         # (n_arms, d)

    def build_context(self, worker_exp: dict, zone_data: dict,
                      distance_km: float, cluster_label: str,
                      hour: int = -1) -> np.ndarray:
        """
        Build context feature vector for a (worker, zone) pair.

        Args:
            worker_exp: worker exposure dict (hours_in_high_aqi, daily_avg_aqi, etc.)
            zone_data: zone score dict (avg_speed, avg_aqi, etc.)
            distance_km: distance from worker's current zone to this zone
            cluster_label: semantic cluster label of this zone
            hour: hour of day (0-23), auto-detected if -1
        """
        if hour < 0:
            hour = time.localtime().tm_hour

        exposure_hours = float(worker_exp.get("hours_in_high_aqi", 0.0))
        worker_aqi     = float(worker_exp.get("daily_avg_aqi", 50.0))
        zone_aqi       = float(zone_data.get("avg_aqi", 50.0))
        zone_speed     = float(zone_data.get("avg_speed", 30.0))
        density_map    = {"MN": 1.0, "BK": 0.75, "QN": 0.60, "BX": 0.55, "SI": 0.30}
        zone_id        = zone_data.get("zone_id", "MN-01")
        density        = density_map.get(zone_id[:2], 0.5)

        # Time features (cyclical encoding)
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        is_rush  = 1.0 if hour in (7, 8, 9, 17, 18, 19) else 0.0

        # Cluster one-hot (4 categories)
        cluster_onehot = [0.0, 0.0, 0.0, 0.0]
        cidx = CLUSTER_TO_IDX.get(cluster_label, 0)
        cluster_onehot[cidx] = 1.0

        # Normalise continuous features to [0, 1]
        ctx = np.array([
            min(1.0, exposure_hours / 8.0),    # exposure hours / WHO limit
            min(1.0, worker_aqi / 200.0),       # worker's avg AQI
            min(1.0, zone_aqi / 200.0),          # zone AQI
            min(1.0, zone_speed / 80.0),         # zone speed
            min(1.0, distance_km / 30.0),        # distance (max ~30km in NYC)
            density,                              # borough density
            (hour_sin + 1) / 2,                   # normalised to [0,1]
            (hour_cos + 1) / 2,                   # normalised to [0,1]
            is_rush,                              # binary
            cluster_onehot[0],                    # Safe Corridor
            cluster_onehot[1],                    # Weather Sensitive
            cluster_onehot[2],                    # Peak Hour Hazardous
            cluster_onehot[3],                    # Permanently Hazardous
        ], dtype=np.float64)

        assert len(ctx) == self.n_features, f"Context dim mismatch: {len(ctx)} vs {self.n_features}"
        return ctx

    def select_arm(self, contexts: dict[str, np.ndarray]) -> tuple[str, float]:
        """
        Select best zone (arm) given contexts for all candidate zones.

        Args:
            contexts: {zone_id: context_vector} for all candidate zones

        Returns:
            (best_zone_id, ucb_score)
        """
        best_zone = None
        best_score = -float("inf")

        for zone_id, x in contexts.items():
            arm_idx = ZONE_IDX.get(zone_id, 0)
            A_inv = np.linalg.inv(self.A[arm_idx])
            theta = A_inv @ self.b[arm_idx]

            # UCB score = predicted reward + exploration bonus
            pred = float(theta @ x)
            exploration = self.alpha * math.sqrt(float(x @ A_inv @ x))
            score = pred + exploration

            if score > best_score:
                best_score = score
                best_zone = zone_id

        return best_zone, best_score

    def update(self, zone_id: str, context: np.ndarray, reward: float):
        """
        Update model with observed reward for a (context, arm) pair.

        Args:
            zone_id: the zone that was recommended
            context: the context vector used for the recommendation
            reward: observed reward (higher = better)
        """
        arm_idx = ZONE_IDX.get(zone_id, 0)
        self.A[arm_idx] += np.outer(context, context)
        self.b[arm_idx] += reward * context
        self.n_obs += 1
        self._cumulative_reward += reward

        # Decay exploration parameter
        self.alpha = max(ALPHA_MIN, self.alpha * ALPHA_DECAY)

    def compute_reward(self, zone_data: dict, distance_km: float,
                        worker_exp: dict) -> float:
        """
        Compute reward for routing a worker to a zone.
        Reward = safety_score - distance_penalty

        Higher reward for:
          - Low AQI zones (worker safety)
          - Close zones (minimal travel)
          - Zones where worker's exposure would decrease
        """
        aqi   = float(zone_data.get("avg_aqi", 50.0))
        score = float(zone_data.get("zone_score", 0.5))

        # Safety component: higher zone score = better (already inverted in Spark)
        safety = score

        # Distance penalty: closer is better
        dist_penalty = 0.1 * min(1.0, distance_km / 20.0)

        # Exposure urgency: reward increases for high-exposure workers going to safe zones
        exposure_hrs = float(worker_exp.get("hours_in_high_aqi", 0.0))
        urgency_bonus = 0.0
        if exposure_hrs > 1.5 and aqi < 50:
            urgency_bonus = 0.2 * min(1.0, exposure_hrs / 3.0)

        return round(safety - dist_penalty + urgency_bonus, 4)

    @property
    def is_ready(self) -> bool:
        """Whether the bandit has enough observations to make good decisions."""
        return self.n_obs >= MIN_OBS_FOR_BANDIT

    @property
    def cumulative_reward(self) -> float:
        return round(self._cumulative_reward, 4)

    @property
    def avg_reward(self) -> float:
        return round(self._cumulative_reward / max(1, self.n_obs), 4)

    def get_stats(self) -> dict:
        """Get bandit statistics for dashboard display."""
        return {
            "n_observations":    self.n_obs,
            "cumulative_reward": self.cumulative_reward,
            "avg_reward":        self.avg_reward,
            "alpha":             round(self.alpha, 4),
            "is_ready":          self.is_ready,
            "n_arms":            self.n_arms,
            "n_features":        self.n_features,
        }

    def save_state(self, path: Optional[str] = None):
        """Save bandit state to disk."""
        path = path or os.path.join(MODEL_DIR, "bandit_state.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 A=self.A, b=self.b,
                 n_obs=self.n_obs,
                 cumulative_reward=self._cumulative_reward,
                 alpha=self.alpha)
        log.info("Bandit state saved to %s (n_obs=%d)", path, self.n_obs)

    def load_state(self, path: Optional[str] = None):
        """Load bandit state from disk."""
        path = path or os.path.join(MODEL_DIR, "bandit_state.npz")
        if not os.path.exists(path):
            log.info("No bandit state found at %s — starting fresh", path)
            return False
        try:
            data = np.load(path)
            self.A = data["A"]
            self.b = data["b"]
            self.n_obs = int(data["n_obs"])
            self._cumulative_reward = float(data["cumulative_reward"])
            self.alpha = float(data["alpha"])
            log.info("Bandit state loaded from %s (n_obs=%d, alpha=%.4f)",
                     path, self.n_obs, self.alpha)
            return True
        except Exception as e:
            log.warning("Failed to load bandit state: %s", e)
            return False
