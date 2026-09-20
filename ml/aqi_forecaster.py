#!/usr/bin/env python3
"""
UrbanStream — LSTM AQI Forecasting Model

Replaces the simple AR(3) baseline with a proper PyTorch LSTM that:
  - Takes last 24 time steps of per-zone AQI data as input
  - Predicts the next 3 time steps of AQI
  - Trains on historical OpenAQ CSV data with train/val/test split
  - Provides a ForecastEngine for runtime inference

Usage:
    python3 ml/aqi_forecaster.py              # train model, print metrics
    python3 ml/aqi_forecaster.py --evaluate   # evaluate saved model
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [AQI-FORECASTER] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "ml", "models")
POLLUTION_CSV = os.path.join(DATA_DIR, "openaq_nyc.csv")
MODEL_PATH    = os.path.join(MODEL_DIR, "aqi_lstm.pt")
METRICS_PATH  = os.path.join(MODEL_DIR, "aqi_lstm_metrics.json")

# Model hyperparameters
SEQ_LEN       = 24    # input: 24 time steps of historical AQI
PRED_LEN      = 3     # output: predict next 3 time steps
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EPOCHS        = 50
PATIENCE      = 8     # early stopping patience


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class AQILSTMModel(nn.Module):
    """
    2-layer LSTM for AQI time series forecasting.
    Input:  (batch, seq_len, 1)  — last 24 AQI readings
    Output: (batch, pred_len)    — next 3 AQI predictions
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, pred_len: int = PRED_LEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden)
        return self.fc(last_hidden)         # (batch, pred_len)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class AQISequenceDataset(Dataset):
    """Sliding window dataset from a 1D AQI time series."""

    def __init__(self, series: np.ndarray, seq_len: int = SEQ_LEN,
                 pred_len: int = PRED_LEN):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.series   = series.astype(np.float32)
        self.n_windows = len(series) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.n_windows)

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len].reshape(-1, 1)
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x), torch.tensor(y)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_aqi_series() -> np.ndarray:
    """Load PM2.5 data from OpenAQ CSV and create a clean time series."""
    log.info("Loading AQI data from %s", POLLUTION_CSV)
    df = pd.read_csv(POLLUTION_CSV)

    # Filter for PM2.5 readings
    df = df[df["parameter"] == "pm25"].copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df = df.sort_values("date_utc")

    # Convert PM2.5 to AQI using EPA linear approximation
    # AQI ≈ PM2.5 * 4.167 for PM2.5 in range 0-12 µg/m³
    df["aqi"] = df["value"].clip(0, 500) * 4.167
    df["aqi"] = df["aqi"].clip(0, 500)

    # Resample to hourly means across all stations
    df = df.set_index("date_utc")
    hourly = df["aqi"].resample("1h").mean().dropna()

    series = hourly.values
    log.info("AQI series: %d hourly readings (%.1f days)",
             len(series), len(series) / 24)
    return series


def normalise_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalise to [0, 1]. Returns (normalised, min_val, max_val)."""
    s_min, s_max = float(series.min()), float(series.max())
    if s_max - s_min < 1e-6:
        return np.zeros_like(series), s_min, s_max
    return (series - s_min) / (s_max - s_min), s_min, s_max


def denormalise(values: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
    """Reverse min-max normalisation."""
    return values * (s_max - s_min) + s_min


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AR(3) BASELINE (for comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def ar3_predict(history: np.ndarray, n_pred: int = PRED_LEN) -> np.ndarray:
    """Simple AR(3) forecast for comparison with LSTM."""
    if len(history) < 4:
        return np.full(n_pred, history[-1] if len(history) > 0 else 0.0)

    y = history.astype(float)
    n = len(y)
    p = 3
    X = np.column_stack([y[i:n-p+i] for i in range(p)])
    y_target = y[p:]

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
    except Exception:
        return np.full(n_pred, y[-1])

    preds = []
    buf = list(y[-p:])
    for _ in range(n_pred):
        pred = float(np.dot(coeffs, buf[-p:]))
        pred = max(0.0, min(500.0, pred))
        preds.append(pred)
        buf.append(pred)
    return np.array(preds)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_forecaster():
    """Train LSTM on historical AQI data with early stopping."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and normalise data
    raw_series = load_aqi_series()
    norm_series, s_min, s_max = normalise_series(raw_series)

    # Split: 70% train, 15% val, 15% test
    n = len(norm_series)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_s = norm_series[:n_train]
    val_s   = norm_series[n_train:n_train + n_val]
    test_s  = norm_series[n_train + n_val:]

    log.info("Split: train=%d val=%d test=%d", len(train_s), len(val_s), len(test_s))

    train_ds = AQISequenceDataset(train_s)
    val_ds   = AQISequenceDataset(val_s)
    test_ds  = AQISequenceDataset(test_s)

    if len(train_ds) < 10:
        log.error("Not enough training data (%d windows). Need at least 10.", len(train_ds))
        return None

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AQILSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    log.info("Training LSTM on %s (params: %d)",
             device, sum(p.numel() for p in model.parameters()))

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        train_loss /= len(train_ds)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * len(x_batch)
        val_loss /= max(len(val_ds), 1)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            log.info("Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                     epoch, EPOCHS, train_loss, val_loss, lr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "s_min": s_min, "s_max": s_max,
                "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d (best val_loss=%.6f)",
                         epoch, best_val_loss)
                break

    # ── Evaluate on test set ─────────────────────────────────────────────────
    # Reload best model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # LSTM predictions
    lstm_preds, lstm_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            lstm_preds.append(pred)
            lstm_targets.append(y_batch.numpy())

    if lstm_preds:
        lstm_preds   = denormalise(np.concatenate(lstm_preds), s_min, s_max)
        lstm_targets = denormalise(np.concatenate(lstm_targets), s_min, s_max)

        lstm_mse  = float(np.mean((lstm_preds - lstm_targets) ** 2))
        lstm_mae  = float(np.mean(np.abs(lstm_preds - lstm_targets)))
        lstm_mape = float(np.mean(np.abs((lstm_preds - lstm_targets) /
                                          np.maximum(lstm_targets, 1.0))) * 100)
    else:
        lstm_mse = lstm_mae = lstm_mape = float("inf")

    # AR(3) baseline predictions on same test data
    raw_test = raw_series[n_train + n_val:]
    ar3_mses, ar3_maes, ar3_mapes = [], [], []
    for i in range(SEQ_LEN, len(raw_test) - PRED_LEN + 1):
        history = raw_test[max(0, i - SEQ_LEN):i]
        ar3_pred = ar3_predict(history, PRED_LEN)
        actual   = raw_test[i:i + PRED_LEN]
        ar3_mses.append(float(np.mean((ar3_pred - actual) ** 2)))
        ar3_maes.append(float(np.mean(np.abs(ar3_pred - actual))))
        ar3_mapes.append(float(np.mean(np.abs((ar3_pred - actual) /
                                               np.maximum(actual, 1.0))) * 100))

    ar3_mse  = float(np.mean(ar3_mses))  if ar3_mses else float("inf")
    ar3_mae  = float(np.mean(ar3_maes))  if ar3_maes else float("inf")
    ar3_mape = float(np.mean(ar3_mapes)) if ar3_mapes else float("inf")

    # ── Print comparison table ───────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("  LSTM vs AR(3) Comparison — Test Set")
    log.info("=" * 60)
    log.info("  %-20s %12s %12s", "Metric", "LSTM", "AR(3)")
    log.info("  " + "-" * 46)
    log.info("  %-20s %12.4f %12.4f", "MSE",  lstm_mse,  ar3_mse)
    log.info("  %-20s %12.4f %12.4f", "MAE",  lstm_mae,  ar3_mae)
    log.info("  %-20s %11.2f%% %11.2f%%", "MAPE", lstm_mape, ar3_mape)
    improvement = (1 - lstm_mse / ar3_mse) * 100 if ar3_mse > 0 else 0
    log.info("  " + "-" * 46)
    log.info("  LSTM improvement: %.1f%% lower MSE", improvement)
    log.info("=" * 60)

    # Save metrics
    metrics = {
        "lstm": {"mse": round(lstm_mse, 4), "mae": round(lstm_mae, 4),
                 "mape": round(lstm_mape, 2)},
        "ar3":  {"mse": round(ar3_mse, 4),  "mae": round(ar3_mae, 4),
                 "mape": round(ar3_mape, 2)},
        "improvement_pct": round(improvement, 1),
        "train_size": len(train_s),
        "test_size": len(test_s),
        "epochs_trained": epoch,
        "best_val_loss": round(best_val_loss, 6),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", METRICS_PATH)
    log.info("Model saved to %s", MODEL_PATH)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FORECAST ENGINE (Runtime inference)
# ═══════════════════════════════════════════════════════════════════════════════

class ForecastEngine:
    """
    Runtime AQI forecasting engine.
    Maintains a sliding window of recent AQI readings per zone.
    Calls LSTM for predictions, falls back to AR(3) if model not available.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.device = torch.device("cpu")
        self.s_min = 0.0
        self.s_max = 500.0
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self._history: dict[str, deque] = {}
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained LSTM model from disk."""
        if not os.path.exists(model_path):
            log.warning("LSTM model not found at %s — using AR(3) fallback", model_path)
            return
        try:
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=True)
            self.s_min = checkpoint["s_min"]
            self.s_max = checkpoint["s_max"]
            self.seq_len = checkpoint.get("seq_len", SEQ_LEN)
            self.pred_len = checkpoint.get("pred_len", PRED_LEN)
            self.model = AQILSTMModel(
                hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
                num_layers=checkpoint.get("num_layers", NUM_LAYERS),
                pred_len=self.pred_len,
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            log.info("LSTM model loaded from %s", model_path)
        except Exception as e:
            log.warning("Failed to load LSTM model: %s — using AR(3) fallback", e)
            self.model = None

    def _get_history(self, zone: str) -> deque:
        """Get or create history deque for a zone."""
        if zone not in self._history:
            self._history[zone] = deque(maxlen=self.seq_len)
        return self._history[zone]

    def forecast(self, zone: str, current_aqi: float) -> float:
        """
        Predict next AQI for a zone given current reading.
        Returns single-step-ahead forecast.
        Uses LSTM if model loaded and enough history, else AR(3).
        """
        hist = self._get_history(zone)
        hist.append(current_aqi)

        # Try LSTM first
        if self.model is not None and len(hist) >= self.seq_len:
            try:
                return self._lstm_forecast(hist)
            except Exception as e:
                log.debug("LSTM forecast failed for %s: %s — using AR(3)", zone, e)

        # AR(3) fallback
        return self._ar3_forecast(hist, current_aqi)

    def _lstm_forecast(self, hist: deque) -> float:
        """Run LSTM inference on the history window."""
        values = np.array(list(hist), dtype=np.float32)
        # Normalise
        if self.s_max - self.s_min > 1e-6:
            normed = (values - self.s_min) / (self.s_max - self.s_min)
        else:
            normed = np.zeros_like(values)

        x = torch.tensor(normed.reshape(1, -1, 1))  # (1, seq_len, 1)
        with torch.no_grad():
            pred = self.model(x).numpy()[0]  # (pred_len,)

        # Denormalise — return first prediction (1-step ahead)
        result = float(pred[0] * (self.s_max - self.s_min) + self.s_min)
        return round(max(0.0, min(500.0, result)), 1)

    def _ar3_forecast(self, hist: deque, current_aqi: float) -> float:
        """AR(3) fallback when LSTM is unavailable."""
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

    @property
    def is_lstm_active(self) -> bool:
        """Whether LSTM model is loaded and ready."""
        return self.model is not None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UrbanStream LSTM AQI Forecaster")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate saved model without retraining")
    args = parser.parse_args()

    if args.evaluate:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            print(json.dumps(metrics, indent=2))
        else:
            log.error("No metrics file found. Run training first.")
    else:
        train_forecaster()


if __name__ == "__main__":
    main()
