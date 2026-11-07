"""Shared training helpers: data loading, device selection, temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from phishshield.config import MODELS_DIR, PROCESSED_DIR
from phishshield.features.lexical import FEATURE_NAMES


@dataclass
class Split:
    urls: list[str]
    X: np.ndarray          # raw engineered features
    Xs: np.ndarray         # scaled engineered features (input to neural models)
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx) -> Split:
        return Split([self.urls[i] for i in idx], self.X[idx], self.Xs[idx], self.y[idx])


def load_split(name: str, scaler=None) -> Split:
    df = pd.read_csv(PROCESSED_DIR / f"{name}_features.csv")
    scaler = scaler or joblib.load(MODELS_DIR / "scaler.joblib")
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    return Split(df["url"].astype(str).tolist(), X, scaler.transform(X).astype(np.float32),
                 df["label"].to_numpy(dtype=np.int64))


def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): find T minimising validation NLL of sigmoid(logit/T)."""
    y = y.astype(float)

    def nll(t: float) -> float:
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)
