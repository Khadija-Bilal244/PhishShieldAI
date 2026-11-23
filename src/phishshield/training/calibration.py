"""Robust calibration for the serving model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from phishshield.evaluation.metrics import classification_metrics
from phishshield.training.common import sigmoid


def fit_temperature_bounded(logits: np.ndarray, y: np.ndarray, t_min: float = 0.5, t_max: float = 10.0) -> float:
    y = y.astype(float)

    def nll(t: float) -> float:
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return float(minimize_scalar(nll, bounds=(t_min, t_max), method="bounded").x)


def pick_threshold(y, prob, max_fnr: float | None = None, tol: float = 1e-3) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scored = []
    for t in grid:
        m = classification_metrics(y, prob, t)
        if max_fnr is None or m["fnr"] <= max_fnr:
            scored.append((float(t), m["f1"]))
    if not scored:
        return 0.5
    best = max(f for _, f in scored)
    return min((t for t, f in scored if f >= best - tol), key=lambda t: abs(t - 0.5))
