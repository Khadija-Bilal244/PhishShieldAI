"""Classification metrics used everywhere (training, promotion gate, experiments)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if both else float("nan"),
        "pr_auc": average_precision_score(y_true, prob) if both else float("nan"),
        "fnr": fn / max(fn + tp, 1),   # missed phishing - the most dangerous error
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def best_threshold(y_true, prob, max_fnr: float | None = None) -> float:
    """Threshold maximising F1 (optionally subject to an FNR ceiling). Used by recalibration."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m = classification_metrics(y_true, prob, t)
        if max_fnr is not None and m["fnr"] > max_fnr:
            continue
        if m["f1"] > best_f1:
            best_t, best_f1 = float(t), m["f1"]
    return best_t
