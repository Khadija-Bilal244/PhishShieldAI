"""Embedding-level drift on transformer [CLS] (or student) embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    return float(_rbf(x, x, gamma).mean() + _rbf(y, y, gamma).mean() - 2 * _rbf(x, y, gamma).mean())


def mmd_test(ref: np.ndarray, cur: np.ndarray, n_perm: int = 200, max_n: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ref = ref[rng.choice(len(ref), min(max_n, len(ref)), replace=False)]
    cur = cur[rng.choice(len(cur), min(max_n, len(cur)), replace=False)]
    z = np.vstack([ref, cur]).astype(np.float64)
    z = (z - z.mean(0)) / (z.std(0) + 1e-8)
    sq = (z * z).sum(1)
    dists = sq[:, None] + sq[None, :] - 2 * z @ z.T
    gamma = 1.0 / (np.median(dists[dists > 0]) + 1e-8)
    n = len(ref)
    observed = mmd2(z[:n], z[n:], gamma)
    k = _rbf(z, z, gamma)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(z))
        a, b = p[:n], p[n:]
        stat = k[np.ix_(a, a)].mean() + k[np.ix_(b, b)].mean() - 2 * k[np.ix_(a, b)].mean()
        count += stat >= observed
    return {"mmd2": observed, "p_value": float((count + 1) / (n_perm + 1))}


def domain_classifier_auc(ref: np.ndarray, cur: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    ref = ref[rng.choice(len(ref), min(max_n, len(ref)), replace=False)]
    cur = cur[rng.choice(len(cur), min(max_n, len(cur)), replace=False)]
    X = np.vstack([ref, cur])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def embedding_drift(ref: np.ndarray, cur: np.ndarray, n_perm: int = 200, max_n: int = 500, seed: int = 0) -> dict:
    mmd = mmd_test(ref, cur, n_perm, max_n, seed)
    auc = domain_classifier_auc(ref, cur, 2 * max_n, seed)
    score = float(np.clip((auc - 0.55) / 0.35, 0, 1))
    significant = mmd["p_value"] < 0.05
    if significant:
        score = max(score, 0.3)
    return {**mmd, "domain_auc": auc, "score": score, "drift_detected": bool(significant and auc > 0.6)}
