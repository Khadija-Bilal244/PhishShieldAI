"""Serving engine: ONNX Runtime + numpy only (no torch in the API image)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np

from phishshield.features import char_tokenizer
from phishshield.features.lexical import FEATURE_NAMES, to_matrix

_SPLIT = re.compile(r"([/\.\-_?=&@:]+)")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def risk_level(prob: float) -> str:
    return "high" if prob >= 0.7 else "medium" if prob >= 0.35 else "low"


def permutation_shap(f, x: np.ndarray, background: np.ndarray, n_perm: int = 64, seed: int = 0) -> np.ndarray:
    """Monte-Carlo Shapley values (Strumbelj & Kononenko, 2014), one batched model call, low memory."""
    rng = np.random.default_rng(seed)
    d = len(x)
    perms = np.stack([rng.permutation(d) for _ in range(n_perm)])
    rows = np.empty((n_perm, d + 1, d), dtype=np.float32)
    for p, perm in enumerate(perms):
        z = background[rng.integers(len(background))].astype(np.float32).copy()
        rows[p, 0] = z
        for k, j in enumerate(perm, 1):
            z[j] = x[j]
            rows[p, k] = z
    preds = f(rows.reshape(-1, d)).reshape(n_perm, d + 1)
    phi = np.zeros(d)
    for p, perm in enumerate(perms):
        phi[perm] += preds[p, 1:] - preds[p, :-1]
    return phi / n_perm


class PhishPredictor:
    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_dir / "student.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.max_len = int(self.config["max_len"])
        self.temperature = float(self.config.get("temperature", 1.0))
        self.threshold = float(self.config.get("threshold", 0.5))
        self.version = self.config.get("version", self.model_dir.name)
        bg = self.model_dir / "shap_background.npy"
        self.background = np.load(bg) if bg.exists() else None

    def _run(self, urls: list[str], feats_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = char_tokenizer.encode_batch(urls, self.max_len)
        logit, emb = self.session.run(["logit", "embedding"],
                                      {"char_ids": ids, "features": feats_scaled.astype(np.float32)})
        return _sigmoid(logit / self.temperature), emb

    def features(self, urls: list[str]) -> np.ndarray:
        return self.scaler.transform(to_matrix(urls)).astype(np.float32)

    def predict_proba(self, urls: list[str]) -> tuple[np.ndarray, np.ndarray]:
        urls = list(urls)
        if not urls:
            return np.zeros(0), np.zeros((0, 64))
        return self._run(urls, self.features(urls))

    def predict(self, urls: list[str]) -> list[dict]:
        probs, _ = self.predict_proba(urls)
        return [{"url": u, "phishing_probability": round(float(p), 4),
                 "prediction": "phishing" if p >= self.threshold else "legitimate",
                 "risk_level": risk_level(float(p)), "threshold": round(self.threshold, 4),
                 "model_version": self.version} for u, p in zip(urls, probs)]

    def segment_importance(self, url: str, base: float) -> list[dict]:
        parts = [p for p in _SPLIT.split(url) if p]
        idx = [i for i, p in enumerate(parts) if not _SPLIT.fullmatch(p)][:40]
        if not idx:
            return []
        variants = ["".join(p for j, p in enumerate(parts) if j != i) for i in idx]
        probs, _ = self.predict_proba(variants)
        out = [{"segment": parts[i], "importance": round(base - float(p), 4)} for i, p in zip(idx, probs)]
        return sorted(out, key=lambda d: abs(d["importance"]), reverse=True)

    def feature_shap(self, url: str, n_perm: int = 64, seed: int = 0) -> tuple[list[dict], str]:
        """Shapley values of the engineered features (permutation sampling, URL characters held fixed)."""
        x = self.features([url])[0]
        raw = to_matrix([url])[0]
        ids = char_tokenizer.encode_batch([url], self.max_len)
        bg = self.background if self.background is not None else np.zeros((1, len(x)), np.float32)
        values = permutation_shap(lambda feats: self._prob_with_features(ids, feats), x, bg, n_perm, seed)
        out = [{"feature": n, "value": round(float(raw[i]), 4), "importance": round(float(values[i]), 4)}
               for i, n in enumerate(FEATURE_NAMES)]
        return sorted(out, key=lambda d: abs(d["importance"]), reverse=True), "shap"

    def _prob_with_features(self, ids: np.ndarray, feats: np.ndarray) -> np.ndarray:
        rep = np.repeat(ids, len(feats), axis=0)
        logit = self.session.run(["logit"], {"char_ids": rep, "features": feats.astype(np.float32)})[0]
        return _sigmoid(logit / self.temperature)

    def explain(self, url: str, top_k: int = 6) -> dict:
        base = float(self._run([url], self.features([url]))[0][0])
        feats, method = self.feature_shap(url)
        return {"url": url, "phishing_probability": round(base, 4), "risk_level": risk_level(base),
                "top_segments": self.segment_importance(url, base)[:top_k],
                "top_features": feats[:top_k], "feature_method": method}
