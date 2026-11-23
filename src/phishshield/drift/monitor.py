"""Drift monitor: compare a new batch of URLs with the training reference and decide what to do."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from phishshield.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_params
from phishshield.drift.embedding_drift import embedding_drift
from phishshield.drift.feature_drift import evidently_report, feature_drift
from phishshield.drift.risk_policy import DriftSignals, decide
from phishshield.evaluation.metrics import classification_metrics
from phishshield.features.lexical import FEATURE_NAMES, extract_many
from phishshield.registry import model_registry as reg
from phishshield.utils.common import get_logger, load_json, save_json

log = get_logger("monitor")
DRIFT_DIR = REPORTS_DIR / "drift"


def _uncertain_share(p: np.ndarray) -> float:
    return float(np.mean((p > 0.3) & (p < 0.7))) if len(p) else 0.0


_TEACHER_CACHE: dict = {}


def embeddings(urls: list[str], predictor, use_teacher: bool) -> tuple[np.ndarray, str]:
    """Teacher [CLS] embeddings when the teacher + torch are available, otherwise student embeddings."""
    tdir = reg.teacher_dir()
    if use_teacher and (tdir / "model.pt").exists():
        try:
            from phishshield.models.hybrid import load_hybrid, predict_logits

            key = str(tdir)
            if key not in _TEACHER_CACHE:
                _TEACHER_CACHE.clear()
                _TEACHER_CACHE[key] = load_hybrid(tdir)[0]
            model = _TEACHER_CACHE[key]
            _, emb = predict_logits(model, urls, predictor.features(urls), return_embeddings=True)
            return emb, "teacher_cls"
        except Exception as e:
            log.warning("Teacher embeddings unavailable (%s) - using student embeddings", e)
    return predictor.predict_proba(urls)[1], "student"


def analyse(batch: pd.DataFrame, params: dict, ref: pd.DataFrame | None = None, predictor=None,
            use_teacher: bool = True, name: str = "batch", write_html: bool = True) -> dict:
    from phishshield.api.predictor import PhishPredictor

    dcfg = params["drift"]
    rng = np.random.default_rng(params["seed"])
    if ref is None:
        ref = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    if len(ref) > dcfg["max_samples"]:
        ref = ref.iloc[rng.choice(len(ref), dcfg["max_samples"], replace=False)]
    if predictor is None:
        if reg.champion_dir() is None:
            raise SystemExit("No champion model yet - run `dvc repro` (stage promote) first.")
        predictor = PhishPredictor(reg.champion_dir())
    cur_urls = batch["url"].astype(str).tolist()
    cur_feats = extract_many(cur_urls)

    fd = feature_drift(ref[FEATURE_NAMES].reset_index(drop=True), cur_feats, dcfg["psi_threshold"], dcfg["ks_pvalue"])
    ref_urls = ref["url"].astype(str).tolist()
    ref_emb, src = embeddings(ref_urls, predictor, use_teacher)
    cur_emb, _ = embeddings(cur_urls, predictor, use_teacher)
    ed = embedding_drift(ref_emb, cur_emb, dcfg["mmd_permutations"], min(500, dcfg["max_samples"]), params["seed"])

    ref_p, _ = predictor.predict_proba(ref_urls)
    cur_p, _ = predictor.predict_proba(cur_urls)
    signals = DriftSignals(feature_drift_share=fd["drift_share"], embedding_score=ed["score"],
                           uncertain_share_ref=_uncertain_share(ref_p), uncertain_share_cur=_uncertain_share(cur_p))
    perf = None
    if "label" in batch.columns and batch["label"].notna().all():
        perf = classification_metrics(batch["label"].astype(int), cur_p, predictor.threshold)
        ref_m = classification_metrics(ref["label"].astype(int), ref_p, predictor.threshold)
        signals.reference_f1, signals.current_f1, signals.current_fnr = ref_m["f1"], perf["f1"], perf["fnr"]
    decision = decide(signals, dcfg, params["promotion"]["max_fnr"])

    if write_html:
        evidently_report(ref[FEATURE_NAMES].reset_index(drop=True), cur_feats, DRIFT_DIR / f"{name}_evidently.html")
    fd_small = {k: v for k, v in fd.items() if k != "per_feature"}
    return {"batch": name, "n_urls": len(batch), "model_version": predictor.version, "embedding_source": src,
            "feature_drift": fd_small, "embedding_drift": ed, "performance": perf,
            "signals": signals.__dict__, "decision": decision.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="CSV with url[,label]")
    ap.add_argument("--student-embeddings", action="store_true", help="skip the teacher (faster, no torch)")
    args = ap.parse_args()
    params = load_params()
    ensure_dirs()
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(args.batch).stem
    result = analyse(pd.read_csv(args.batch), params, use_teacher=not args.student_embeddings, name=name)
    out = DRIFT_DIR / f"{name}_drift.json"
    save_json(result, out)
    shutil.copy2(out, DRIFT_DIR / "latest.json")
    d = result["decision"]
    log.info("Batch %s | risk=%.3f -> ACTION: %s", name, d["risk"], d["action"].upper())
    for r in d["reasons"]:
        log.info("  - %s", r)


def latest_action() -> str:
    return load_json(DRIFT_DIR / "latest.json")["decision"]["action"]


if __name__ == "__main__":
    main()
