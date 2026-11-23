"""Execute the tier chosen by the risk policy, then validate -> promote / reject (proposal 5.6 + 5.7)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishshield.config import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_params, set_seed
from phishshield.drift.monitor import latest_action
from phishshield.features import char_tokenizer
from phishshield.features.lexical import FEATURE_NAMES, to_matrix
from phishshield.registry import model_registry as reg
from phishshield.registry.promote import validate_and_promote
from phishshield.training.calibration import fit_temperature_bounded as fit_temperature
from phishshield.training.calibration import pick_threshold as best_threshold
from phishshield.training.common import Split, load_split, sigmoid
from phishshield.utils import tracking
from phishshield.utils.common import get_logger, save_json

log = get_logger("update")


def to_split(df: pd.DataFrame, scaler) -> Split:
    urls = df["url"].astype(str).tolist()
    X = to_matrix(urls)
    return Split(urls, X, scaler.transform(X).astype(np.float32), df["label"].to_numpy(dtype=np.int64))


def concat(a: Split, b: Split) -> Split:
    return Split(a.urls + b.urls, np.vstack([a.X, b.X]), np.vstack([a.Xs, b.Xs]), np.r_[a.y, b.y])


def recalibrate(champ: Path, adapt: Split, out: Path, params: dict) -> None:
    from phishshield.api.predictor import PhishPredictor

    shutil.copytree(champ, out, dirs_exist_ok=True)
    pred = PhishPredictor(champ)
    logits = pred.session.run(["logit"], {"char_ids": char_tokenizer.encode_batch(adapt.urls, pred.max_len),
                                          "features": adapt.Xs})[0]
    t = fit_temperature(logits, adapt.y)
    thr = best_threshold(adapt.y, sigmoid(logits / t), params["promotion"]["max_fnr"])
    cfg = json.loads((out / "config.json").read_text())
    cfg.update({"temperature": t, "threshold": thr})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))


def retrain(adapt: Split, out: Path, params: dict, partial: bool) -> None:
    from phishshield.models.hybrid import load_hybrid, predict_logits, save_hybrid
    from phishshield.models.student import build_student, load_student, save_student, student_logits
    from phishshield.training.distill import export_onnx
    from phishshield.training.hybrid_trainer import build_hybrid, fit_hybrid
    from phishshield.training.student_trainer import fit_student

    hcfg, scfg = dict(params["hybrid"]), params["student"]
    base = load_split("train")
    rng = np.random.default_rng(params["seed"])
    if partial:
        replay = base.subset(rng.choice(len(base), min(len(base), 3 * len(adapt)), replace=False))
        data = concat(replay, adapt)
        teacher, tmeta = load_hybrid(reg.teacher_dir())
        fit_hybrid(teacher, data, adapt, hcfg, epochs=1, partial=True, log_prefix="partial_")
        student, _ = load_student(reg.champion_dir())
        student_epochs = max(2, scfg["epochs"] // 2)
    else:
        if hcfg["max_train_samples"] and len(base) > hcfg["max_train_samples"]:
            base = base.subset(np.sort(rng.choice(len(base), hcfg["max_train_samples"], replace=False)))
        data = concat(base, adapt)
        teacher = build_hybrid(hcfg, len(FEATURE_NAMES))
        fit_hybrid(teacher, data, adapt, hcfg, log_prefix="full_")
        tmeta = {"pretrained_name": hcfg["pretrained_name"], "debug_tiny": hcfg["debug_tiny"],
                 "max_len": hcfg["max_len"], "n_features": len(FEATURE_NAMES), "fusion_dim": hcfg["fusion_dim"],
                 "dropout": hcfg["dropout"], "mode": hcfg["mode"],
                 "n_params": sum(p.numel() for p in teacher.parameters())}
        student = build_student(scfg, len(FEATURE_NAMES))
        student_epochs = scfg["epochs"]

    v_logits = predict_logits(teacher, adapt.urls, adapt.Xs)
    t_temp = fit_temperature(v_logits, adapt.y)
    tmeta = {**tmeta, "temperature": t_temp, "threshold": best_threshold(adapt.y, sigmoid(v_logits / t_temp))}
    save_hybrid(teacher, out / "teacher", tmeta)

    t_logits = predict_logits(teacher, data.urls, data.Xs) / t_temp
    fit_student(student, data, scfg, teacher_logits=t_logits, epochs=student_epochs, seed=params["seed"])
    s_logits = student_logits(student, adapt.urls, adapt.Xs, scfg["max_len"])
    s_temp = fit_temperature(s_logits, adapt.y)
    smeta = {**{k: scfg[k] for k in ("max_len", "embed_dim", "num_filters", "kernel_sizes")},
             "n_features": len(FEATURE_NAMES), "temperature": s_temp,
             "threshold": best_threshold(adapt.y, sigmoid(s_logits / s_temp), params["promotion"]["max_fnr"]),
             "n_params": sum(p.numel() for p in student.parameters())}
    save_student(student, out, smeta)
    export_onnx(student, out / "student.onnx", scfg["max_len"], len(FEATURE_NAMES))
    champ = reg.champion_dir()
    for f in ("scaler.joblib", "shap_background.npy"):
        if champ and (champ / f).exists():
            shutil.copy2(champ / f, out / f)


def run_update(batch: pd.DataFrame, action: str, params: dict) -> dict:
    if action == "none":
        log.info("Policy decided NONE - nothing to do.")
        return {"action": "none", "promoted": False}
    if "label" not in batch.columns:
        raise SystemExit("Updating needs labelled data (a `label` column).")
    if reg.champion_dir() is None:
        raise SystemExit("No champion model - run `dvc repro` first.")
    set_seed(params["seed"])
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    batch = batch.sample(frac=1.0, random_state=params["seed"]).reset_index(drop=True)
    half = len(batch) // 2
    adapt, holdout = to_split(batch.iloc[:half], scaler), to_split(batch.iloc[half:], scaler)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp, tracking.run(
            f"update-{action}", params["mlflow"]["experiment"],
            {"action": action, "batch_size": len(batch)}, {"stage": "update"}):
        out = Path(tmp) / "candidate"
        out.mkdir()
        if action == "recalibrate":
            recalibrate(reg.champion_dir(), adapt, out, params)
        else:
            retrain(adapt, out, params, partial=(action == "partial_finetune"))
        cost = time.perf_counter() - t0
        result = validate_and_promote(out, f"policy action: {action}", holdout.urls, holdout.y, params)
        tracking.log_metrics({"update_cost_s": cost, "promoted": float(result["promoted"]),
                              **{f"candidate_{k}": v for k, v in result["candidate"].items() if isinstance(v, float)}})
    result.update({"action": action, "update_cost_s": cost})
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--action", default=None, choices=["none", "recalibrate", "partial_finetune", "full_retrain"])
    args = ap.parse_args()
    params = load_params()
    ensure_dirs()
    action = args.action or latest_action()
    result = run_update(pd.read_csv(args.batch), action, params)
    save_json(result, REPORTS_DIR / "drift" / "latest_update.json")
    log.info("Update finished: action=%s promoted=%s", action, result.get("promoted"))


if __name__ == "__main__":
    main()
