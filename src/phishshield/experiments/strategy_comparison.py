"""DVC stage `strategy_comparison` (proposal 5.8): STATIC vs FIXED-SCHEDULE vs TIERED DRIFT-AWARE."""

from __future__ import annotations

import copy
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phishshield.config import (  # noqa: E402
    MODELS_DIR,
    REPORTS_DIR,
    STREAM_DIR,
    ensure_dirs,
    load_params,
    set_seed,
)
from phishshield.drift.embedding_drift import embedding_drift  # noqa: E402
from phishshield.drift.feature_drift import feature_drift  # noqa: E402
from phishshield.drift.risk_policy import DriftSignals, decide  # noqa: E402
from phishshield.evaluation.metrics import classification_metrics  # noqa: E402
from phishshield.features.lexical import FEATURE_NAMES, to_matrix  # noqa: E402
from phishshield.models.student import build_student, load_student, student_logits  # noqa: E402
from phishshield.training.calibration import fit_temperature_bounded as fit_temperature  # noqa: E402
from phishshield.training.calibration import pick_threshold as best_threshold  # noqa: E402
from phishshield.training.common import Split, load_split, sigmoid  # noqa: E402
from phishshield.training.student_trainer import fit_student  # noqa: E402
from phishshield.utils import tracking  # noqa: E402
from phishshield.utils.common import get_logger, save_json  # noqa: E402

log = get_logger("strategy_comparison")


class Deployed:
    """A student model + its calibration, as it would be deployed."""

    def __init__(self, model, temperature: float = 1.0, threshold: float = 0.5):
        self.model, self.temperature, self.threshold = model, temperature, threshold

    def proba(self, s: Split, max_len: int, emb: bool = False):
        out = student_logits(self.model, s.urls, s.Xs, max_len, return_embeddings=emb)
        if emb:
            return sigmoid(out[0] / self.temperature), out[1]
        return sigmoid(out / self.temperature)


def make_stream(scaler) -> list[Split]:
    files = sorted(STREAM_DIR.glob("batch_*.csv"))
    if not files:
        raise SystemExit("No stream batches - run the build_stream stage first.")
    out = []
    for f in files:
        df = pd.read_csv(f)
        urls = df["url"].astype(str).tolist()
        X = to_matrix(urls)
        out.append(Split(urls, X, scaler.transform(X).astype(np.float32), df["label"].to_numpy()))
    return out


def cat(splits: list[Split]) -> Split:
    return Split(sum((s.urls for s in splits), []), np.vstack([s.X for s in splits]),
                 np.vstack([s.Xs for s in splits]), np.concatenate([s.y for s in splits]))


def calibrate(dep: Deployed, s: Split, max_len: int, max_fnr: float) -> None:
    logits = student_logits(dep.model, s.urls, s.Xs, max_len)
    dep.temperature = fit_temperature(logits, s.y)
    dep.threshold = best_threshold(s.y, sigmoid(logits / dep.temperature), max_fnr)


def full_retrain(data: Split, calib: Split, params: dict) -> Deployed:
    m = build_student(params["student"], len(FEATURE_NAMES))
    fit_student(m, data, params["student"], seed=params["seed"])
    dep = Deployed(m)
    calibrate(dep, calib, params["student"]["max_len"], params["promotion"]["max_fnr"])
    return dep


def run_strategy(name: str, initial: Deployed, train: Split, val: Split, stream: list[Split], params: dict) -> dict:
    scfg, dcfg = params["student"], params["drift"]
    max_len, max_fnr = scfg["max_len"], params["promotion"]["max_fnr"]
    dep = Deployed(copy.deepcopy(initial.model), initial.temperature, initial.threshold)
    labelled = [train]
    ref_split = train.subset(np.random.default_rng(0).choice(len(train), min(800, len(train)), replace=False))
    ref_p, ref_emb = dep.proba(ref_split, max_len, emb=True)
    ref_f1 = classification_metrics(val.y, dep.proba(val, max_len), dep.threshold)["f1"]
    rows, total_cost = [], 0.0
    window: list[Split] = []
    for i, batch in enumerate(stream):
        p, emb = dep.proba(batch, max_len, emb=True)
        m = classification_metrics(batch.y, p, dep.threshold)
        action, risk, cost = "none", None, 0.0
        labelled.append(batch)
        t0 = time.perf_counter()
        if name == "fixed_schedule" and (i + 1) % params["experiment"]["fixed_schedule_every"] == 0:
            dep = full_retrain(cat(labelled), batch, params)
            action = "full_retrain"
        elif name == "tiered_drift_aware":
            fd = feature_drift(pd.DataFrame(ref_split.X, columns=FEATURE_NAMES), pd.DataFrame(batch.X, columns=FEATURE_NAMES),
                               dcfg["psi_threshold"], dcfg["ks_pvalue"])
            ed = embedding_drift(ref_emb, emb, n_perm=50, max_n=300, seed=params["seed"])
            dec = decide(DriftSignals(fd["drift_share"], ed["score"], ref_f1, m["f1"], m["fnr"]), dcfg, max_fnr)
            action, risk = dec.action.value, dec.risk
            half = len(batch) // 2
            adapt, hold = batch.subset(np.arange(half)), batch.subset(np.arange(half, len(batch)))
            recent = cat(window + [adapt])
            cand, weights_updated = None, False
            if action == "recalibrate":
                cand = Deployed(dep.model, dep.temperature, dep.threshold)
                calibrate(cand, recent, max_len, max_fnr)
            elif action == "partial_finetune":
                cm = copy.deepcopy(dep.model)
                replay = train.subset(np.random.default_rng(i).choice(len(train), min(len(train), len(recent)),
                                                                      replace=False))
                fit_student(cm, cat([replay, recent]), scfg, epochs=3, partial=True, seed=i)
                cand = Deployed(cm)
                calibrate(cand, recent, max_len, max_fnr)
            elif action == "full_retrain":
                cand = full_retrain(cat(labelled[:-1] + [adapt]), adapt, params)
            if cand is not None:
                c_f1 = classification_metrics(hold.y, cand.proba(hold, max_len), cand.threshold)["f1"]
                o_f1 = classification_metrics(hold.y, dep.proba(hold, max_len), dep.threshold)["f1"]
                if c_f1 >= o_f1:
                    dep = cand
                    if action in ("partial_finetune", "full_retrain"):
                        ref_split = cat([ref_split.subset(np.arange(0, len(ref_split), 2)), batch])
                        ref_p, ref_emb = dep.proba(ref_split, max_len, emb=True)
                        weights_updated = True
                else:
                    action += " (rejected)"
            window = [] if weights_updated else window + [batch]
        cost = time.perf_counter() - t0
        total_cost += cost
        rows.append({"batch": i, "f1": m["f1"], "recall": m["recall"], "fnr": m["fnr"], "precision": m["precision"],
                     "roc_auc": m["roc_auc"],
                     "action": action, "risk": risk, "update_cost_s": cost})
        log.info("[%s] batch %d F1=%.4f FNR=%.4f action=%s", name, i, m["f1"], m["fnr"], action)
    df = pd.DataFrame(rows)
    n_updates = int((~df.action.str.startswith("none")).sum())
    applied = df.action[~df.action.str.startswith("none") & ~df.action.str.contains("rejected")]
    return {"per_batch": rows, "mean_f1": df.f1.mean(), "mean_recall": df.recall.mean(), "mean_fnr": df.fnr.mean(),
            "mean_precision": df.precision.mean(), "mean_roc_auc": df.roc_auc.mean(),
            "applied_updates": applied.value_counts().to_dict(),
            "worst_f1": df.f1.min(), "n_updates": n_updates, "total_update_cost_s": total_cost,
            "actions": df.action.value_counts().to_dict()}


def main() -> None:
    params = load_params()
    ensure_dirs()
    set_seed(params["seed"])
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    train, val = load_split("train"), load_split("val")
    stream = make_stream(scaler)
    log.info("Stream: %d batches x %d URLs", len(stream), len(stream[0]))

    student, smeta = load_student(MODELS_DIR / "student")
    initial = Deployed(student, smeta["temperature"], smeta["threshold"])
    results = {}
    with tracking.run("strategy-comparison", params["mlflow"]["experiment"], params["experiment"], {"stage": "experiment"}):
        for name in ("static", "fixed_schedule", "tiered_drift_aware"):
            set_seed(params["seed"])
            results[name] = run_strategy(name, initial, train, val, stream, params)
            tracking.log_metrics({k: v for k, v in results[name].items() if isinstance(v, (int, float))}, prefix=f"{name}.")

    summary = {n: {k: r[k] for k in ("mean_precision", "mean_recall", "mean_f1", "worst_f1", "mean_roc_auc", "mean_fnr",
                                     "n_updates", "applied_updates", "total_update_cost_s")}
               for n, r in results.items()}
    save_json(results, REPORTS_DIR / "strategy_comparison.json")
    save_json(summary, REPORTS_DIR / "strategy_summary.json")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n, r in results.items():
        df = pd.DataFrame(r["per_batch"])
        ax[0].plot(df.batch, df.f1, marker="o", label=n)
        ax[1].plot(df.batch, df.update_cost_s.cumsum(), marker="o", label=n)
    ax[0].set(title="F1 on each incoming batch (before update)", xlabel="batch (drift increases ->)", ylabel="F1")
    ax[1].set(title="Cumulative update cost", xlabel="batch (drift increases ->)", ylabel="seconds")
    ax[0].legend()
    plt.tight_layout()
    (REPORTS_DIR / "figures").mkdir(exist_ok=True)
    plt.savefig(REPORTS_DIR / "figures" / "strategy_comparison.png", dpi=130)
    plt.close()

    lines = ["| Strategy | Precision | Recall | F1 | Worst F1 | ROC-AUC | FNR | Updates (applied by type) | Update cost (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, s in summary.items():
        kinds = ", ".join(f"{v}x {k}" for k, v in s["applied_updates"].items()) or "none"
        lines.append(f"| {n} | {s['mean_precision']:.4f} | {s['mean_recall']:.4f} | {s['mean_f1']:.4f} | {s['worst_f1']:.4f} | "
                     f"{s['mean_roc_auc']:.4f} | {s['mean_fnr']:.4f} | {s['n_updates']} ({kinds}) | "
                     f"{s['total_update_cost_s']:.1f} |")
    (REPORTS_DIR / "strategy_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
