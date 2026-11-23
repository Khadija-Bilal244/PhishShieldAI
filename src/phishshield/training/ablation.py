"""DVC stage `ablation` (proposal 5.3): how much does each branch of the hybrid model contribute?"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phishshield.config import REPORTS_DIR, ensure_dirs, load_params, set_seed  # noqa: E402
from phishshield.evaluation.metrics import best_threshold, classification_metrics  # noqa: E402
from phishshield.features.lexical import FEATURE_NAMES  # noqa: E402
from phishshield.models.hybrid import predict_logits  # noqa: E402
from phishshield.training.common import Split, fit_temperature, load_split, sigmoid  # noqa: E402
from phishshield.training.hybrid_trainer import build_hybrid, fit_hybrid  # noqa: E402
from phishshield.utils import tracking  # noqa: E402
from phishshield.utils.common import get_logger, load_json, save_json  # noqa: E402

log = get_logger("ablation")
KEYS = ("precision", "recall", "f1", "roc_auc", "fnr")


def cap(split: Split, n: int | None, seed: int) -> Split:
    if not n or len(split) <= n:
        return split
    return split.subset(np.sort(np.random.default_rng(seed).choice(len(split), n, replace=False)))


def train_variant(mode: str, params: dict, train: Split, val: Split, test: Split) -> dict:
    cfg = params["hybrid"]
    set_seed(params["seed"])
    with tracking.run(f"ablation-{mode}", params["mlflow"]["experiment"],
                      {"model": "hybrid_distilbert_fusion", "mode": mode, "n_train": len(train), **cfg},
                      {"stage": "ablation", "mode": mode}):
        model = build_hybrid(cfg, len(FEATURE_NAMES), mode)
        summary = fit_hybrid(model, train, val, cfg)
        v = predict_logits(model, val.urls, val.Xs)
        t = fit_temperature(v, val.y)
        thr = best_threshold(val.y, sigmoid(v / t), params["evaluation"]["max_fnr"])
        m = classification_metrics(test.y, sigmoid(predict_logits(model, test.urls, test.Xs) / t), thr)
        tracking.log_metrics(m, prefix="test_")
        tracking.log_metrics({"train_time_s": summary["train_time_s"],
                              "n_params": sum(p.numel() for p in model.parameters())})
    log.info("%-14s F1=%.4f recall=%.4f FNR=%.4f", mode, m["f1"], m["recall"], m["fnr"])
    return {k: m[k] for k in KEYS} | {"train_time_s": summary["train_time_s"]}


def main() -> None:
    params = load_params()
    ensure_dirs()
    train = cap(load_split("train"), params["hybrid"]["max_train_samples"], params["seed"])
    val, test = load_split("val"), load_split("test")

    fusion = load_json(REPORTS_DIR / "hybrid_metrics.json")
    results = {"fusion": {k: fusion["test"][k] for k in KEYS} | {"train_time_s": fusion["train_time_s"]}}
    for mode in params["ablation"]["modes"]:
        results[mode] = train_variant(mode, params, train, val, test)
    save_json(results, REPORTS_DIR / "ablation.json")

    lines = ["| Variant | Precision | Recall | F1 | ROC-AUC | FNR |", "|---|---|---|---|---|---|"]
    names = {"fusion": "Fusion (text + features)", "text_only": "Text only (DistilBERT)",
             "features_only": "Features only (MLP)"}
    for mode, r in results.items():
        lines.append(f"| {names.get(mode, mode)} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                     f"{r['roc_auc']:.4f} | {r['fnr']:.4f} |")
    (REPORTS_DIR / "ablation.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    labels = [names.get(m, m).split(" (")[0] for m in results]
    ax[0].bar(labels, [r["f1"] for r in results.values()], color=["#0f766e", "#64748b", "#94a3b8"])
    ax[0].set(title="F1 (higher is better)", ylim=(min(r["f1"] for r in results.values()) - 0.01, 1.0))
    ax[1].bar(labels, [r["fnr"] for r in results.values()], color=["#0f766e", "#64748b", "#94a3b8"])
    ax[1].set(title="False-negative rate (lower is better)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "ablation.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
