"""DVC stage `evaluate`: compare the baselines with the main hybrid model on the held-out TEST set.

Outputs (also logged as artifacts of an MLflow `evaluation` run):
  reports/metrics.json            DVC metrics (`dvc metrics show` / `dvc metrics diff`)
  reports/model_comparison.md     table for the write-up
  reports/figures/roc_curves.png, confusion_matrices.png
"""

from __future__ import annotations

import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402

from phishshield.config import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_params  # noqa: E402
from phishshield.evaluation.metrics import classification_metrics  # noqa: E402
from phishshield.models.hybrid import load_hybrid, predict_logits  # noqa: E402
from phishshield.training.common import load_split, sigmoid  # noqa: E402
from phishshield.utils import tracking  # noqa: E402
from phishshield.utils.common import get_logger, save_json  # noqa: E402

log = get_logger("evaluate")
FIG = REPORTS_DIR / "figures"
BASELINES = ("logistic_regression", "random_forest", "xgboost")


def _latency_ms(fn, n: int = 20) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    params = load_params()
    ensure_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    test = load_split("test")
    probs, thresholds, latency = {}, {}, {}

    for name in BASELINES:
        m = joblib.load(MODELS_DIR / "baselines" / f"{name}.joblib")
        probs[name] = m.predict_proba(test.X)[:, 1]
        thresholds[name] = 0.5
        latency[name] = _latency_ms(lambda m=m: m.predict_proba(test.X[:1]))

    hybrid, meta = load_hybrid(MODELS_DIR / "hybrid")
    probs["hybrid_distilbert"] = sigmoid(predict_logits(hybrid, test.urls, test.Xs) / meta["temperature"])
    thresholds["hybrid_distilbert"] = meta["threshold"]
    latency["hybrid_distilbert"] = _latency_ms(lambda: predict_logits(hybrid, test.urls[:1], test.Xs[:1]), 10)

    results = {n: {**classification_metrics(test.y, p, thresholds[n]), "latency_ms": latency[n]} for n, p in probs.items()}

    keep = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fnr", "fpr", "latency_ms")
    save_json({n: {k: round(float(r[k]), 5) for k in keep} for n, r in results.items()}, REPORTS_DIR / "metrics.json")
    lines = ["| Model | Role | Accuracy | Precision | Recall | F1 | ROC-AUC | FNR | Latency (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        role = "main" if n == "hybrid_distilbert" else "baseline"
        lines.append(f"| {n} | {role} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                     f"{r['roc_auc']:.4f} | {r['fnr']:.4f} | {r['latency_ms']:.2f} |")
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n")

    plt.figure(figsize=(6, 5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(test.y, p)
        plt.plot(fpr, tpr, label=f"{n} (AUC={results[n]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves - test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "roc_curves.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, n in zip(axes, ("xgboost", "hybrid_distilbert")):
        ConfusionMatrixDisplay.from_predictions(test.y, (probs[n] >= thresholds[n]).astype(int), ax=ax,
                                                display_labels=["legit", "phish"], colorbar=False)
        ax.set_title(n)
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrices.png", dpi=130)
    plt.close()

    with tracking.run("evaluation-test-set", params["mlflow"]["experiment"], {"n_test": len(test)},
                      {"stage": "evaluation"}):
        for n, r in results.items():
            tracking.log_metrics({k: r[k] for k in keep}, prefix=f"{n}.")
        for f in ("metrics.json", "model_comparison.md", "figures/roc_curves.png", "figures/confusion_matrices.png"):
            tracking.log_artifact(str(REPORTS_DIR / f), "evaluation")

    for n, r in results.items():
        log.info("%-20s F1=%.4f recall=%.4f FNR=%.4f AUC=%.4f %.2fms", n, r["f1"], r["recall"], r["fnr"],
                 r["roc_auc"], r["latency_ms"])


if __name__ == "__main__":
    main()
