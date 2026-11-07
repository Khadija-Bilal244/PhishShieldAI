"""DVC stage `train_hybrid`: train the MAIN model (hybrid DistilBERT + engineered-feature fusion).

Logged to MLflow: all hyper-parameters, per-epoch loss / validation metrics, test metrics,
calibration (temperature, threshold), a training-curve figure, and the model itself as an
MLflow pyfunc model (registered in the `register` stage).

GPU strongly recommended for the real DistilBERT (Colab / Kaggle - see docs/SETUP.md).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phishshield.config import MODELS_DIR, REPORTS_DIR, ROOT, ensure_dirs, load_params, set_seed  # noqa: E402
from phishshield.evaluation.metrics import best_threshold, classification_metrics  # noqa: E402
from phishshield.features.lexical import FEATURE_NAMES  # noqa: E402
from phishshield.models.hybrid import predict_logits, save_hybrid  # noqa: E402
from phishshield.training.common import Split, fit_temperature, load_split, sigmoid  # noqa: E402
from phishshield.training.hybrid_trainer import build_hybrid, fit_hybrid  # noqa: E402
from phishshield.utils import tracking  # noqa: E402
from phishshield.utils.common import get_logger, save_json  # noqa: E402

log = get_logger("train_hybrid")


def cap(split: Split, n: int | None, seed: int) -> Split:
    if not n or len(split) <= n:
        return split
    idx = np.random.default_rng(seed).choice(len(split), n, replace=False)
    return split.subset(np.sort(idx))


def plot_history(history: list[dict], path) -> None:
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.loss, marker="o")
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(h.epoch, h.val_f1, marker="o", label="val F1")
    ax[1].plot(h.epoch, h.val_recall, marker="o", label="val recall")
    ax[1].set(title="Validation metrics", xlabel="epoch")
    ax[1].legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


def log_pyfunc_model(model_dir, input_urls: list[str]) -> None:
    import mlflow.pyfunc

    from phishshield.models.pyfunc import HybridPhishPyfunc

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=HybridPhishPyfunc(),
        artifacts={"model_dir": str(model_dir), "scaler": str(MODELS_DIR / "scaler.joblib")},
        code_path=[str(ROOT / "src" / "phishshield")],
        input_example=pd.DataFrame({"url": input_urls}),
        pip_requirements=["torch", "transformers", "scikit-learn", "pandas", "numpy", "joblib"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, help="fusion | text_only | features_only")
    args = ap.parse_args()
    params = load_params()
    ensure_dirs()
    set_seed(params["seed"])
    cfg = params["hybrid"]
    mode = args.mode or cfg["mode"]
    train = cap(load_split("train"), cfg["max_train_samples"], params["seed"])
    val, test = load_split("val"), load_split("test")
    out_dir = MODELS_DIR / "hybrid"

    with tracking.run(f"hybrid-{mode}", params["mlflow"]["experiment"],
                      {"model": "hybrid_distilbert_fusion", "mode": mode, "n_train": len(train), **cfg},
                      {"stage": "main_model", "model_family": "hybrid_distilbert"}):
        model = build_hybrid(cfg, len(FEATURE_NAMES), mode)
        n_params = sum(p.numel() for p in model.parameters())
        summary = fit_hybrid(model, train, val, cfg)

        val_logits = predict_logits(model, val.urls, val.Xs)
        temperature = fit_temperature(val_logits, val.y)
        threshold = best_threshold(val.y, sigmoid(val_logits / temperature), params["evaluation"]["max_fnr"])
        test_prob = sigmoid(predict_logits(model, test.urls, test.Xs) / temperature)
        test_m = classification_metrics(test.y, test_prob, threshold)
        tracking.log_metrics(test_m, prefix="test_")
        tracking.log_metrics({"temperature": temperature, "threshold": threshold, "n_params": n_params,
                              "train_time_s": summary["train_time_s"]})

        meta = {"pretrained_name": cfg["pretrained_name"], "debug_tiny": cfg["debug_tiny"], "max_len": cfg["max_len"],
                "n_features": len(FEATURE_NAMES), "fusion_dim": cfg["fusion_dim"], "dropout": cfg["dropout"],
                "mode": mode, "temperature": temperature, "threshold": threshold, "n_params": n_params}
        save_hybrid(model, out_dir, meta)

        fig = REPORTS_DIR / "figures" / "hybrid_training_curve.png"
        plot_history(summary["history"], fig)
        tracking.log_artifact(str(fig), "figures")
        if tracking.tracking_enabled():
            log_pyfunc_model(out_dir, test.urls[:3])
        run_id = tracking.active_run_id()

    save_json({"mode": mode, "test": test_m, "temperature": temperature, "threshold": threshold,
               "n_params": n_params, "train_time_s": summary["train_time_s"], "history": summary["history"]},
              REPORTS_DIR / "hybrid_metrics.json")
    save_json({"hybrid": run_id}, REPORTS_DIR / "runs" / "hybrid.json")
    log.info("[%s] test F1=%.4f recall=%.4f FNR=%.4f", mode, test_m["f1"], test_m["recall"], test_m["fnr"])


if __name__ == "__main__":
    main()
