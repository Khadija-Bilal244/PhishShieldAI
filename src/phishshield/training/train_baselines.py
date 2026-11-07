"""DVC stage `train_baselines`: Logistic Regression, Random Forest and XGBoost on engineered features.

Each model is one MLflow run (params, validation metrics, the fitted model as an MLflow sklearn
model). Run IDs are written to reports/runs/baselines.json for the `register` stage.
"""

from __future__ import annotations

import time

import joblib
import pandas as pd

from phishshield.config import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_params, set_seed
from phishshield.evaluation.metrics import classification_metrics
from phishshield.features.lexical import FEATURE_NAMES
from phishshield.models.baselines import build_baselines
from phishshield.training.common import load_split
from phishshield.utils import tracking
from phishshield.utils.common import get_logger, save_json

log = get_logger("train_baselines")


def main() -> None:
    params = load_params()
    ensure_dirs()
    set_seed(params["seed"])
    train, val = load_split("train"), load_split("val")
    out_dir = MODELS_DIR / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, run_ids = {}, {}
    for name, model in build_baselines(params["baselines"], params["seed"]).items():
        with tracking.run(f"baseline-{name}", params["mlflow"]["experiment"],
                          {"model": name, "n_train": len(train), "n_features": len(FEATURE_NAMES),
                           **params["baselines"].get(name, {})},
                          {"stage": "baseline", "model_family": name}):
            t0 = time.perf_counter()
            model.fit(train.X, train.y)
            train_time = time.perf_counter() - t0
            prob = model.predict_proba(val.X)[:, 1]
            m = {**classification_metrics(val.y, prob), "train_time_s": train_time}
            tracking.log_metrics(m, prefix="val_")
            path = out_dir / f"{name}.joblib"
            joblib.dump(model, path)
            if tracking.tracking_enabled():
                import mlflow.sklearn
                from mlflow.models import infer_signature

                X_df = pd.DataFrame(val.X[:5], columns=FEATURE_NAMES)
                mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_df,
                                         signature=infer_signature(X_df, prob[:5]))
            run_ids[name] = tracking.active_run_id()
            results[name] = m
            log.info("%-20s val F1=%.4f recall=%.4f FNR=%.4f (%.1fs)", name, m["f1"], m["recall"], m["fnr"], train_time)
    save_json(results, REPORTS_DIR / "baselines_val.json")
    save_json(run_ids, REPORTS_DIR / "runs" / "baselines.json")


if __name__ == "__main__":
    main()
