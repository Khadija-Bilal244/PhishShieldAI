"""DVC stage `register`: register models in the MLflow Model Registry (DagsHub) and move the
main model to the `Staging` stage.

  phishshield-hybrid            <- main model (hybrid DistilBERT fusion)  -> stage: Staging
  phishshield-baseline-xgboost  <- best baseline, kept for comparison     -> stage: None

Each version gets a description and tags with its test metrics, git commit and data hash.
Promotion to `Production` is deliberately a manual decision (done in the DagsHub/MLflow UI
or with `--promote-production`) after the model is reviewed.

    python -m phishshield.registry.register
    python -m phishshield.registry.register --promote-production
"""

from __future__ import annotations

import argparse

from phishshield.config import REPORTS_DIR, ensure_dirs, load_params
from phishshield.utils import tracking
from phishshield.utils.common import get_logger, load_json, save_json

log = get_logger("register")


def _register(client, mlflow, run_id: str, name: str, description: str, tags: dict) -> str:
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.update_model_version(name, mv.version, description=description)
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    return mv.version


def _set_stage(client, name: str, version: str, stage: str) -> str:
    """Classic registry stage (Staging/Production) + matching alias for newer MLflow UIs."""
    applied = stage
    try:
        client.transition_model_version_stage(name, version, stage, archive_existing_versions=False)
    except Exception as e:  # stages are deprecated in very new MLflow servers -> alias only
        log.warning("Stage transition not supported by server (%s); using alias only", e)
        applied = f"alias:{stage.lower()}"
    try:
        client.set_registered_model_alias(name, stage.lower(), version)
    except Exception as e:
        log.warning("Alias not supported by server: %s", e)
    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote-production", action="store_true",
                    help="move the latest Staging version of the main model to Production")
    args = ap.parse_args()
    params = load_params()
    ensure_dirs()
    cfg = params["mlflow"]
    if not tracking.tracking_enabled():
        log.warning("MLflow disabled - nothing registered.")
        save_json({"status": "skipped (MLflow disabled)"}, REPORTS_DIR / "registry.json")
        return

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking.setup_mlflow(cfg["experiment"])
    client = MlflowClient()

    if args.promote_production:
        versions = client.get_latest_versions(cfg["main_model_name"], stages=["Staging"])
        if not versions:
            raise SystemExit("No Staging version to promote")
        v = versions[0].version
        _set_stage(client, cfg["main_model_name"], v, "Production")
        log.info("%s v%s -> Production", cfg["main_model_name"], v)
        return

    metrics = load_json(REPORTS_DIR / "metrics.json")
    hybrid_run = load_json(REPORTS_DIR / "runs" / "hybrid.json")["hybrid"]
    xgb_run = load_json(REPORTS_DIR / "runs" / "baselines.json")["xgboost"]
    common = {"git_commit": tracking._git_commit(), "data_sha256": tracking._data_hash()}

    hm = metrics["hybrid_distilbert"]
    h_ver = _register(client, mlflow, hybrid_run, cfg["main_model_name"],
                      "Hybrid DistilBERT + engineered-feature gated fusion (main model). "
                      f"Test F1={hm['f1']:.4f}, recall={hm['recall']:.4f}, FNR={hm['fnr']:.4f}, ROC-AUC={hm['roc_auc']:.4f}.",
                      {**common, **{f"test_{k}": v for k, v in hm.items()}})
    h_stage = _set_stage(client, cfg["main_model_name"], h_ver, cfg["registry_stage"])

    xm = metrics["xgboost"]
    x_ver = _register(client, mlflow, xgb_run, cfg["baseline_model_name"],
                      f"XGBoost baseline on 31 engineered URL features. Test F1={xm['f1']:.4f}, FNR={xm['fnr']:.4f}.",
                      {**common, **{f"test_{k}": v for k, v in xm.items()}})

    result = {"tracking_uri": tracking.tracking_uri(),
              "main_model": {"name": cfg["main_model_name"], "version": h_ver, "stage": h_stage, "run_id": hybrid_run},
              "baseline_model": {"name": cfg["baseline_model_name"], "version": x_ver, "stage": "None", "run_id": xgb_run}}
    save_json(result, REPORTS_DIR / "registry.json")
    log.info("Registered %s v%s (%s) and %s v%s", cfg["main_model_name"], h_ver, h_stage, cfg["baseline_model_name"], x_ver)


if __name__ == "__main__":
    main()
