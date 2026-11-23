"""Versioned serving-model registry with validation gate, promotion and rollback (proposal 5.7)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from phishshield.config import MODELS_DIR, REGISTRY_DIR, ROOT, load_params
from phishshield.evaluation.metrics import classification_metrics
from phishshield.utils import tracking
from phishshield.utils.common import get_logger, load_json, save_json

log = get_logger("registry")
INDEX = REGISTRY_DIR / "registry.json"
FILES = ("student.onnx", "student.pt", "config.json", "scaler.joblib", "shap_background.npy")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_index() -> dict:
    return load_json(INDEX) if INDEX.exists() else {"champion": None, "versions": {}, "history": []}


def save_index(idx: dict) -> None:
    save_json(idx, INDEX)


def champion_dir() -> Path | None:
    idx = load_index()
    return REGISTRY_DIR / idx["champion"] if idx.get("champion") else None


def teacher_dir() -> Path:
    """Teacher matching the current champion (an update may have retrained it), else the pipeline teacher."""
    c = champion_dir()
    return c / "teacher" if c and (c / "teacher" / "model.pt").exists() else MODELS_DIR / "hybrid"


def register(src_dir: Path, reason: str) -> str:
    idx = load_index()
    version = f"v{len(idx['versions']) + 1}"
    dst = REGISTRY_DIR / version
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        src = src_dir / f
        if not src.exists() and f == "scaler.joblib":
            src = MODELS_DIR / "scaler.joblib"
        if src.exists():
            shutil.copy2(src, dst / f)
    if (src_dir / "teacher").exists():
        shutil.copytree(src_dir / "teacher", dst / "teacher", dirs_exist_ok=True)
    cfg = json.loads((dst / "config.json").read_text())
    cfg["version"] = version
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    idx["versions"][version] = {"created": _now(), "reason": reason, "status": "candidate", "metrics": {}}
    mv = _mlflow_register(dst, version, reason)
    if mv:
        idx["versions"][version]["mlflow_version"] = mv
    save_index(idx)
    log.info("Registered %s (%s)%s", version, reason, f" -> MLflow v{mv} Staging" if mv else "")
    return version


def evaluate_version(version_dir: Path, urls: list[str], y) -> dict:
    from phishshield.api.predictor import PhishPredictor

    pred = PhishPredictor(version_dir)
    probs, _ = pred.predict_proba(urls)
    return classification_metrics(y, probs, pred.threshold)


def gate(candidate: dict, champion: dict | None, cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["fnr"] > cfg["max_fnr"]:
        reasons.append(f"FNR {candidate['fnr']:.4f} exceeds safety limit {cfg['max_fnr']}")
    if champion is not None:
        pm = cfg["primary_metric"]
        if candidate[pm] < champion[pm] + cfg["min_improvement"]:
            reasons.append(f"{pm} {candidate[pm]:.4f} < champion {champion[pm]:.4f} + {cfg['min_improvement']}")
        if champion["recall"] - candidate["recall"] > cfg["max_recall_drop"]:
            reasons.append(f"recall drop {champion['recall'] - candidate['recall']:.4f} > {cfg['max_recall_drop']}")
    return (not reasons), reasons


def promote(version: str, note: str = "") -> None:
    idx = load_index()
    prev = idx.get("champion")
    if prev:
        idx["versions"][prev]["status"] = "archived"
    idx["champion"] = version
    idx["versions"][version]["status"] = "champion"
    idx["history"].append({"time": _now(), "event": "promote", "from": prev, "to": version, "note": note})
    save_index(idx)
    _mlflow_stage(idx, version, "Production", archive_previous=True)
    log.info("Promoted %s -> champion/Production (previous: %s)", version, prev)


def reject(version: str, reasons: list[str]) -> None:
    idx = load_index()
    idx["versions"][version].update({"status": "rejected", "rejection_reasons": reasons})
    idx["history"].append({"time": _now(), "event": "reject", "version": version, "reasons": reasons})
    save_index(idx)
    _mlflow_stage(idx, version, "Archived")
    log.warning("Rejected %s: %s", version, "; ".join(reasons))


def rollback() -> str | None:
    """Step back to the most recent earlier champion that has not itself been rolled back or rejected."""
    idx = load_index()
    current = idx["champion"]
    past = [h["to"] for h in idx["history"] if h["event"] in ("promote", "rollback")]
    target = next((v for v in reversed(past) if v != current
                   and idx["versions"][v]["status"] not in ("rolled_back", "rejected")), None)
    if target is None:
        log.warning("Nothing to roll back to")
        return None
    idx["versions"][current]["status"] = "rolled_back"
    idx["versions"][target]["status"] = "champion"
    idx["champion"] = target
    idx["history"].append({"time": _now(), "event": "rollback", "from": current, "to": target})
    save_index(idx)
    _mlflow_stage(idx, target, "Production", archive_previous=True)
    log.info("Rolled back %s -> %s", current, target)
    return target


def _model_name() -> str:
    return load_params()["registry"]["student_model_name"]


def _mlflow_register(path: Path, version: str, reason: str) -> str | None:
    if not tracking.tracking_enabled():
        return None
    try:
        import mlflow
        import mlflow.pyfunc
        import pandas as pd

        from phishshield.models.student_pyfunc import StudentPyfunc

        params = load_params()
        tracking.setup_mlflow(params["mlflow"]["experiment"])
        with mlflow.start_run(run_name=f"registry-{version}", nested=mlflow.active_run() is not None):
            mlflow.set_tags({"stage": "registry", "local_version": version, "reason": reason})
            mlflow.pyfunc.log_model(artifact_path="model", python_model=StudentPyfunc(),
                                    artifacts={"model_dir": str(path)}, code_path=[str(ROOT / "src" / "phishshield")],
                                    input_example=pd.DataFrame({"url": ["http://example.com"]}),
                                    pip_requirements=["onnxruntime", "scikit-learn", "pandas", "numpy", "joblib"])
            uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mv = mlflow.register_model(uri, _model_name())
        client = mlflow.MlflowClient()
        client.set_model_version_tag(_model_name(), mv.version, "local_version", version)
        client.update_model_version(_model_name(), mv.version, description=f"{version}: {reason}")
        client.transition_model_version_stage(_model_name(), mv.version, "Staging")
        return str(mv.version)
    except Exception as e:
        log.warning("MLflow registry mirror failed (local registry still updated): %s", e)
        return None


def _mlflow_stage(idx: dict, version: str, stage: str, archive_previous: bool = False) -> None:
    mv = idx["versions"].get(version, {}).get("mlflow_version")
    if not mv or not tracking.tracking_enabled():
        return
    try:
        import mlflow

        tracking.setup_mlflow(load_params()["mlflow"]["experiment"])
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(_model_name(), mv, stage, archive_existing_versions=archive_previous)
        try:
            client.set_registered_model_alias(_model_name(), "champion" if stage == "Production" else stage.lower(), mv)
        except Exception:
            pass
    except Exception as e:
        log.warning("MLflow stage update failed: %s", e)
