"""MLflow tracking helpers (remote DagsHub server).

Credentials are read from the environment or from `.env` in the project root:
    MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME=<user>
    MLFLOW_TRACKING_PASSWORD=<dagshub token>

Every run is tagged with the git commit, the DVC data hash (SHA-256 from the ingest stage)
and the pipeline stage, so any MLflow run can be traced back to the exact code + data.
Set PHISHSHIELD_DISABLE_MLFLOW=1 to switch tracking off (unit tests / CI).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Iterator
from typing import Any

from phishshield.config import REPORTS_DIR, ROOT
from phishshield.utils.common import get_logger, load_json

log = get_logger("tracking")

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None
    _HAS_MLFLOW = False

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def tracking_enabled() -> bool:
    return _HAS_MLFLOW and os.environ.get("PHISHSHIELD_DISABLE_MLFLOW") != "1"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def setup_mlflow(experiment: str) -> None:
    if not tracking_enabled():
        return
    uri = tracking_uri()
    if not uri.startswith("http"):
        log.warning("MLFLOW_TRACKING_URI is not set -> logging to LOCAL %s. "
                    "The assignment requires the DagsHub server: fill in .env (see docs/SETUP.md).", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _data_hash() -> str:
    try:
        return load_json(REPORTS_DIR / "ingest.json")["sha256"]
    except Exception:
        return "unknown"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


@contextlib.contextmanager
def run(name: str, experiment: str, params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None) -> Iterator[Any]:
    """`with run("baseline-xgboost", exp, params) as r:` - yields the active run (or None when disabled)."""
    if not tracking_enabled():
        yield None
        return
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=name) as active:
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in _flatten(params).items()})
        mlflow.set_tags({"project": "phishshield-ai", "git_commit": _git_commit(),
                         "data_sha256": _data_hash(), **(tags or {})})
        yield active


def log_metrics(metrics: dict[str, float], prefix: str = "", step: int | None = None) -> None:
    if not tracking_enabled() or mlflow.active_run() is None:
        return
    clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()
             if isinstance(v, (int, float)) and v == v}  # skip NaN
    mlflow.log_metrics(clean, step=step)


def log_artifact(path: str, artifact_path: str | None = None) -> None:
    if not tracking_enabled() or mlflow.active_run() is None:
        return
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as e:  # never let an upload hiccup kill a training run
        log.warning("Could not upload artifact %s: %s", path, e)


def active_run_id() -> str | None:
    if not tracking_enabled() or mlflow.active_run() is None:
        return None
    return mlflow.active_run().info.run_id
