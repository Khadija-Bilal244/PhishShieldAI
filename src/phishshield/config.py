"""Central configuration: project paths and params.yaml loading."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(os.environ.get("PHISHSHIELD_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
STREAM_DIR = DATA_DIR / "stream"
LIVE_DIR = DATA_DIR / "live"
PROD_DIR = Path(os.environ.get("PHISHSHIELD_PROD_DIR", DATA_DIR / "production"))
MODELS_DIR = ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
REPORTS_DIR = ROOT / "reports"


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load params.yaml. Environment overrides."""
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if os.environ.get("PHISHSHIELD_TINY") == "1":
        params["hybrid"]["debug_tiny"] = True
        params["hybrid"]["epochs"] = 5
        params["hybrid"]["lr"] = 1e-3
        params["hybrid"]["max_train_samples"] = params["hybrid"]["max_train_samples"] or 8000
    if os.environ.get("PHISHSHIELD_SAMPLE_SIZE"):
        params["data"]["sample_size"] = int(os.environ["PHISHSHIELD_SAMPLE_SIZE"])
    return params


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "runs",
              REPORTS_DIR / "figures"):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
