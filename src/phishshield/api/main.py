"""PhishShield AI - FastAPI service."""

from __future__ import annotations

import csv
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from phishshield import __version__
from phishshield.api.predictor import PhishPredictor
from phishshield.api.schemas import BatchRequest, BatchResponse, FeedbackRequest, Prediction, URLRequest
from phishshield.config import PROD_DIR, ROOT
from phishshield.registry import model_registry as reg
from phishshield.utils.common import get_logger

log = get_logger("api")
WEB_DIR = ROOT / "web"
_lock = threading.Lock()

try:
    from prometheus_client import Counter, Gauge, Histogram
    from prometheus_fastapi_instrumentator import Instrumentator

    PRED_COUNTER = Counter("phishshield_predictions_total", "Predictions served", ["prediction", "risk_level"])
    PROB_HIST = Histogram("phishshield_phishing_probability", "Distribution of predicted phishing probability",
                          buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    INFER_HIST = Histogram("phishshield_inference_seconds", "Model inference time",
                           buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25])
    FEEDBACK_COUNTER = Counter("phishshield_feedback_total", "User feedback labels received", ["label"])
    MODEL_INFO = Gauge("phishshield_model_info", "Currently served model", ["version"])
    _PROM = True
except ImportError:
    _PROM = False


class State:
    predictor: PhishPredictor | None = None


def load_model() -> PhishPredictor:
    path = os.environ.get("MODEL_DIR") or reg.champion_dir()
    if not path or not Path(path).exists():
        raise RuntimeError("No champion model found. Run the training pipeline (`dvc repro`) first.")
    pred = PhishPredictor(path)
    if _PROM:
        MODEL_INFO.clear()
        MODEL_INFO.labels(version=pred.version).set(1)
    log.info("Loaded model %s from %s", pred.version, path)
    return pred


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        State.predictor = load_model()
    except Exception as e:
        log.error("Model not loaded: %s", e)
    yield


app = FastAPI(title="PhishShield AI", version=__version__, lifespan=lifespan,
              description="Drift-aware hybrid transformer framework for adaptive phishing URL detection.")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)
if _PROM:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _predictor() -> PhishPredictor:
    if State.predictor is None:
        raise HTTPException(503, "Model not loaded")
    return State.predictor


def _log_predictions(preds: list[dict]) -> None:
    """Production log -> input for the drift monitor (data/production/predictions.csv)."""
    try:
        PROD_DIR.mkdir(parents=True, exist_ok=True)
        f = PROD_DIR / "predictions.csv"
        with _lock:
            new = not f.exists()
            with open(f, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["timestamp", "url", "phishing_probability", "prediction", "model_version"])
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for p in preds:
                    w.writerow([ts, p["url"], p["phishing_probability"], p["prediction"], p["model_version"]])
    except OSError as e:
        log.warning("Could not log predictions: %s", e)


def _observe(preds: list[dict], seconds: float) -> None:
    if not _PROM:
        return
    INFER_HIST.observe(seconds)
    for p in preds:
        PRED_COUNTER.labels(p["prediction"], p["risk_level"]).inc()
        PROB_HIST.observe(p["phishing_probability"])


@app.get("/", include_in_schema=False)
def index():
    page = WEB_DIR / "index.html"
    return FileResponse(page) if page.exists() else JSONResponse({"service": "PhishShield AI", "docs": "/docs"})


@app.get("/health")
def health():
    p = State.predictor
    return {"status": "ok" if p else "degraded", "model_loaded": p is not None,
            "model_version": p.version if p else None, "api_version": __version__}


@app.get("/model")
def model_info():
    p = _predictor()
    idx = reg.load_index()
    return {"version": p.version, "threshold": p.threshold, "temperature": p.temperature,
            "max_len": p.max_len, "registry_champion": idx.get("champion"),
            "metrics": idx.get("versions", {}).get(p.version, {}).get("metrics"),
            "history": idx.get("history", [])[-10:]}


@app.post("/predict", response_model=Prediction)
def predict(req: URLRequest):
    t0 = time.perf_counter()
    preds = _predictor().predict([req.url])
    _observe(preds, time.perf_counter() - t0)
    _log_predictions(preds)
    return preds[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    preds = _predictor().predict(req.urls)
    _observe(preds, time.perf_counter() - t0)
    _log_predictions(preds)
    return {"predictions": preds}


@app.post("/explain")
def explain(req: URLRequest):
    p = _predictor()
    out = p.explain(req.url)
    out["threshold"] = p.threshold
    out["prediction"] = "phishing" if out["phishing_probability"] >= p.threshold else "legitimate"
    out["model_version"] = p.version
    return out


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Analyst / user-reported label. These labelled URLs feed the next drift check and update."""
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    f = PROD_DIR / "feedback.csv"
    with _lock:
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["timestamp", "url", "label"])
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), req.url, req.label])
    if _PROM:
        FEEDBACK_COUNTER.labels(str(req.label)).inc()
    return {"status": "recorded"}

@app.post("/admin/reload")
def reload_model(x_admin_token: str | None = Header(default=None)):
    """Hot-reload the champion after a promotion or rollback."""
    expected = os.environ.get("ADMIN_TOKEN")

    if not expected:
        raise HTTPException(503, "Admin authentication is not configured")

    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(401, "Invalid admin token")

    State.predictor = load_model()
    return {"status": "reloaded", "model_version": State.predictor.version}
