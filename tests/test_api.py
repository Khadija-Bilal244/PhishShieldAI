"""API tests. Builds a throw-away ONNX model so the tests never depend on a trained pipeline."""

import json

import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from phishshield.config import load_params  # noqa: E402
from phishshield.features.lexical import FEATURE_NAMES, to_matrix  # noqa: E402
from phishshield.models.student import build_student  # noqa: E402
from phishshield.training.distill import export_onnx  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    cfg = load_params()["student"]
    model = build_student(cfg, len(FEATURE_NAMES))
    export_onnx(model, d / "student.onnx", cfg["max_len"], len(FEATURE_NAMES))
    joblib.dump(StandardScaler().fit(to_matrix(["http://a.com", "https://b.org/x?y=1", "http://1.2.3.4/z"])),
                d / "scaler.joblib")
    (d / "config.json").write_text(json.dumps({"max_len": cfg["max_len"], "temperature": 1.0,
                                               "threshold": 0.5, "version": "test"}))
    np.save(d / "shap_background.npy", np.zeros((10, len(FEATURE_NAMES)), dtype=np.float32))
    import os

    os.environ["MODEL_DIR"] = str(d)
    import phishshield.api.main as main

    main.PROD_DIR = tmp_path_factory.mktemp("prod")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_version"] == "test"


def test_predict(client):
    r = client.post("/predict", json={"url": "http://paypa1-login.xyz/verify"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["phishing_probability"] <= 1
    assert body["prediction"] in {"phishing", "legitimate"}
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_adds_scheme(client):
    assert client.post("/predict", json={"url": "example.com"}).json()["url"] == "http://example.com"


def test_validation_errors(client):
    assert client.post("/predict", json={"url": ""}).status_code == 422
    assert client.post("/predict", json={"url": "a" * 3000}).status_code == 422
    assert client.post("/predict/batch", json={"urls": []}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"urls": ["http://a.com", "http://b.com"]}).json()
    assert len(r["predictions"]) == 2


def test_explain(client):
    r = client.post("/explain", json={"url": "http://192.168.1.1/paypal/login.php"}).json()
    assert r["top_features"] and r["top_segments"]
    assert r["feature_method"] in {"shap", "occlusion"}
    assert {"feature", "value", "importance"} <= set(r["top_features"][0])


def test_feedback_and_metrics(client):
    assert client.post("/feedback", json={"url": "http://x.com", "label": 1}).json()["status"] == "recorded"
    assert client.post("/feedback", json={"url": "http://x.com", "label": 5}).status_code == 422
    m = client.get("/metrics")
    assert m.status_code == 200 and "phishshield_predictions_total" in m.text


def test_onnx_output_is_deterministic(client):
    a = client.post("/predict", json={"url": "http://z.com"}).json()["phishing_probability"]
    b = client.post("/predict", json={"url": "http://z.com"}).json()["phishing_probability"]
    assert np.isclose(a, b)

def test_admin_reload_requires_configured_token(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    response = client.post("/admin/reload")
    assert response.status_code == 503


def test_admin_reload_rejects_invalid_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-secret")
    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "wrong-token"},
    )
    assert response.status_code == 401


def test_admin_reload_accepts_valid_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-secret")
    response = client.post(
        "/admin/reload",
        headers={"X-Admin-Token": "test-secret"},
    )
    assert response.status_code == 200
    assert response.json()["model_version"] == "test"
