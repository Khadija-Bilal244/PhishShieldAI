# 03 · Serving (FastAPI + web UI), Docker, Prometheus and Grafana

The API serves the **registry champion** (ONNX student) with ONNX Runtime. There's no torch in the serving image.

## Run locally (no Docker)

```bash
uvicorn phishshield.api.main:app --reload --port 8000
```

| URL | What |
|---|---|
| http://localhost:8000 | **Web UI**: verdict, risk level, highlighted URL segments, SHAP feature contributions, feedback buttons, batch tab |
| http://localhost:8000/docs | Swagger, to try every endpoint |
| `POST /predict` · `POST /predict/batch` | probability, label, risk level (low/medium/high), model version |
| `POST /explain` | segment occlusion + **SHAP** feature contributions (permutation-sampling Shapley values, low memory) |
| `POST /feedback` | labelled report → `data/production/feedback.csv` (input for the drift loop) |
| `GET /model` | champion version, metrics, recent registry history |
| `POST /admin/reload` | hot-swap to the new champion (header `X-Admin-Token`) |
| `GET /metrics` | Prometheus metrics |

## Docker

Run `dvc pull` first: the image copies `models/registry`.

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
python scripts/load_test.py --n 500 --period 0       # older-style traffic
python scripts/load_test.py --n 500 --period 5       # drifted traffic: watch Grafana change
docker compose down
```

| Service | URL |
|---|---|
| API + UI | http://localhost:8000 |
| Prometheus | http://localhost:9090 (Status → Targets, Alerts) |
| Grafana | http://localhost:3000 (admin/admin) → Dashboards → PhishShield |

The API image (`docker/Dockerfile.api`) is slim: no torch, a non-root user, a HEALTHCHECK, and layer caching (requirements are installed
before the code is copied). `models/registry` is bind-mounted read-only, so after a promotion or rollback, `POST /admin/reload` serves the new model.

## Metrics and alerts

Custom metrics: `phishshield_predictions_total{prediction,risk_level}`, `phishshield_phishing_probability` (histogram,
whose mean is a cheap drift signal), `phishshield_inference_seconds`, `phishshield_feedback_total`, `phishshield_model_info{version}`.

Alerts (`monitoring/alert_rules.yml`):
- `PhishShieldApiDown`
- `HighInferenceLatency` (p95 > 50 ms)
- `PhishingRateShift` (prediction mix moved by more than 20 points against the last day, a drift proxy)
- `HighErrorRate`

To try one: `docker compose stop api`, and after about a minute `PhishShieldApiDown` fires.
