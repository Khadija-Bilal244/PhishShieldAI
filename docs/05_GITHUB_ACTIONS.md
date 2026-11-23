# 05 · GitHub Actions: CI, reproducibility, CD, monitoring

| Workflow | Trigger | What it does |
|---|---|---|
| **CI** `ci.yml` | every push / PR | Ruff lint + 51 pytest tests with coverage |
| **Reproducibility** `reproducibility.yml` | push changing `dvc.lock`/`dvc.yaml`/`params.yaml`, manual | `dvc pull` from DagsHub, then **fails unless `dvc status` is up to date**; posts `dvc metrics show` and `final_results.md` to the job summary |
| **CD** `cd.yml` | after CI succeeds on `main`, manual | pulls the champion (`models/registry.dvc`) → builds the API image → container smoke test (`/health`, `/predict`, `/metrics`) → pushes to **GHCR** → redeploys the free **Render** web service and smoke-tests the public URL |
| **Drift Monitor** `drift-monitor.yml` | daily 08:30 PKT, manual | collects **real live data** → drift check against the champion → uploads the Evidently report → **opens a GitHub issue** when an update is recommended |

## Setup (repo owner, once)

1. GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:
   - `DAGSHUB_USERNAME`
   - `DAGSHUB_TOKEN`
   - `RENDER_DEPLOY_HOOK`, `RENDER_URL` (free Render deployment, see docs/06)
2. **Settings → Actions → General → Workflow permissions → Read and write**, so CD can push to GHCR and the monitor can open issues.
3. Push, then watch the **Actions** tab. Run *Reproducibility*, *CD* and *Drift Monitor* once by hand (**Run workflow**) to get screenshots.

After CD runs, the image is public at `ghcr.io/khadija-bilal244/phishshieldai/phishshield-api:latest`:

```bash
docker run -p 8000:8000 ghcr.io/khadija-bilal244/phishshieldai/phishshield-api:latest
```

## Vocabulary for the viva

| Term | Meaning |
|---|---|
| **CI** | every change is linted and tested automatically |
| **CT / reproducibility** | the committed `dvc.lock` must match the data and models in remote storage, so any commit can be reproduced |
| **CD** | a tested, versioned (commit SHA) image is built, smoke-tested, published and deployed |
| **CM** | real data is checked for drift every day, and humans are alerted with an issue |

Training itself runs on a GPU (Colab or Airflow), not on free CPU runners. That is a deliberate cost decision worth mentioning.
