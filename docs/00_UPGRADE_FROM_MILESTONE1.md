# 00 · Continue from milestone 1

This zip is your milestone-1 repo plus everything else in the proposal. It contains:

- **All code with comments trimmed.** Every Python file was checked to be functionally identical (same syntax tree).
- **Your `dvc.lock`** with only the code hashes of the milestone-1 stages updated (22 lines), the same thing `dvc commit` does
  after comment-only edits. Your data, trained DistilBERT, metrics and MLflow runs stay valid.
- **Your `reports/`** from the milestone-1 run.

## 1. Copy into your repo

```powershell
cd PhishShieldAI
.venv\Scripts\Activate.ps1
git checkout main
git pull
git checkout -b milestone2
```

Unzip `PhishShieldAI.zip` somewhere else. Copy **everything inside its `PhishShieldAI` folder** into your repo folder and
choose **Replace**. That includes hidden files such as `.github`, `.gitignore` and `.gitattributes`. Your `.git`, `.dvc`, `.env`, `data`
and `models` are not in the zip, so they are kept.

## 2. Check

```powershell
pip install -r requirements-dev.txt
pip install -e .
pytest
dvc pull
dvc status
```

- `pytest` should show **51 passed**.
- `dvc status` must list only the 8 new stages: `ablation, distill, explain, promote, build_stream, drift_eval, strategy_comparison, summary`.

If a milestone-1 stage such as `train_hybrid` also appears, don't run `dvc repro`. Run `dvc commit train_hybrid` (with the stage name
that appeared), because the only change is comments.

## 3. Commit and push

```powershell
git add -A
git commit -m "Milestone 2: ablation, distillation, explainability, drift loop, registry, serving, CI/CD"
git push -u origin milestone2
```

On GitHub, open a **Pull Request** from `milestone2` into `main`, wait for **CI**, then **Merge**.

## 4. Run the new stages on a GPU (Colab)

1. Upload `notebooks/colab_run_pipeline.ipynb` to Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells in order. `dvc repro` runs only the 8 new stages, in about 30–45 minutes.
3. The last cell runs `dvc add models/registry`, `dvc push` and `git push`.

## 5. Sync and check

```powershell
git pull
dvc pull
dvc status
```

`dvc status` should say **Data and pipelines are up to date**. `reports/final_results.md` has every result table.

## 6. Check on DagsHub / MLflow

| Where | Expect |
|---|---|
| MLflow → experiment `phishshield-ai` | new runs: `ablation-*`, `distill-student`, `explainability`, `registry-v1`, `drift-evaluation`, `strategy-comparison` |
| MLflow → Models | `phishshield-hybrid` (Staging) and `phishshield-student` v1 (Production) |
| DagsHub → Data Pipeline | 16 stages |

Next guides: [01_PIPELINE](01_PIPELINE.md) · [02_ADAPTIVE_LOOP](02_ADAPTIVE_LOOP.md) ·
[03_SERVING_DOCKER_MONITORING](03_SERVING_DOCKER_MONITORING.md) · [04_AIRFLOW](04_AIRFLOW.md) ·
[05_GITHUB_ACTIONS](05_GITHUB_ACTIONS.md) · [06_FREE_DEPLOYMENT_K8S](06_FREE_DEPLOYMENT_K8S.md) ·
[07_REPORT_AND_VIVA](07_REPORT_AND_VIVA.md) · [TROUBLESHOOTING](TROUBLESHOOTING.md)
