# Troubleshooting

| Problem | Fix |
|---|---|
| `dvc status` after copying the new files lists `train_hybrid` or other milestone-1 stages | Your `params.yaml` had different milestone-1 values (e.g. `hybrid.*`, `data.max_rows`). Copy those values back from your old `params.yaml` (`git diff params.yaml`), keep the new sections, and check `dvc status` again |
| `No module named phishshield` | `pip install -e .` |
| `No champion model yet` | Run `dvc repro` (the `promote` stage) or `dvc pull` |
| Warning *"logging to LOCAL"* | `.env` is not filled in, so MLflow isn't pointing to DagsHub |
| `dvc push` 401/403 | Re-run the `dvc remote modify origin --local user/password` lines with a valid token |
| CUDA out of memory (Colab) | `hybrid.batch_size: 16` in `params.yaml`. **Note:** this changes `train_hybrid`, so it retrains. Prefer a fresh Colab session |
| `collect_live` fails | Download manually into `data/live/manual/openphish.txt` and `data/live/manual/tranco.csv` (see docs/02) |
| MLflow stage transition warning | Keep `mlflow<3` from `requirements.txt` |
| Docker build: `models/registry not found` | Run `dvc pull` (or `dvc repro`) before building |
| API shows an old model after an update | `POST /admin/reload` with header `X-Admin-Token` |
| Ports 8000/9090/3000/8080 busy | Change the left number in `ports:` of the compose file |
| Airflow: `network phishnet not found` | Start `docker compose up -d` before the Airflow compose |
| Windows: `make` not found | Run the command inside the Makefile target directly |
| PowerShell blocks `.venv` activation | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| GitHub Actions cannot push or open issues | Settings → Actions → General → Workflow permissions → Read and write |
