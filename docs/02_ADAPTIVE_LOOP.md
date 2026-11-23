# 02 · The adaptive loop: drift → policy → update → gate → registry

```
collect data ─► drift monitor ─► risk policy ─┬─ none
                                              ├─ recalibrate      (seconds)
                                              ├─ partial fine-tune (minutes)
                                              └─ full retrain      (most expensive)
                                                     │
                                              candidate → MLflow Staging
                                                     │
                                       promotion gate (same held-out data)
                                         pass → Production (old → Archived)
                                         fail → Archived
```

## 1. Get a batch of new URLs

| Source | Command | Notes |
|---|---|---|
| Simulated drift stream | already in `data/stream/` | `batch_00` has no drift, `batch_07` has 60% new styles |
| **Real live data** | `python -m phishshield.data.collect_live` | OpenPhish (phishing) + Tranco (legit) → `data/live/live_YYYYMMDD.csv` |
| User feedback | the web UI's *"Was this wrong?"* buttons → `data/production/feedback.csv` | labelled by people |

If OpenPhish or Tranco are blocked: save https://openphish.com/feed.txt as `data/live/manual/openphish.txt`, and the
Tranco top-1M CSV (from https://tranco-list.eu) as `data/live/manual/tranco.csv`.

## 2. Drift check

```bash
python -m phishshield.drift.monitor --batch data/stream/batch_07.csv
python -m phishshield.drift.monitor --batch data/live/live_20261107.csv
```

The command prints the risk score, the chosen action and the reasons. Full details go to `reports/drift/<batch>_drift.json`
and `reports/drift/latest.json`. The interactive per-feature dashboard is `reports/drift/<batch>_evidently.html`; open it in a browser.

## 3. Execute the decision

```bash
python -m phishshield.pipeline.update --batch data/stream/batch_07.csv
python -m phishshield.pipeline.update --batch data/stream/batch_07.csv --action full_retrain   # force a tier
```

Half the batch is used to adapt the model and the other half is held out for the gate. A full retrain fine-tunes a new
DistilBERT teacher, so use a GPU for that tier (Colab).

## 4. Registry

```bash
python -m phishshield.registry.promote --status      # versions, statuses, history
python -m phishshield.registry.promote --rollback    # previous good champion back to Production
```

In MLflow → **Models → phishshield-student** you'll see the versions moving between Staging, Production and Archived.
The running API picks up a new champion without a restart:

```bash
curl -X POST localhost:8000/admin/reload -H "X-Admin-Token: change-me"
```

## 5. Version the result (important)

Updates change `models/registry` outside `dvc repro`, so snapshot it:

```bash
dvc add models/registry
dvc push
git add models/registry.dvc
git commit -m "Registry after drift update"
git push
```

The Airflow DAG does the `dvc add` and `dvc push` part automatically (`version_registry` task).

## Gate rules (`params.yaml → promotion`)

A candidate is promoted only if all of these hold:
- F1 ≥ champion F1 + `min_improvement`
- recall drop ≤ `max_recall_drop` (1%)
- FNR ≤ `max_fnr` (15%, a hard safety limit)

Rejected versions stay in the registry with their reasons, which is your audit trail.
