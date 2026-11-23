# 01 · The pipeline: 16 stages and their outputs

`dvc dag` shows the graph. All parameters are in `params.yaml`.

## Milestone 1 (unchanged)

| Stage | Output |
|---|---|
| `ingest` | PhiUSIIL from UCI → `data/raw/urls.csv`; SHA-256 in `reports/ingest.json` |
| `validate` | schema, null, range, format and duplicate checks; the pipeline fails on bad data (`reports/data_validation.json`) |
| `split` | stratified 70/15/15 split |
| `featurize` | 31 features + scaler (fitted on train only) |
| `train_baselines` | LR / RF / XGBoost, as MLflow runs |
| `train_hybrid` | **main model** (DistilBERT fusion) as an MLflow pyfunc model |
| `evaluate` | test-set comparison → `reports/model_comparison.md` |
| `register` | `phishshield-hybrid` → **Staging** in MLflow |

## Milestone 2+ (new)

| Stage | Proposal | What it does | Outputs |
|---|---|---|---|
| `ablation` | 5.3 | Trains **text-only** and **features-only** variants with the same settings and compares them with fusion | `reports/ablation.md`, `figures/ablation.png`, 2 MLflow runs |
| `distill` | 5.4 | Teacher soft labels → **CharCNN student**, temperature-scaled distillation loss, export to **ONNX**; measures agreement, compression and latency | `models/student/`, `reports/student_metrics.json`, MLflow run + pyfunc model |
| `explain` | features | **Integrated Gradients** (Captum) on teacher tokens; **SHAP** on engineered features | `reports/explanations.html`, `figures/shap_summary.png` |
| `promote` | 5.7 | Registers the student as a new version (MLflow **Staging**), compares it with the champion on the same test data, then promotes it (**Production**) or rejects it (**Archived**) | `models/registry/`, `reports/promotion.json` |
| `build_stream` | 5.8 | 8 labelled batches of **unseen PhiUSIIL URLs**, with the share of newer attack styles rising 0 → 60% | `data/stream/batch_00..07.csv` |
| `drift_eval` | 5.5 / 5.6 | Runs the drift monitor on every batch: PSI/KS share, embedding MMD and domain AUC, F1/FNR, risk score and chosen tier | `reports/drift_detection.md`, `figures/drift_detection.png`, Evidently HTML |
| `strategy_comparison` | 5.8 | Prequential test-then-train over the stream for **static vs fixed-schedule vs tiered drift-aware** | `reports/strategy_comparison.md`, `figures/strategy_comparison.png` |
| `summary` | — | All result tables in one file | `reports/final_results.md` |

## Design notes for the report and viva

- **Distillation loss:** `L = α·T²·BCE(σ(z_s/T), σ(z_t/T)) + (1−α)·BCE(σ(z_s), y)`, with `T = 2` and `α = 0.5`.
- **Calibration:** on near-separable data, unconstrained temperature scaling pushes T → 0, so every probability
  becomes 0 or 1. The student therefore uses a **bounded temperature (T ≥ 0.5)**, and its threshold is chosen among the
  F1-tied candidates as the one closest to 0.5 (`training/calibration.py`). This keeps the risk levels (low/medium/high) meaningful.
- **Embedding drift:** the teacher's text embeddings, compared between a reference sample and the new batch with
  an MMD permutation test (RBF kernel, median heuristic) and a domain classifier (5-fold CV AUC). AUC ≈ 0.5 means no
  drift; AUC → 1 means strong drift.
- **Risk score:** `risk = 0.35·feature_drift_share + 0.35·embedding_score + 0.30·performance_score`. The tiers are
  0.25 / 0.45 / 0.65. A safety override applies when the FNR is above 0.15.
- **The cheap tiers learn from all labels since the last update** plus a replay sample of old data (to prevent
  forgetting). Our first version trained them on a single half-batch, about 150 URLs. That wasn't enough: the gate rejected
  every partial fine-tune. Explaining this design choice is good material for the viva.
- **Stream vs live data:** the stream is a reproducible, *simulated* drift scenario. The live OpenPhish/Tranco snapshots
  are real data, newer than the training data. Report them separately.

## Useful DVC commands

```bash
dvc repro ablation                   # run up to one stage
dvc exp run -S student.alpha=0.7     # experiment with a different param
dvc exp show --only-changed
dvc metrics diff main                # compare with main
dvc plots show                       # figures as HTML
```
