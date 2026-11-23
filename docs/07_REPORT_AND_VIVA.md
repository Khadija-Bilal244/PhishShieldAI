# 07 · Final report, demo and viva

## Where each result comes from

| Report section | File (after `dvc repro`) | Figure |
|---|---|---|
| E1 Baselines vs hybrid vs student | `reports/model_comparison.md`, `reports/student_metrics.json` | `figures/roc_curves.png`, `confusion_matrices.png` |
| E2 Ablation | `reports/ablation.md` | `figures/ablation.png` |
| E3 Distillation | `reports/final_results.md` (E3 block) | — |
| E4 Drift detection | `reports/drift_detection.md` | `figures/drift_detection.png`, `reports/drift/batch_07_evidently.html` |
| E5 Update strategies | `reports/strategy_comparison.md` | `figures/strategy_comparison.png` |
| E6 Explainability | `reports/explanations.html`, `reports/shap_importance.json` | `figures/shap_summary.png` |
| Live temporal test | `reports/drift/live_*_drift.json` (F1/FNR on real OpenPhish + Tranco) | — |

`reports/final_results.md` contains all the tables in one place.

## Report structure (suggested)

1. Introduction and motivation
2. Related work: URL classifiers, transformer URL models, drift detection (MMD, classifier two-sample tests), distillation
3. System architecture: the README diagram and the MLOps tool map
4. Methodology: data and validation; hybrid model (gate equation); distillation loss; calibration; drift tests; risk score and tiers; promotion gate
5. Experiments E1–E6, plus the live test
6. MLOps implementation: screenshots of DagsHub, MLflow runs and registry stages, the 4 GitHub Actions workflows, Docker/Grafana, the Airflow graph
7. Discussion and limitations (see below)
8. Conclusion and future work: email-body phishing with the same encoder, active learning for labels, Kubeflow

## Limitations to state honestly

- PhiUSIIL is close to separable (every model F1 > 0.99), so the model comparison differs only in the third decimal. That's why missed phishing (FNR) and latency matter more.
- PhiUSIIL has no timestamps. The temporal check is the live OpenPhish/Tranco data collected after training. The drift stream is a controlled simulation.
- The tiered strategy trades some accuracy for much lower update cost. Fixed-schedule full retraining was the most accurate in our tests. State the trade-off with numbers.
- The risk-score weights and tier thresholds are set by hand. A sensitivity study is optional (`dvc exp run -S drift.tiers.partial_finetune=0.35`).
- The strategy simulation updates the student directly, for CPU cost reasons. Production updates also retrain the teacher.

## 8-minute demo script

1. GitHub: README, the 4 workflow badges and Actions runs, a merged PR.
2. DagsHub: the 16-stage pipeline; MLflow runs; Models with `phishshield-hybrid` in Staging and `phishshield-student` in Production plus Archived versions.
3. The public Render URL (free deployment from CD). Then `docker compose up` locally: web UI, check `http://paypa1-secure-login.xyz/verify`, show the highlighted segments and SHAP features, click feedback.
4. Grafana while `load_test.py --period 5` runs.
5. Airflow: trigger the adaptive loop and show the branch → update → gate → reload, then `/model` showing the new version.
6. Rollback from the command line, then `/admin/reload`.
7. Results slides: E2, E3, E4, E5.

## Likely viva questions

| Question | Short answer |
|---|---|
| Why distil instead of serving DistilBERT? | Size and latency: the student is a few MB in ONNX and runs in under 1 ms with no torch, with high agreement with the teacher. |
| Why calibrate, and why bound the temperature? | Risk levels need real probabilities. On near-separable data, unbounded scaling collapses probabilities to 0 or 1, so T ≥ 0.5 keeps them meaningful. |
| What is embedding drift, and why use it alongside PSI/KS? | Per-feature tests are univariate. MMD and the domain classifier compare the joint, learned representation, so they can catch style changes. |
| How does the policy decide without labels? | It uses the share of uncertain predictions (0.3 < p < 0.7) plus drift signals. Labels, from feedback or live sources, replace that proxy. |
| Why tiers? | Retraining a transformer is expensive. Many shifts are fixed by recalibration or partial fine-tuning, and E5 quantifies the cost/accuracy trade-off. |
| What stops a bad model from shipping? | Same-data comparison with the champion: F1 must not drop, recall may drop at most 1%, and FNR has a hard 15% limit. The MLflow stage trail plus rollback cover the rest. |
| Why is the registry versioned with `dvc add` and not as a pipeline output? | Updates happen outside `dvc repro`. Keeping it separate means the pipeline never overwrites the update history, and every snapshot is still versioned. |
| Why Render instead of AWS? | Every tool in the project is free with no card. Render runs the exact image CD tested from GHCR, and the ONNX student uses about 330 MB, under the 512 MB free limit. Hugging Face Docker Spaces became paid in July 2026. |
| Why your own SHAP in the API? | The `shap` library pushed the server to about 930 MB, too much for the free tier. Permutation-sampling Shapley values give the same top features (correlation 0.88–0.99 with KernelSHAP) at about 330 MB and 0.3 s per explanation. |
| `dvc.yaml` vs `dvc.lock`? | The pipeline definition vs the exact hashes of the last run. |
