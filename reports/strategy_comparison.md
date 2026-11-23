| Strategy | Precision | Recall | F1 | Worst F1 | ROC-AUC | FNR | Updates (applied by type) | Update cost (s) |
|---|---|---|---|---|---|---|---|---|
| static | 0.7411 | 0.9956 | 0.8419 | 0.7173 | 0.8982 | 0.0044 | 0 (none) | 0.0 |
| fixed_schedule | 0.8747 | 0.9863 | 0.9261 | 0.8687 | 0.9849 | 0.0137 | 4 (4x full_retrain) | 191.8 |
| tiered_drift_aware | 0.8599 | 0.9705 | 0.9081 | 0.8286 | 0.9540 | 0.0295 | 6 (4x recalibrate, 1x partial_finetune) | 13.6 |
