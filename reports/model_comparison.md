| Model | Role | Accuracy | Precision | Recall | F1 | ROC-AUC | FNR | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | baseline | 0.9928 | 0.9988 | 0.9844 | 0.9916 | 0.9971 | 0.0156 | 0.45 |
| random_forest | baseline | 0.9957 | 0.9996 | 0.9903 | 0.9949 | 0.9986 | 0.0097 | 71.18 |
| xgboost | baseline | 0.9965 | 0.9996 | 0.9922 | 0.9959 | 0.9976 | 0.0078 | 0.42 |
| hybrid_distilbert | main | 0.9980 | 1.0000 | 0.9953 | 0.9977 | 0.9989 | 0.0047 | 85.05 |
