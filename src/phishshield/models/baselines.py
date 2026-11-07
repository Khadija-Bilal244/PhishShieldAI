"""Classical baselines (Logistic Regression, Random Forest, XGBoost) on engineered features.

These are BASELINES ONLY - the primary model is the hybrid DistilBERT fusion model.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_baselines(cfg: dict, seed: int) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=cfg["logistic_regression"]["C"],
                                       max_iter=cfg["logistic_regression"]["max_iter"],
                                       class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg["random_forest"]["n_estimators"], max_depth=cfg["random_forest"]["max_depth"],
            class_weight="balanced", n_jobs=-1, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=cfg["xgboost"]["n_estimators"], max_depth=cfg["xgboost"]["max_depth"],
            learning_rate=cfg["xgboost"]["learning_rate"], subsample=cfg["xgboost"]["subsample"],
            colsample_bytree=cfg["xgboost"]["colsample_bytree"], eval_metric="logloss",
            n_jobs=-1, random_state=seed),
    }
