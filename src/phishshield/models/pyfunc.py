"""MLflow pyfunc wrapper so the hybrid model can be logged, registered and loaded with
`mlflow.pyfunc.load_model("models:/phishshield-hybrid/Staging")`.

Input : pandas DataFrame with a `url` column
Output: DataFrame with `phishing_probability` and `prediction` (1 = phishing)
"""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd


class HybridPhishPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from pathlib import Path

        import joblib

        from phishshield.models.hybrid import load_hybrid

        self.model, self.meta = load_hybrid(Path(context.artifacts["model_dir"]))
        self.scaler = joblib.load(context.artifacts["scaler"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        import numpy as np

        from phishshield.features.lexical import to_matrix
        from phishshield.models.hybrid import predict_logits

        urls = model_input["url"].astype(str).tolist()
        feats = self.scaler.transform(to_matrix(urls)).astype(np.float32)
        logits = predict_logits(self.model, urls, feats) / self.meta["temperature"]
        prob = 1 / (1 + np.exp(-logits))
        return pd.DataFrame({"phishing_probability": prob,
                             "prediction": (prob >= self.meta["threshold"]).astype(int)})
