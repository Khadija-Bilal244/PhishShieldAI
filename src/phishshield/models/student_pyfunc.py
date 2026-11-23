"""MLflow pyfunc wrapper for the distilled ONNX student, so it can be registered and loaded with."""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd


class StudentPyfunc(mlflow.pyfunc.PythonModel):
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "predictor"}

    def load_context(self, context):
        from phishshield.api.predictor import PhishPredictor

        self.predictor = PhishPredictor(context.artifacts["model_dir"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        out = self.predictor.predict(model_input["url"].astype(str).tolist())
        return pd.DataFrame(out)[["phishing_probability", "prediction", "risk_level"]]
