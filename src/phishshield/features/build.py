"""DVC stage `featurize`: compute engineered features and fit the scaler on TRAIN only."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phishshield.config import MODELS_DIR, PROCESSED_DIR, ensure_dirs
from phishshield.features.lexical import FEATURE_NAMES, extract_many
from phishshield.utils.common import get_logger

log = get_logger("featurize")


def main() -> None:
    ensure_dirs()
    scaler = StandardScaler()
    for split in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
        feats = extract_many(df["url"])
        if split == "train":
            scaler.fit(feats[FEATURE_NAMES].to_numpy())
        out = pd.concat([df[["url", "label"]].reset_index(drop=True), feats], axis=1)
        out.to_csv(PROCESSED_DIR / f"{split}_features.csv", index=False)
        log.info("%s: %d rows x %d features", split, len(out), len(FEATURE_NAMES))
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")


if __name__ == "__main__":
    main()
