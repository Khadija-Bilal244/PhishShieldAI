"""DVC stage `split`: stratified train / validation / test split (70 / 15 / 15 by default)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from phishshield.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, load_params
from phishshield.utils.common import get_logger, save_json

log = get_logger("split")


def stratified_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> dict[str, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    params = load_params()
    ensure_dirs()
    df = pd.read_csv(INTERIM_DIR / "urls_clean.csv")
    parts = stratified_split(df, params["split"]["test_size"], params["split"]["val_size"], params["seed"])
    summary = {}
    for name, part in parts.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        summary[name] = {"rows": len(part), "phishing_ratio": round(float(part["label"].mean()), 4)}
        log.info("%-5s %6d rows (%.1f%% phishing)", name, len(part), 100 * part["label"].mean())
    save_json(summary, REPORTS_DIR / "split_summary.json")


if __name__ == "__main__":
    main()
