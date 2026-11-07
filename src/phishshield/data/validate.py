"""DVC stage `validate`: data-quality gate that runs BEFORE any training.

Checks (thresholds in params.yaml -> validation):
  schema      required columns present, label is integer-like
  nulls       null ratio in required columns <= max_null_ratio (null rows are dropped)
  ranges      label in allowed_labels; URL length in [min_url_length, max_url_length];
              URL has a scheme and host; each class >= min_class_ratio; enough rows
  duplicates  duplicate-URL ratio <= max_duplicate_ratio (duplicates are dropped);
              URLs with conflicting labels are dropped

Writes reports/data_validation.json (also a DVC metric). Exits with code 1 if any hard check
fails, which stops `dvc repro` before a model is trained on bad data.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from phishshield.config import INTERIM_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs, load_params
from phishshield.utils.common import get_logger, save_json

log = get_logger("validate")
URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/?#]+")


def _check(report: dict, name: str, passed: bool, **details) -> None:
    report["checks"][name] = {"passed": bool(passed), **details}


def validate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_in": int(len(df)), "checks": {}}

    # ---------- schema ----------
    missing = [c for c in cfg["required_columns"] if c not in df.columns]
    _check(report, "schema_required_columns", not missing, missing=missing)
    if missing:
        report["passed"] = False
        return df.iloc[0:0], report
    label_numeric = pd.to_numeric(df["label"], errors="coerce")
    _check(report, "schema_label_numeric", label_numeric.notna().mean() >= 1 - cfg["max_null_ratio"],
           non_numeric=int(label_numeric.isna().sum() - df["label"].isna().sum()))
    df = df.assign(label=label_numeric)

    # ---------- nulls ----------
    null_rows = df[cfg["required_columns"]].isna().any(axis=1)
    _check(report, "nulls", null_rows.mean() <= cfg["max_null_ratio"],
           null_rows=int(null_rows.sum()), null_ratio=round(float(null_rows.mean()), 5))
    df = df[~null_rows].copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # ---------- ranges ----------
    bad_label = ~df["label"].isin(cfg["allowed_labels"])
    _check(report, "range_label", not bad_label.any(), out_of_range=int(bad_label.sum()))
    df = df[~bad_label]

    lengths = df["url"].str.len()
    bad_len = (lengths < cfg["min_url_length"]) | (lengths > cfg["max_url_length"])
    _check(report, "range_url_length", bad_len.mean() <= cfg["max_null_ratio"], dropped=int(bad_len.sum()),
           min=int(lengths.min()) if len(lengths) else None, max=int(lengths.max()) if len(lengths) else None,
           mean=round(float(lengths.mean()), 2) if len(lengths) else None)
    df = df[~bad_len]

    # PhiUSIIL stores some URLs without a scheme -> normalise rather than drop
    no_scheme = ~df["url"].str.contains("://", regex=False)
    df.loc[no_scheme, "url"] = "http://" + df.loc[no_scheme, "url"]
    bad_fmt = ~df["url"].str.match(URL_RE)
    _check(report, "format_url", bad_fmt.mean() <= cfg["max_null_ratio"], malformed=int(bad_fmt.sum()),
           scheme_added=int(no_scheme.sum()))
    df = df[~bad_fmt]

    # ---------- duplicates / conflicts ----------
    dup = df.duplicated("url")
    _check(report, "duplicates", dup.mean() <= cfg["max_duplicate_ratio"], duplicates=int(dup.sum()))
    conflicts = df.groupby("url")["label"].nunique()
    conflict_urls = set(conflicts[conflicts > 1].index)
    _check(report, "conflicting_labels", True, conflicting_urls=len(conflict_urls))
    df = df[~df["url"].isin(conflict_urls)].drop_duplicates("url")

    # ---------- dataset-level ----------
    ratio = float(df["label"].mean()) if len(df) else 0.0
    _check(report, "range_class_balance", cfg["min_class_ratio"] <= ratio <= 1 - cfg["min_class_ratio"],
           phishing_ratio=round(ratio, 4))
    _check(report, "min_rows", len(df) >= cfg["min_rows"], rows=int(len(df)))

    report["rows_out"] = int(len(df))
    report["passed"] = all(c["passed"] for c in report["checks"].values())
    return df.reset_index(drop=True), report


def main() -> None:
    params = load_params()
    ensure_dirs()
    df = pd.read_csv(RAW_DIR / "urls.csv")
    clean, report = validate(df, params["validation"])
    save_json(report, REPORTS_DIR / "data_validation.json")
    for name, c in report["checks"].items():
        log.info("%-26s %s  %s", name, "PASS" if c["passed"] else "FAIL",
                 {k: v for k, v in c.items() if k != "passed"})
    if not report["passed"]:
        log.error("Data validation FAILED - stopping the pipeline.")
        sys.exit(1)
    clean.to_csv(INTERIM_DIR / "urls_clean.csv", index=False)
    log.info("Validation passed: %d -> %d rows", report["rows_in"], report["rows_out"])


if __name__ == "__main__":
    main()
