"""DVC stage `ingest`: download the dataset from its source and write data/raw/urls.csv.

Source: PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024), UCI ML Repository #967
        https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
        235,795 URLs (134,850 legitimate / 100,945 phishing).
        NOTE: in PhiUSIIL label 1 = legitimate, 0 = phishing. We flip it so 1 = phishing.

The dataset is a static, published snapshot, so re-running this stage from scratch always
produces the same data (the SHA-256 of the download is recorded in reports/ingest.json).

If the download is blocked, download the zip in a browser and put it at
data/raw/phiusiil.zip (or the extracted CSV at data/raw/PhiUSIIL_Phishing_URL_Dataset.csv).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phishshield.config import RAW_DIR, REPORTS_DIR, ensure_dirs, load_params, set_seed
from phishshield.utils.common import get_logger, save_json

log = get_logger("ingest")
ZIP_PATH = RAW_DIR / "phiusiil.zip"
CSV_PATH = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"


def download(url: str, dest: Path) -> None:
    import requests

    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=300, headers={"User-Agent": "phishshield-ai-ingest"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_phiusiil(url: str) -> tuple[pd.DataFrame, str]:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, low_memory=False), sha256(CSV_PATH)
    if not ZIP_PATH.exists():
        download(url, ZIP_PATH)
    digest = sha256(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
    return df, digest


def to_schema(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in raw.columns}
    if "url" not in cols or "label" not in cols:
        raise ValueError(f"Unexpected PhiUSIIL columns: {list(raw.columns)[:10]} ...")
    return pd.DataFrame({
        "url": raw[cols["url"]].astype(str).str.strip(),
        "label": 1 - pd.to_numeric(raw[cols["label"]], errors="coerce"),   # 1 = phishing
        "source": "phiusiil",
    })


def stratified_subsample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n or len(df) <= n:
        return df
    frac = n / len(df)
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=seed))
              .sample(frac=1.0, random_state=seed)
              .reset_index(drop=True))


def main() -> None:
    params = load_params()
    ensure_dirs()
    set_seed(params["seed"])
    d = params["data"]
    if d["source"] == "sample":
        from phishshield.data.make_sample import generate

        df = generate(d["sample_size"], d["n_periods"], params["seed"])[["url", "label", "source"]]
        digest = "synthetic"
    else:
        raw, digest = read_phiusiil(d["url"])
        log.info("Downloaded %d rows, SHA-256 %s", len(raw), digest[:16])
        df = to_schema(raw)
        df = stratified_subsample(df.dropna(subset=["label"]), d["max_rows"], params["seed"])
        df["label"] = df["label"].astype(int)
    out = RAW_DIR / "urls.csv"
    df.to_csv(out, index=False)
    save_json({"source": d["source"], "source_url": d.get("url"), "sha256": digest, "rows": len(df),
               "phishing_ratio": round(float(df["label"].mean()), 4),
               "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
              REPORTS_DIR / "ingest.json")
    log.info("Wrote %d URLs (%.1f%% phishing) -> %s", len(df), 100 * df["label"].mean(), out)


if __name__ == "__main__":
    main()
