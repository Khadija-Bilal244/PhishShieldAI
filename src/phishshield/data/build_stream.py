"""DVC stage `build_stream`: a reproducible stream of "production" batches with increasing drift."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from phishshield.config import RAW_DIR, REPORTS_DIR, STREAM_DIR, ensure_dirs, load_params, set_seed
from phishshield.data import make_sample
from phishshield.data.ingest import read_phiusiil, to_schema
from phishshield.utils.common import get_logger, save_json

log = get_logger("build_stream")
NEW_PERIOD = 5


def unseen_pool(params: dict) -> pd.DataFrame:
    raw, _ = read_phiusiil(params["data"]["url"])
    df = to_schema(raw).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    seen = set(pd.read_csv(RAW_DIR / "urls.csv")["url"])
    df = df[~df["url"].isin(seen)].drop_duplicates("url")
    n = min(params["stream"]["pool_size"], len(df))
    return df.sample(n=n, random_state=params["seed"]).reset_index(drop=True)


def new_style_rows(n: int, phish_ratio: float, seed: int) -> pd.DataFrame:
    set_seed(seed)
    rows = []
    for _ in range(n):
        if random.random() < phish_ratio:
            rows.append((make_sample.phish_url(NEW_PERIOD), 1))
        else:
            rows.append((make_sample.legit_url(NEW_PERIOD), 0))
    return pd.DataFrame(rows, columns=["url", "label"]).assign(source="new_style")


def main() -> None:
    params = load_params()
    ensure_dirs()
    STREAM_DIR.mkdir(parents=True, exist_ok=True)
    s = params["stream"]
    if params["data"]["source"] == "sample":
        pool = make_sample.generate(s["n_batches"] * s["batch_size"] * 2, 3, params["seed"] + 7)
        pool = pool[["url", "label"]].assign(source="synthetic")
    else:
        pool = unseen_pool(params)
    phish_ratio = float(pool["label"].mean())
    for f in STREAM_DIR.glob("batch_*.csv"):
        f.unlink()

    summary, offset = [], 0
    for i in range(s["n_batches"]):
        share = s["max_new_style_share"] * i / max(s["n_batches"] - 1, 1)
        n_new = int(round(share * s["batch_size"]))
        base = pool.iloc[offset: offset + s["batch_size"] - n_new]
        offset += len(base)
        batch = pd.concat([base, new_style_rows(n_new, phish_ratio, params["seed"] + 100 + i)], ignore_index=True)
        batch = batch.sample(frac=1.0, random_state=params["seed"] + i).reset_index(drop=True)
        batch.insert(0, "batch", i)
        batch.to_csv(STREAM_DIR / f"batch_{i:02d}.csv", index=False)
        summary.append({"batch": i, "rows": len(batch), "new_style_share": round(share, 3),
                        "phishing_ratio": round(float(batch["label"].mean()), 4)})
        log.info("batch %02d: %d URLs, %.0f%% new styles", i, len(batch), 100 * share)
    save_json({"batches": summary, "pool_size": len(pool), "simulated": True}, REPORTS_DIR / "stream_summary.json")
    np.random.seed(params["seed"])


if __name__ == "__main__":
    main()
