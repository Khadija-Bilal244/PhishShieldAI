"""Collect a REAL, time-stamped batch of URLs after training (proposal 5.1: temporal evaluation)."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import pandas as pd

from phishshield.config import LIVE_DIR, load_params
from phishshield.utils.common import get_logger, save_json

log = get_logger("collect_live")
UA = {"User-Agent": "phishshield-ai-research (university MLOps project)"}


def _get(url: str) -> bytes | None:
    try:
        import requests

        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning("Download failed for %s: %s", url, e)
        return None


def openphish(cfg: dict) -> list[str]:
    manual = LIVE_DIR / "manual" / "openphish.txt"
    text = manual.read_text() if manual.exists() else (_get(cfg["openphish_url"]) or b"").decode(errors="ignore")
    urls = [u.strip() for u in text.splitlines() if u.strip().startswith("http")]
    return urls[: cfg["max_phishing"]]


def tranco(cfg: dict) -> list[str]:
    manual = LIVE_DIR / "manual" / "tranco.csv"
    if manual.exists():
        df = pd.read_csv(manual, header=None, names=["rank", "domain"], nrows=cfg["tranco_top_n"])
    else:
        blob = _get(cfg["tranco_url"])
        if blob is None:
            return []
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=["rank", "domain"],
                             nrows=cfg["tranco_top_n"])
    return ("https://www." + df["domain"].astype(str)).tolist()


def main() -> None:
    cfg = load_params()["live"]
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    phish, legit = openphish(cfg), tranco(cfg)
    if not phish or not legit:
        raise SystemExit("Live collection failed (see warnings). Use the manual files described in the docstring.")
    df = pd.DataFrame({"url": phish + legit, "label": [1] * len(phish) + [0] * len(legit),
                       "source": ["openphish"] * len(phish) + ["tranco"] * len(legit),
                       "collected_at": now.isoformat(timespec="seconds")})
    df = df.drop_duplicates("url").sample(frac=1.0, random_state=0)
    out = LIVE_DIR / f"live_{now:%Y%m%d}.csv"
    df.to_csv(out, index=False)
    save_json({"file": out.name, "rows": len(df), "phishing": len(phish), "legit": len(legit),
               "collected_at": now.isoformat()}, LIVE_DIR / "latest.json")
    log.info("Collected %d phishing + %d legitimate URLs -> %s", len(phish), len(legit), out)


if __name__ == "__main__":
    main()
