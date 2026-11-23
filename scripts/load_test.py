"""Tiny load generator - produces traffic for the Grafana dashboard demo."""
import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phishshield.data.make_sample import legit_url, phish_url  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--period", type=int, default=5, help="0 = old attack styles, 5 = newest (drifted)")
args = ap.parse_args()

t0, errors = time.time(), 0
for i in range(args.n):
    u = phish_url(args.period) if random.random() < 0.45 else legit_url(args.period)
    try:
        requests.post(f"{args.url}/predict", json={"url": u}, timeout=5).raise_for_status()
    except Exception:
        errors += 1
    if i % 50 == 0:
        print(f"{i}/{args.n} sent")
print(f"done: {args.n} requests in {time.time() - t0:.1f}s, errors={errors}")
