"""DVC stage `promote` + registry CLI."""

from __future__ import annotations

import argparse
import json

from phishshield.config import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_params
from phishshield.registry import model_registry as reg
from phishshield.training.common import load_split
from phishshield.utils.common import get_logger, save_json

log = get_logger("promote")


def validate_and_promote(candidate_dir, reason: str, eval_urls, eval_y, params: dict) -> dict:
    """Candidate and champion are evaluated on the SAME data before any promotion."""
    version = reg.register(candidate_dir, reason)
    cand = reg.evaluate_version(reg.REGISTRY_DIR / version, eval_urls, eval_y)
    champ_dir = reg.champion_dir()
    champ = reg.evaluate_version(champ_dir, eval_urls, eval_y) if champ_dir else None
    ok, reasons = reg.gate(cand, champ, params["promotion"])
    idx = reg.load_index()
    idx["versions"][version]["metrics"] = {k: v for k, v in cand.items() if isinstance(v, float)}
    reg.save_index(idx)
    reg.promote(version, reason) if ok else reg.reject(version, reasons)
    return {"version": version, "promoted": ok, "reasons": reasons, "candidate": cand, "champion": champ}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    if args.status:
        print(json.dumps(reg.load_index(), indent=2))
        return
    if args.rollback:
        reg.rollback()
        return
    params = load_params()
    test = load_split("test")
    result = validate_and_promote(MODELS_DIR / "student", "pipeline: distilled student", test.urls, test.y, params)
    save_json(result, REPORTS_DIR / "promotion.json")
    log.info("Promotion: %s", {k: result[k] for k in ("version", "promoted", "reasons")})


if __name__ == "__main__":
    main()
