"""Tiered risk-based update policy (core contribution #3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Action(str, Enum):
    NONE = "none"
    RECALIBRATE = "recalibrate"
    PARTIAL_FINETUNE = "partial_finetune"
    FULL_RETRAIN = "full_retrain"


ORDER = [Action.NONE, Action.RECALIBRATE, Action.PARTIAL_FINETUNE, Action.FULL_RETRAIN]


@dataclass
class DriftSignals:
    feature_drift_share: float = 0.0
    embedding_score: float = 0.0
    reference_f1: float | None = None
    current_f1: float | None = None
    current_fnr: float | None = None
    uncertain_share_ref: float | None = None
    uncertain_share_cur: float | None = None


@dataclass
class Decision:
    action: Action
    risk: float
    components: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


def performance_score(s: DriftSignals) -> tuple[float, str]:
    if s.reference_f1 is not None and s.current_f1 is not None:
        rel_drop = max(0.0, (s.reference_f1 - s.current_f1) / max(s.reference_f1, 1e-6))
        return min(rel_drop / 0.15, 1.0), f"relative F1 drop {rel_drop:.1%} (labels available)"
    if s.uncertain_share_ref is not None and s.uncertain_share_cur is not None:
        delta = max(0.0, s.uncertain_share_cur - s.uncertain_share_ref)
        return min(delta / 0.15, 1.0), f"uncertain-prediction share +{delta:.1%} (label-free proxy)"
    return 0.0, "no performance evidence"


def decide(s: DriftSignals, cfg: dict, max_fnr: float | None = None) -> Decision:
    w, tiers = cfg["weights"], cfg["tiers"]
    perf, perf_reason = performance_score(s)
    comps = {"feature": float(s.feature_drift_share), "embedding": float(s.embedding_score), "performance": perf}
    risk = w["feature"] * comps["feature"] + w["embedding"] * comps["embedding"] + w["performance"] * comps["performance"]
    reasons = [f"feature drift share {comps['feature']:.2f}", f"embedding drift score {comps['embedding']:.2f}", perf_reason]

    if risk >= tiers["full_retrain"]:
        action = Action.FULL_RETRAIN
    elif risk >= tiers["partial_finetune"]:
        action = Action.PARTIAL_FINETUNE
    elif risk >= tiers["recalibrate"]:
        action = Action.RECALIBRATE
    else:
        action = Action.NONE

    if max_fnr is not None and s.current_fnr is not None and s.current_fnr > max_fnr:
        if ORDER.index(action) < ORDER.index(Action.PARTIAL_FINETUNE):
            action = Action.PARTIAL_FINETUNE
            reasons.append(f"SAFETY OVERRIDE: FNR {s.current_fnr:.3f} > limit {max_fnr}")
    return Decision(action, float(risk), comps, reasons)
