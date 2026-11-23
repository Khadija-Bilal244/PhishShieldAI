import pytest

from phishshield.config import load_params
from phishshield.drift.risk_policy import Action, DriftSignals, decide

CFG = load_params()["drift"]


@pytest.mark.parametrize("feat,emb,ref,cur,expected", [
    (0.0, 0.0, 0.95, 0.95, Action.NONE),
    (0.5, 0.3, 0.95, 0.95, Action.RECALIBRATE),
    (0.8, 0.7, 0.95, 0.93, Action.PARTIAL_FINETUNE),
    (1.0, 1.0, 0.95, 0.70, Action.FULL_RETRAIN),
])
def test_tiers(feat, emb, ref, cur, expected):
    d = decide(DriftSignals(feat, emb, ref, cur), CFG)
    assert d.action == expected


def test_risk_is_monotonic_in_drift():
    lo = decide(DriftSignals(0.1, 0.1), CFG).risk
    hi = decide(DriftSignals(0.9, 0.9), CFG).risk
    assert hi > lo


def test_safety_override_on_high_fnr():
    d = decide(DriftSignals(0.0, 0.0, 0.95, 0.95, current_fnr=0.4), CFG, max_fnr=0.15)
    assert d.action == Action.PARTIAL_FINETUNE
    assert any("SAFETY" in r for r in d.reasons)


def test_label_free_proxy_used_without_labels():
    d = decide(DriftSignals(0.0, 0.0, uncertain_share_ref=0.05, uncertain_share_cur=0.30), CFG)
    assert d.components["performance"] == 1.0
    assert "label-free" in d.reasons[2]


def test_decision_serialises():
    assert decide(DriftSignals(), CFG).to_dict()["action"] == "none"
