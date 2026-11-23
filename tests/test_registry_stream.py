import pandas as pd

from phishshield.config import load_params
from phishshield.data.build_stream import new_style_rows
from phishshield.registry.model_registry import gate

P = load_params()


def test_promotion_gate_rules():
    cfg = P["promotion"]
    champ = {"f1": 0.90, "recall": 0.90, "fnr": 0.10}
    assert gate({"f1": 0.92, "recall": 0.91, "fnr": 0.09}, champ, cfg)[0]
    ok, reasons = gate({"f1": 0.88, "recall": 0.85, "fnr": 0.15}, champ, cfg)
    assert not ok and len(reasons) == 2
    assert not gate({"f1": 0.99, "recall": 0.80, "fnr": 0.20}, None, cfg)[0]


def test_new_style_rows_are_labelled_and_reproducible():
    a, b = new_style_rows(200, 0.4, seed=1), new_style_rows(200, 0.4, seed=1)
    assert a.equals(b)
    assert set(a["label"]) <= {0, 1} and 0.25 < a["label"].mean() < 0.55
    assert (a["source"] == "new_style").all()


def test_stream_params_are_consistent():
    s = P["stream"]
    assert 0 < s["max_new_style_share"] <= 1 and s["n_batches"] >= 2
    t = P["drift"]["tiers"]
    assert t["recalibrate"] < t["partial_finetune"] < t["full_retrain"]
    assert abs(sum(P["drift"]["weights"].values()) - 1.0) < 1e-9


def test_student_partial_freeze_keeps_convs_trainable():
    import pytest

    torch = pytest.importorskip("torch")  # noqa: F841
    from phishshield.models.student import build_student

    m = build_student(P["student"], 31)
    m.freeze_for_partial()
    assert not any(p.requires_grad for p in m.embed.parameters())
    assert all(p.requires_grad for p in m.convs.parameters())
    m.unfreeze_all()
    assert all(p.requires_grad for p in m.parameters())


def test_drift_monitor_handles_unlabelled_batches():
    from phishshield.drift.risk_policy import DriftSignals, decide

    d = decide(DriftSignals(0.5, 0.5, uncertain_share_ref=0.05, uncertain_share_cur=0.25), P["drift"])
    assert d.action.value in {"recalibrate", "partial_finetune", "full_retrain"}
    assert isinstance(pd.Series([d.risk]).iloc[0], float)


def test_bounded_calibration_never_collapses_to_0_or_1():
    import numpy as np

    from phishshield.training.calibration import fit_temperature_bounded, pick_threshold

    y = np.array([0] * 50 + [1] * 50)
    logits = np.r_[np.full(50, -3.0), np.full(50, 3.0)]
    t = fit_temperature_bounded(logits, y)
    assert t >= 0.5
    prob = 1 / (1 + np.exp(-logits / t))
    assert abs(pick_threshold(y, prob, 0.15) - 0.5) < 0.05


def test_permutation_shap_is_exact_for_linear_models():
    import numpy as np

    from phishshield.api.predictor import permutation_shap

    rng = np.random.default_rng(0)
    w, x, bg = rng.normal(size=5), rng.normal(size=5), rng.normal(size=(1, 5))
    phi = permutation_shap(lambda F: F @ w, x, bg, n_perm=16)
    assert np.allclose(phi, w * (x - bg[0]), atol=1e-6)
