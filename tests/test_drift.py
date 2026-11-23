import numpy as np
import pandas as pd

from phishshield.drift.embedding_drift import domain_classifier_auc, embedding_drift, mmd_test
from phishshield.drift.feature_drift import feature_drift, psi

rng = np.random.default_rng(0)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.5


def test_psi_binary_feature():
    assert psi(np.r_[np.zeros(900), np.ones(100)], np.r_[np.zeros(500), np.ones(500)]) > 0.2


def test_feature_drift_share():
    ref = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(3, 1, 1000)})
    out = feature_drift(ref, cur)
    assert out["drift_share"] == 0.5 and out["per_feature"]["b"]["drifted"]


def test_mmd_detects_shift_but_not_noise():
    x = rng.normal(0, 1, (300, 8))
    assert mmd_test(x, rng.normal(0, 1, (300, 8)), n_perm=100)["p_value"] > 0.01
    assert mmd_test(x, rng.normal(0.8, 1, (300, 8)), n_perm=100)["p_value"] < 0.05


def test_domain_classifier_auc():
    x = rng.normal(0, 1, (400, 8))
    assert domain_classifier_auc(x, rng.normal(0, 1, (400, 8))) < 0.62
    assert domain_classifier_auc(x, rng.normal(1.5, 1, (400, 8))) > 0.9


def test_embedding_drift_score_range():
    out = embedding_drift(rng.normal(0, 1, (200, 4)), rng.normal(2, 1, (200, 4)), n_perm=50, max_n=200)
    assert 0 <= out["score"] <= 1 and out["drift_detected"]
