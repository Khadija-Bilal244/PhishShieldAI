"""Model tests with the tiny offline transformer (no downloads, CPU, seconds)."""

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from phishshield.config import load_params  # noqa: E402
from phishshield.evaluation.metrics import best_threshold, classification_metrics  # noqa: E402
from phishshield.features.lexical import FEATURE_NAMES, to_matrix  # noqa: E402
from phishshield.models.baselines import build_baselines  # noqa: E402
from phishshield.models.hybrid import load_hybrid, predict_logits, save_hybrid  # noqa: E402
from phishshield.training.common import fit_temperature  # noqa: E402
from phishshield.training.hybrid_trainer import build_hybrid  # noqa: E402

P = load_params()
CFG = {**P["hybrid"], "debug_tiny": True}
URLS = ["http://192.168.0.1/paypal/login.php", "https://github.com/microsoft/vscode",
        "http://paypa1-verify.xyz/secure", "https://www.wikipedia.org/wiki/Python"] * 8
Y = np.array([1, 0, 1, 0] * 8)
X = to_matrix(URLS)
XS = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)


@pytest.mark.parametrize("mode", ["fusion", "text_only", "features_only"])
def test_hybrid_forward_shapes(mode):
    m = build_hybrid(CFG, len(FEATURE_NAMES), mode)
    logits, emb = predict_logits(m, URLS[:4], XS[:4], return_embeddings=True)
    assert logits.shape == (4,) and emb.shape[0] == 4


def test_hybrid_save_load_roundtrip(tmp_path):
    m = build_hybrid(CFG, len(FEATURE_NAMES))
    meta = {"pretrained_name": CFG["pretrained_name"], "debug_tiny": True, "max_len": CFG["max_len"],
            "n_features": len(FEATURE_NAMES), "fusion_dim": CFG["fusion_dim"], "dropout": CFG["dropout"],
            "mode": "fusion", "temperature": 1.0, "threshold": 0.5, "n_params": 1}
    save_hybrid(m, tmp_path, meta)
    m2, _ = load_hybrid(tmp_path)
    assert np.allclose(predict_logits(m, URLS[:4], XS[:4]), predict_logits(m2, URLS[:4], XS[:4]), atol=1e-5)


def test_baselines_fit_and_predict():
    for name, model in build_baselines(P["baselines"], 0).items():
        model.fit(X, Y)
        assert model.predict_proba(X).shape == (len(X), 2), name


def test_metrics_and_fnr():
    m = classification_metrics([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.6])
    assert m["fnr"] == 0.5 and m["fpr"] == 0.5 and m["tp"] == 1


def test_best_threshold_respects_fnr_limit():
    y = np.array([1] * 50 + [0] * 50)
    p = np.r_[np.linspace(0.3, 1, 50), np.linspace(0, 0.7, 50)]
    t = best_threshold(y, p, max_fnr=0.0)
    assert classification_metrics(y, p, t)["fnr"] == 0.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    logits = np.where(y == 1, 1.0, -1.0) * 8 + rng.normal(0, 6, 500)
    assert fit_temperature(logits, y) > 1.0
