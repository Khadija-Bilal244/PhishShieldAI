"""DVC stage `explain`: explainability report."""

from __future__ import annotations

import argparse
import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from phishshield.config import MODELS_DIR, REPORTS_DIR, ensure_dirs, load_params  # noqa: E402
from phishshield.features import char_tokenizer  # noqa: E402
from phishshield.features.lexical import FEATURE_NAMES  # noqa: E402
from phishshield.models.hybrid import load_hybrid  # noqa: E402
from phishshield.training.common import load_split  # noqa: E402
from phishshield.utils import tracking  # noqa: E402
from phishshield.utils.common import get_logger, save_json  # noqa: E402

log = get_logger("explain")
_ID2CHAR = {v: k for k, v in char_tokenizer.CHAR2ID.items()}


def integrated_gradients(urls: list[str], feats: np.ndarray, steps: int = 32) -> list[dict]:
    from captum.attr import LayerIntegratedGradients

    model, meta = load_hybrid(MODELS_DIR / "hybrid")
    model.eval()
    enc = model.encoder_wrapper

    def forward(input_ids, attention_mask, features):
        return model(input_ids, attention_mask, features)

    lig = LayerIntegratedGradients(forward, model.encoder.embeddings.word_embeddings)
    out = []
    for url, f in zip(urls, feats):
        tok = enc.tokenize([url], torch.device("cpu"))
        ids, mask = tok["input_ids"], tok["attention_mask"]
        baseline = torch.zeros_like(ids)
        if not meta["debug_tiny"]:
            baseline[:, 0] = ids[:, 0]
            for r, c in (ids == enc.tokenizer.sep_token_id).nonzero().tolist():
                baseline[r, c] = ids[r, c]
        ft = torch.as_tensor(f[None, :], dtype=torch.float32)
        attr = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask, ft), n_steps=steps)
        scores = attr.sum(-1).squeeze(0).detach().numpy()
        n = int(mask.sum())
        tokens = ([_ID2CHAR.get(int(i), "?") for i in ids[0, :n]] if meta["debug_tiny"]
                  else enc.tokenizer.convert_ids_to_tokens(ids[0, :n]))
        with torch.no_grad():
            prob = float(torch.sigmoid(forward(ids, mask, ft) / meta["temperature"]))
        out.append({"url": url, "phishing_probability": prob, "tokens": tokens,
                    "attributions": scores[:n].round(5).tolist()})
    return out


def to_html(items: list[dict]) -> str:
    rows = []
    for it in items:
        a = np.array(it["attributions"])
        m = np.abs(a).max() or 1.0
        spans = "".join(
            f'<span style="background:{"rgba(220,38,38," if s > 0 else "rgba(37,99,235,"}{abs(s) / m:.2f});'
            f'padding:1px 0" title="{s:.4f}">{html.escape(t)}</span>' for t, s in zip(it["tokens"], a))
        rows.append(f"<tr><td>{it['phishing_probability']:.3f}</td><td style='font-family:monospace'>{spans}</td></tr>")
    return ("<!doctype html><meta charset='utf-8'><title>PhishShield - Integrated Gradients</title>"
            "<style>body{font-family:system-ui;margin:24px}td{padding:6px 10px;border-bottom:1px solid #ddd}</style>"
            "<h2>Integrated Gradients - hybrid DistilBERT teacher</h2>"
            "<p>Red = token pushes toward <b>phishing</b>, blue = toward <b>legitimate</b>.</p>"
            "<table><tr><th>P(phish)</th><th>URL tokens</th></tr>" + "".join(rows) + "</table>")


def shap_summary(params: dict) -> dict:
    import shap

    from phishshield.api.predictor import PhishPredictor

    pred = PhishPredictor(MODELS_DIR / "student")
    test = load_split("test")
    rng = np.random.default_rng(params["seed"])
    idx = rng.choice(len(test), min(params["explain"]["shap_samples"], len(test)), replace=False)
    urls, X = [test.urls[i] for i in idx], test.Xs[idx]
    ids = char_tokenizer.encode_batch(urls, pred.max_len)
    values = []
    for i in range(len(urls)):
        def f(feats, i=i):
            rep = np.repeat(ids[i:i + 1], len(feats), axis=0)
            logit = pred.session.run(["logit"], {"char_ids": rep, "features": feats.astype(np.float32)})[0]
            return 1 / (1 + np.exp(-logit / pred.temperature))
        ex = shap.KernelExplainer(f, pred.background)
        values.append(np.asarray(ex.shap_values(X[i:i + 1], nsamples=150, silent=True)).reshape(-1))
    values = np.vstack(values)
    raw = test.X[idx]
    plt.figure()
    shap.summary_plot(values, raw, feature_names=FEATURE_NAMES, show=False, max_display=15)
    plt.title("SHAP - engineered-feature contributions (student)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    imp = dict(sorted(zip(FEATURE_NAMES, np.abs(values).mean(0).round(5).tolist()), key=lambda kv: -kv[1]))
    save_json(imp, REPORTS_DIR / "shap_importance.json")
    return imp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", help="explain specific URL(s) instead of test examples")
    args = ap.parse_args()
    params = load_params()
    ensure_dirs()
    if args.url:
        import joblib

        from phishshield.features.lexical import to_matrix

        urls = args.url
        feats = joblib.load(MODELS_DIR / "scaler.joblib").transform(to_matrix(urls)).astype(np.float32)
    else:
        test = load_split("test")
        rng = np.random.default_rng(0)
        n = params["explain"]["n_examples"]
        idx = np.r_[rng.choice(np.where(test.y == 1)[0], n // 2, replace=False),
                    rng.choice(np.where(test.y == 0)[0], n - n // 2, replace=False)]
        urls, feats = [test.urls[i] for i in idx], test.Xs[idx]

    with tracking.run("explainability", params["mlflow"]["experiment"], params["explain"], {"stage": "explain"}):
        items = integrated_gradients(urls, feats)
        save_json(items, REPORTS_DIR / "explanations.json")
        (REPORTS_DIR / "explanations.html").write_text(to_html(items), encoding="utf-8")
        tracking.log_artifact(str(REPORTS_DIR / "explanations.html"), "explain")
        if not args.url:
            imp = shap_summary(params)
            tracking.log_artifact(str(REPORTS_DIR / "figures" / "shap_summary.png"), "explain")
            tracking.log_artifact(str(REPORTS_DIR / "shap_importance.json"), "explain")
            log.info("Top SHAP features: %s", list(imp)[:5])
    log.info("Integrated Gradients for %d URLs -> reports/explanations.html", len(items))


if __name__ == "__main__":
    main()
