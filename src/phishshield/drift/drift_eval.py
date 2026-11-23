"""DVC stage `drift_eval`: run the drift monitor over every stream batch."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from phishshield.config import REPORTS_DIR, STREAM_DIR, ensure_dirs, load_params  # noqa: E402
from phishshield.drift.monitor import DRIFT_DIR, analyse  # noqa: E402
from phishshield.utils import tracking  # noqa: E402
from phishshield.utils.common import get_logger, load_json, save_json  # noqa: E402

log = get_logger("drift_eval")


def main() -> None:
    params = load_params()
    ensure_dirs()
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    shares = {b["batch"]: b["new_style_share"] for b in load_json(REPORTS_DIR / "stream_summary.json")["batches"]}
    files = sorted(STREAM_DIR.glob("batch_*.csv"))
    rows = []
    for k, f in enumerate(files):
        res = analyse(pd.read_csv(f), params, name=f.stem, write_html=(k == len(files) - 1))
        d, fd, ed, perf = res["decision"], res["feature_drift"], res["embedding_drift"], res["performance"] or {}
        rows.append({"batch": k, "new_style_share": shares.get(k), "feature_drift_share": fd["drift_share"],
                     "mmd_p_value": ed["p_value"], "domain_auc": ed["domain_auc"], "embedding_score": ed["score"],
                     "embedding_drift_detected": ed["drift_detected"], "f1": perf.get("f1"), "fnr": perf.get("fnr"),
                     "risk": d["risk"], "action": d["action"], "embedding_source": res["embedding_source"]})
        log.info("batch %d (%.0f%% new) feat=%.2f embAUC=%.3f risk=%.3f -> %s", k, 100 * (shares.get(k) or 0),
                 fd["drift_share"], ed["domain_auc"], d["risk"], d["action"])
    df = pd.DataFrame(rows)
    save_json(rows, REPORTS_DIR / "drift_detection.json")
    lines = ["| Batch | New-style share | Feature drift share | Embedding domain AUC | MMD p-value | F1 | FNR | Risk | Policy action |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['batch']} | {r['new_style_share']:.0%} | {r['feature_drift_share']:.2f} | {r['domain_auc']:.3f} | "
                     f"{r['mmd_p_value']:.3f} | {r['f1']:.4f} | {r['fnr']:.4f} | {r['risk']:.3f} | {r['action']} |")
    (REPORTS_DIR / "drift_detection.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df.new_style_share, df.feature_drift_share, marker="o", label="feature drift share (PSI/KS)")
    ax[0].plot(df.new_style_share, (df.domain_auc - 0.5) * 2, marker="s", label="embedding drift (2·(AUC−0.5))")
    ax[0].plot(df.new_style_share, df.risk, marker="^", label="policy risk score")
    ax[0].set(xlabel="share of new attack styles", ylabel="signal (0-1)", title="Drift signals vs true drift")
    ax[0].legend(fontsize=8)
    ax[1].plot(df.new_style_share, df.f1, marker="o", color="#0f766e")
    ax[1].set(xlabel="share of new attack styles", ylabel="F1 of deployed model", title="Model degradation under drift")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "drift_detection.png", dpi=130)
    plt.close()

    with tracking.run("drift-evaluation", params["mlflow"]["experiment"], params["drift"], {"stage": "drift"}):
        for r in rows:
            tracking.log_metrics({k: r[k] for k in ("feature_drift_share", "domain_auc", "embedding_score", "risk", "f1", "fnr")
                                  if r[k] is not None}, step=r["batch"])
        tracking.log_artifact(str(REPORTS_DIR / "figures" / "drift_detection.png"), "drift")
        tracking.log_artifact(str(REPORTS_DIR / "drift_detection.md"), "drift")


if __name__ == "__main__":
    main()
