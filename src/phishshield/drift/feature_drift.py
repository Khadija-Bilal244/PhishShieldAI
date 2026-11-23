"""Feature-level drift: PSI + Kolmogorov-Smirnov per feature, plus an Evidently HTML report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from phishshield.utils.common import get_logger

log = get_logger("feature_drift")


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.2 moderate, >0.2 significant shift."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        vals = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == v).mean() for v in vals])
        c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, psi_threshold: float = 0.2,
                  ks_pvalue: float = 0.01) -> dict:
    per_feature = {}
    for col in ref.columns:
        p = psi(ref[col].to_numpy(), cur[col].to_numpy())
        ks = ks_2samp(ref[col], cur[col])
        drifted = bool(p > psi_threshold or (ks.pvalue < ks_pvalue and ks.statistic > 0.1))
        per_feature[col] = {"psi": p, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue), "drifted": drifted}
    share = float(np.mean([v["drifted"] for v in per_feature.values()]))
    top = sorted(per_feature.items(), key=lambda kv: kv[1]["psi"], reverse=True)[:5]
    return {"drift_share": share, "n_drifted": int(sum(v["drifted"] for v in per_feature.values())),
            "top_drifted": [{"feature": k, **v} for k, v in top], "per_feature": per_feature}


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, out_html: Path) -> bool:
    """Write an Evidently data-drift HTML dashboard. Supports Evidently >=0.7 and the legacy API."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
        snapshot.save_html(str(out_html))
        return True
    except ImportError:
        pass
    except Exception as e:
        log.warning("Evidently (new API) failed: %s", e)
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=ref, current_data=cur)
        rep.save_html(str(out_html))
        return True
    except Exception as e:
        log.warning("Evidently report skipped: %s", e)
        return False
