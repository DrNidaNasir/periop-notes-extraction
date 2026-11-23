"""Evaluation helpers shared by all models.

All confidence intervals are obtained by resampling PATIENTS (cases), not samples.
Consecutive 30-second samples from one operation are strongly correlated, so a
sample-level bootstrap would give intervals that are far too narrow.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability cut-off that maximises sensitivity + specificity - 1 (chosen on validation data)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
    }


def case_bootstrap(y: np.ndarray, p: np.ndarray, cases: np.ndarray, threshold: float,
                   n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimates with 95% intervals from a patient-level bootstrap."""
    rng = np.random.default_rng(seed)
    ids = np.unique(cases)
    index = {c: np.flatnonzero(cases == c) for c in ids}
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = np.concatenate([index[c] for c in rng.choice(ids, len(ids), replace=True)])
        if y[pick].min() == y[pick].max():      # a resample without both classes cannot be scored
            continue
        for k, v in point_metrics(y[pick], p[pick], threshold).items():
            draws.setdefault(k, []).append(v)
    est = point_metrics(y, p, threshold)
    return {k: {"value": est[k], "ci95": [float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5))]}
            for k in est}


def paired_auroc_difference(y, p_a, p_b, cases, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b) with a 95% interval, resampling the same patients for both models."""
    rng = np.random.default_rng(seed)
    ids = np.unique(cases)
    index = {c: np.flatnonzero(cases == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([index[c] for c in rng.choice(ids, len(ids), replace=True)])
        if y[pick].min() == y[pick].max():
            continue
        diffs.append(roc_auc_score(y[pick], p_a[pick]) - roc_auc_score(y[pick], p_b[pick]))
    return {"value": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Observed event rate in equal-size bins of predicted probability."""
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, n_bins):
        out.append({"mean_predicted": float(p[chunk].mean()), "observed": float(y[chunk].mean()), "n": int(len(chunk))})
    return out
