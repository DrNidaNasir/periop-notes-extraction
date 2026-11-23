"""Draw the result figures from the saved metrics and predictions.

Usage:
    python -m periop.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

LABEL = {"map_only": "Current MAP only", "record_only": "Patient record only",
         "wave_features": "Waveform features", "wave_cnn": "Waveform CNN", "fusion": "Waveform + record (fusion)"}
COLOR = {"map_only": "#7f7f7f", "record_only": "#bcbd22", "wave_features": "#1f77b4", "wave_cnn": "#2ca02c", "fusion": "#8c152b"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def waveform_figures(res: Path, fig: Path) -> None:
    m = json.loads((res / "metrics.json").read_text())
    pred = pd.read_csv(res / "predictions_test.csv")
    y = pred["y"].to_numpy()
    names = [n for n in LABEL if "p_" + n in pred.columns]

    # 1. ROC and precision-recall curves
    fg, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for n in names:
        p = pred["p_" + n].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a, ap = m["metrics"][n]["auroc"], m["metrics"][n]["auprc"]
        ax[0].plot(fpr, tpr, color=COLOR[n], lw=2, label=f"{LABEL[n]}: {a['value']:.3f} ({a['ci95'][0]:.3f} to {a['ci95'][1]:.3f})")
        ax[1].plot(rec, prec, color=COLOR[n], lw=2, label=f"{LABEL[n]}: {ap['value']:.3f}")
    ax[0].plot([0, 1], [0, 1], color="#cccccc", lw=1, ls="--")
    ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC, test patients (AUROC, 95% interval)")
    ax[1].axhline(y.mean(), color="#cccccc", lw=1, ls="--")
    ax[1].set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value", title="Precision-recall (AUPRC)")
    ax[0].legend(fontsize=8.5, loc="lower right"); ax[1].legend(fontsize=8.5, loc="upper right")
    fg.tight_layout(); fg.savefig(fig / "waveform_roc_pr.png"); plt.close(fg)

    # 2. calibration
    fg, ax = plt.subplots(figsize=(5.2, 4.8))
    for n in names:
        cal = m["metrics"][n]["calibration"]
        ax.plot([c["mean_predicted"] for c in cal], [c["observed"] for c in cal], "o-", color=COLOR[n], ms=4, label=LABEL[n])
    lim = max(0.05, max(c["observed"] for n in names for c in m["metrics"][n]["calibration"]) * 1.1)
    ax.plot([0, 1], [0, 1], color="#cccccc", ls="--", lw=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate", title="Calibration, test patients", ylim=(0, lim))
    ax.legend(fontsize=8.5)
    fg.tight_layout(); fg.savefig(fig / "waveform_calibration.png"); plt.close(fg)

    # 3. permutation importance of the fusion model
    imp = pd.Series(m["permutation_importance_validation"]).sort_values().tail(15)
    fg, ax = plt.subplots(figsize=(6.4, 5))
    ax.barh(imp.index, imp.values, color="#8c152b")
    ax.set(xlabel="Drop in validation AUROC when the feature is shuffled", title="Fusion model: 15 most important inputs")
    fg.tight_layout(); fg.savefig(fig / "fusion_importance.png"); plt.close(fg)

    # 4. honest task versus grey-zone selection
    fg, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [m["metrics"][n]["auroc"]["value"] for n in names], w, color="#8c152b", label="All samples (this study)")
    ax.bar(x + w / 2, [m["grey_zone_test_auroc"][n] for n in names], w, color="#c9c9c9", label="Future MAP 65 to 75 mmHg removed")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[n].replace(" (fusion)", "\n(fusion)").replace(" only", "\nonly") for n in names], fontsize=8.5)
    ax.set(ylabel="Test AUROC", ylim=(0.5, 1.0), title="Effect of removing the grey zone from the test set")
    ax.legend(fontsize=9, loc="lower right")
    fg.tight_layout(); fg.savefig(fig / "grey_zone_effect.png"); plt.close(fg)


def example_trace(data: Path, res: Path, fig: Path) -> None:
    """One test patient: MAP over time with the fusion model's predicted risk."""
    pred = pd.read_csv(res / "predictions_test.csv")
    if "p_fusion" not in pred.columns:
        return
    ev = pred.groupby("caseid")["y"].sum()
    case = int(ev[ev > 0].sort_values().index[len(ev[ev > 0]) // 2])    # a patient with a typical number of events
    g = pred[pred["caseid"] == case].sort_values("t")
    fg, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(g["t"] / 60, g["map_now"], color="#333333", lw=1.2)
    ax[0].axhline(65, color="#8c152b", ls="--", lw=1)
    ax[0].set(ylabel="MAP (mmHg)", title=f"VitalDB case {case}: MAP at prediction time and predicted risk 5 minutes ahead")
    ax[1].fill_between(g["t"] / 60, g["p_fusion"], color="#8c152b", alpha=0.35, step="mid")
    hit = g[g["y"] == 1]
    ax[1].scatter(hit["t"] / 60, np.full(len(hit), 1.02), marker="v", color="#333333", s=14, label="hypotension 5 min later")
    ax[1].set(xlabel="Time since start of recording (min)", ylabel="Predicted risk", ylim=(0, 1.1))
    ax[1].legend(fontsize=8.5, loc="upper left")
    fg.tight_layout(); fg.savefig(fig / "example_case.png"); plt.close(fg)


def notes_figure(res: Path, fig: Path) -> None:
    s = json.loads((res / "summary.json").read_text())
    title = {"ebl": "Estimated blood loss", "anes": "Type of anaesthesia"}
    methods = [("rules", "Rules only", "#7f7f7f"), ("llm_snippet", "LLM on snippets", "#8c152b"), ("llm_full", "LLM on full note", "#1f77b4")]
    fg, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(s)); w = 0.26
    for k, (key, lab, col) in enumerate(methods):
        acc = [s[t][key]["accuracy"] for t in s]
        lo = [s[t][key]["accuracy"] - s[t][key]["ci95"][0] for t in s]
        hi = [s[t][key]["ci95"][1] - s[t][key]["accuracy"] for t in s]
        ax[0].bar(x + (k - 1) * w, acc, w, yerr=[lo, hi], capsize=3, color=col, label=lab)
        ax[1].bar(x + (k - 1) * w, [max(s[t][key]["seconds_per_note"], 1e-4) for t in s], w, color=col, label=lab)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([title[t] for t in s])
    ax[0].set(ylabel="Exact agreement with reference", ylim=(0, 1.05), title="Accuracy on 60 hand-checked notes (95% interval)")
    ax[1].set(ylabel="Seconds per note (log scale)", yscale="log", title="Processing time on a laptop CPU")
    ax[0].legend(fontsize=9, loc="lower right")
    fg.tight_layout(); fg.savefig(fig / "notes_accuracy_time.png"); plt.close(fg)


if __name__ == "__main__":
    fig = Path("results/figures"); fig.mkdir(parents=True, exist_ok=True)
    if Path("results/waveform/metrics.json").exists():
        waveform_figures(Path("results/waveform"), fig)
        example_trace(Path("data/processed"), Path("results/waveform"), fig)
    if Path("results/notes/summary.json").exists():
        notes_figure(Path("results/notes"), fig)
    print("figures written to", fig)
