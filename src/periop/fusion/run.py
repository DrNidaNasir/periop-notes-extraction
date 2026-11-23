"""Train and compare the hypotension prediction models.

Models, from simplest to richest
--------------------------------
map_only        logistic regression on the current MAP alone. This is the reference
                every other model must beat: a falling blood pressure predicts a low
                blood pressure.
record_only     gradient boosting on the patient record alone (no waveform).
wave_features   gradient boosting on hand-crafted waveform and MAP-trend features.
wave_cnn        1-D CNN on the raw 30-s waveform.
fusion          gradient boosting on waveform features + patient record
                (numeric, categorical and free-text components).

Protocol
--------
* Patients are split 60 / 20 / 20 into training, validation and test sets. All
  samples of one patient stay in one set.
* Hyper-parameters, the number of boosting rounds, the CNN epoch and the decision
  threshold are chosen on the validation set. The test set is used once.
* 95% intervals come from a patient-level bootstrap.

Usage:
    python -m periop.fusion.run --data data/processed --out results/waveform
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..waveform import cnn as cnn_mod
from ..waveform.evaluate import calibration_table, case_bootstrap, paired_auroc_difference, youden_threshold
from ..waveform.features import build_feature_table
from .clinical import ClinicalEncoder

SEED = 0


def split_cases(case_ids: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(case_ids))
    n_tr, n_va = int(0.6 * len(ids)), int(0.2 * len(ids))
    return {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}


def fit_gbm(x_tr, y_tr, x_va, y_va, max_iter: int = 400):
    """Gradient boosting; the number of rounds is the one with the best validation AUROC.

    The built-in early stopping of scikit-learn holds out random samples, which would
    mix samples of the same patient between fitting and stopping. Staged predictions on
    the patient-level validation set avoid that.
    """
    model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=50, l2_regularization=1.0,
                                           class_weight="balanced", early_stopping=False, random_state=SEED)
    model.fit(x_tr, y_tr)
    aucs = [roc_auc_score(y_va, p[:, 1]) for p in model.staged_predict_proba(x_va)]
    best = int(np.argmax(aucs)) + 1

    def predict(x):
        for k, p in enumerate(model.staged_predict_proba(x), 1):
            if k == best:
                return p[:, 1]
    return model, best, predict


def main(data_dir: Path, cases_csv: Path, out_dir: Path, skip_cnn: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data_dir / "samples.csv")
    waves = np.load(data_dir / "waveforms.npy", mmap_mode="r")
    hist = np.load(data_dir / "map_history.npy")

    feat_file = data_dir / "features.parquet"
    if feat_file.exists():
        feats = pd.read_parquet(feat_file)
    else:
        print("computing waveform features ...", flush=True)
        feats = build_feature_table(waves, hist)
        feats.to_parquet(feat_file)
    feats["elapsed_min"] = meta["t"].to_numpy() / 60.0

    # primary cohort: not hypotensive at prediction time (the model must predict the onset)
    keep = (meta["hypo_now"] == 0).to_numpy()
    meta_p, feats_p = meta[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)
    waves_p = np.asarray(waves[np.flatnonzero(keep)])

    split = split_cases(meta_p["caseid"].to_numpy())
    part = {k: meta_p["caseid"].isin(v).to_numpy() for k, v in split.items()}
    y = meta_p["y"].to_numpy()
    cases = meta_p["caseid"].to_numpy()

    # patient-record features, encoder fitted on training patients only
    table = pd.read_csv(cases_csv)
    enc = ClinicalEncoder().fit(table[table["caseid"].isin(split["train"])])
    record = enc.transform(table[table["caseid"].isin(np.unique(cases))])
    rec_p = record.loc[cases].reset_index(drop=True)

    wave_cols = list(feats_p.columns)
    rec_cols = list(rec_p.columns)
    full = pd.concat([feats_p, rec_p], axis=1)

    cohort = {k: {"patients": int(len(v)), "samples": int(part[k].sum()), "events": int(y[part[k]].sum()),
                  "event_rate": float(y[part[k]].mean())} for k, v in split.items()}
    print(json.dumps(cohort, indent=2), flush=True)

    tr, va, te = part["train"], part["val"], part["test"]
    prob_va: dict[str, np.ndarray] = {}
    prob_te: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}

    # 1. current MAP only
    lr = LogisticRegression(class_weight="balanced").fit(full.loc[tr, ["map_now"]], y[tr])
    prob_va["map_only"] = lr.predict_proba(full.loc[va, ["map_now"]])[:, 1]
    prob_te["map_only"] = lr.predict_proba(full.loc[te, ["map_now"]])[:, 1]

    # 2-4. gradient boosting on different feature sets
    for name, cols in [("record_only", rec_cols), ("wave_features", wave_cols), ("fusion", wave_cols + rec_cols)]:
        model, best, predict = fit_gbm(full.loc[tr, cols], y[tr], full.loc[va, cols], y[va])
        prob_va[name], prob_te[name] = predict(full.loc[va, cols]), predict(full.loc[te, cols])
        info[name] = {"n_features": len(cols), "boosting_rounds": best}
        print(f"{name}: {best} rounds, validation AUROC {roc_auc_score(y[va], prob_va[name]):.4f}", flush=True)

    # 5. modality ablation of the fusion model: drop one group of record features at a time
    ablation = {}
    for group, cols in enc.groups.items():
        kept = [c for c in wave_cols + rec_cols if c not in cols]
        _, _, predict = fit_gbm(full.loc[tr, kept], y[tr], full.loc[va, kept], y[va])
        ablation["fusion_without_" + group] = float(roc_auc_score(y[te], predict(full.loc[te, kept])))

    # 6. CNN on the raw waveform
    if not skip_cnn:
        net, history = cnn_mod.train(waves_p[tr], y[tr], waves_p[va], y[va])
        prob_va["wave_cnn"], prob_te["wave_cnn"] = cnn_mod.predict(net, waves_p[va]), cnn_mod.predict(net, waves_p[te])
        info["wave_cnn"] = {"history": history, "n_weights": int(sum(p.numel() for p in net.parameters()))}

    # --- evaluation on the test patients ---
    metrics = {}
    for name in prob_te:
        thr = youden_threshold(y[va], prob_va[name])
        metrics[name] = case_bootstrap(y[te], prob_te[name], cases[te], thr)
        metrics[name]["threshold"] = thr
        metrics[name]["calibration"] = calibration_table(y[te], prob_te[name])
        print(f"{name}: test AUROC {metrics[name]['auroc']['value']:.3f} "
              f"{metrics[name]['auroc']['ci95']}, AUPRC {metrics[name]['auprc']['value']:.3f}", flush=True)

    differences = {f"{a}_minus_{b}": paired_auroc_difference(y[te], prob_te[a], prob_te[b], cases[te])
                   for a, b in [("wave_features", "map_only"), ("fusion", "wave_features"), ("fusion", "map_only")]
                   + ([("wave_cnn", "map_only"), ("wave_cnn", "wave_features")] if "wave_cnn" in prob_te else [])}

    # --- sensitivity analysis: the "grey zone" selection used in part of the literature ---
    # negatives are kept only if MAP stays above 75 mmHg in the outcome minute
    fmin = meta_p["future_min"].to_numpy()
    sel = te & ((y == 1) | (fmin > 75))
    grey = {name: float(roc_auc_score(y[sel], p[sel[te]])) for name, p in prob_te.items()}

    # --- permutation importance of the fusion model on validation patients ---
    cols = wave_cols + rec_cols
    model, best, predict = fit_gbm(full.loc[tr, cols], y[tr], full.loc[va, cols], y[va])
    base = roc_auc_score(y[va], predict(full.loc[va, cols]))
    rng = np.random.default_rng(SEED)
    importance = {}
    for c in cols:
        shuffled = full.loc[va, cols].copy()
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
        importance[c] = float(base - roc_auc_score(y[va], predict(shuffled)))

    out = {"cohort": cohort, "models": info, "metrics": metrics, "auroc_differences": differences,
           "fusion_ablation_test_auroc": ablation, "grey_zone_test_auroc": grey,
           "grey_zone_n": {"samples": int(sel.sum()), "events": int(y[sel].sum())},
           "permutation_importance_validation": dict(sorted(importance.items(), key=lambda kv: -kv[1]))}
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    pred = meta_p.loc[te, ["caseid", "t", "map_now", "y", "future_min"]].copy()
    for name, p in prob_te.items():
        pred["p_" + name] = p
    pred.to_csv(out_dir / "predictions_test.csv", index=False)
    print("saved", out_dir / "metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--cases", type=Path, default=Path("data/vitaldb_cases.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/waveform"))
    ap.add_argument("--skip-cnn", action="store_true")
    a = ap.parse_args()
    main(a.data, a.cases, a.out, a.skip_cnn)
