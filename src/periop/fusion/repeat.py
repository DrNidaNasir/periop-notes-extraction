"""Robustness check: repeat the comparison over several patient splits.

With 40 test patients, one split can be lucky or unlucky. This script redraws the
60 / 20 / 20 patient split with different seeds and reports the test AUROC of the three
tabular models each time. The CNN is left out to keep the run short.

Usage:
    python -m periop.fusion.repeat --seeds 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .clinical import ClinicalEncoder
from .run import fit_gbm, split_cases


def one_split(seed: int, meta: pd.DataFrame, feats: pd.DataFrame, table: pd.DataFrame) -> dict:
    cases, y = meta["caseid"].to_numpy(), meta["y"].to_numpy()
    split = split_cases(cases, seed)
    tr, va, te = (meta["caseid"].isin(split[k]).to_numpy() for k in ("train", "val", "test"))
    enc = ClinicalEncoder().fit(table[table["caseid"].isin(split["train"])])
    rec = enc.transform(table[table["caseid"].isin(np.unique(cases))]).loc[cases].reset_index(drop=True)
    full = pd.concat([feats, rec], axis=1)
    wave_cols, rec_cols = list(feats.columns), list(rec.columns)

    out = {"seed": seed, "test_patients": int(len(split["test"])), "test_events": int(y[te].sum())}
    lr = LogisticRegression(class_weight="balanced").fit(full.loc[tr, ["map_now"]], y[tr])
    out["map_only"] = float(roc_auc_score(y[te], lr.predict_proba(full.loc[te, ["map_now"]])[:, 1]))
    for name, cols in [("wave_features", wave_cols), ("fusion", wave_cols + rec_cols)]:
        _, _, predict = fit_gbm(full.loc[tr, cols], y[tr], full.loc[va, cols], y[va])
        out[name] = float(roc_auc_score(y[te], predict(full.loc[te, cols])))
    return out


def main(data_dir: Path, cases_csv: Path, out_dir: Path, n_seeds: int) -> None:
    meta = pd.read_csv(data_dir / "samples.csv")
    feats = pd.read_parquet(data_dir / "features.parquet")
    feats["elapsed_min"] = meta["t"].to_numpy() / 60.0
    keep = (meta["hypo_now"] == 0).to_numpy()
    meta, feats = meta[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)
    table = pd.read_csv(cases_csv)

    rows = []
    for seed in range(n_seeds):
        rows.append(one_split(seed, meta, feats, table))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["wave_minus_map"] = df["wave_features"] - df["map_only"]
    df["fusion_minus_wave"] = df["fusion"] - df["wave_features"]
    cols = ["map_only", "wave_features", "fusion", "wave_minus_map", "fusion_minus_wave"]
    summary = {c: {"mean": float(df[c].mean()), "sd": float(df[c].std()), "min": float(df[c].min()), "max": float(df[c].max())} for c in cols}
    summary["splits_where_wave_beats_map"] = int((df["wave_minus_map"] > 0).sum())
    summary["splits_where_fusion_beats_wave"] = int((df["fusion_minus_wave"] > 0).sum())
    summary["n_splits"] = n_seeds
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "repeated_splits.csv", index=False)
    (out_dir / "repeated_splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--cases", type=Path, default=Path("data/vitaldb_cases.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/waveform"))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    main(a.data, a.cases, a.out, a.seeds)
