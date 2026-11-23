"""Download arterial pressure waveforms from VitalDB.

VitalDB (https://vitaldb.net) is an open dataset of 6,388 surgical cases recorded at
Seoul National University Hospital. This script selects adult cases under general
anaesthesia that have an invasive arterial pressure waveform (track ``SNUADC/ART``),
and stores each waveform at 100 Hz as a compact NumPy file.

The data are not redistributed in this repository (licence CC BY-NC-SA 4.0).
Run this script to fetch them yourself.

Usage:
    python -m periop.waveform.download --n-cases 200 --out data/vitaldb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

ART_TRACK = "SNUADC/ART"
FS = 100  # Hz


def select_cases(cases: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    """Return the first ``n_cases`` eligible cases, ordered by case id.

    Eligibility:
      * adult (18 years or older)
      * general anaesthesia
      * arterial waveform track present
      * anaesthesia lasting at least 2 hours, so that each case gives enough segments
    Ordering by case id (no random draw) keeps the selection reproducible.
    """
    with_art = set(vitaldb.find_cases([ART_TRACK]))
    c = cases.copy()
    c["ane_hours"] = (c["aneend"] - c["anestart"]) / 3600.0
    keep = (
        (c["age"] >= 18)
        & (c["ane_type"] == "General")
        & (c["caseid"].isin(with_art))
        & (c["ane_hours"] >= 2.0)
    )
    return c.loc[keep].sort_values("caseid").head(n_cases).reset_index(drop=True)


def download(n_cases: int, out_dir: Path, cases_csv: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cases_csv.exists():
        cases = pd.read_csv(cases_csv)
    else:
        cases = pd.read_csv("https://api.vitaldb.net/cases")
        cases.to_csv(cases_csv, index=False)

    chosen = select_cases(cases, n_cases)
    chosen.to_csv(out_dir / "selected_cases.csv", index=False)
    print(f"{len(chosen)} cases selected")

    for k, caseid in enumerate(chosen["caseid"], 1):
        target = out_dir / f"art_{caseid:04d}.npy"
        if target.exists():
            continue
        t0 = time.time()
        try:
            art = vitaldb.load_case(int(caseid), [ART_TRACK], 1 / FS)[:, 0]
        except Exception as exc:  # network errors: skip, can be re-run later
            print(f"case {caseid}: failed ({exc})", flush=True)
            continue
        # float16 keeps 0.1 mmHg resolution in the physiological range and halves the size
        np.save(target, art.astype(np.float16))
        print(f"[{k}/{len(chosen)}] case {caseid}: {len(art) / FS / 60:.0f} min, {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/vitaldb"))
    ap.add_argument("--cases-csv", type=Path, default=Path("data/vitaldb_cases.csv"))
    a = ap.parse_args()
    download(a.n_cases, a.out, a.cases_csv)
