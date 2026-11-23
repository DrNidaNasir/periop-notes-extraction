"""Turn continuous arterial pressure recordings into labelled prediction samples.

Prediction task
---------------
At time t, using only data up to t, predict whether the patient will be
hypotensive five minutes later.

    input      the last 30 s of the arterial waveform (100 Hz) and the last 5 min of
               mean arterial pressure (MAP, 1 Hz)
    gap        5 min (the lead time a clinician would have to act)
    outcome    1 if MAP stays below 65 mmHg for the whole minute [t+5 min, t+6 min],
               0 otherwise

Two design choices matter for an honest result:

* No "grey zone" is removed. Several published models drop the samples whose future
  MAP lies between 65 and 75 mmHg; that makes the task easier than it is in practice
  (Enevoldsen and Vistisen, Anesthesiology 2022). Here every valid sample is kept.
* Only samples where the patient is NOT hypotensive at time t are used in the primary
  analysis, so the model has to predict the onset, not report an ongoing episode.

Signal quality
--------------
Arterial lines are flushed, zeroed, kinked and disconnected during surgery. A sample is
discarded if its input or outcome window contains values outside 20 to 250 mmHg, a jump
of more than 40 mmHg between two samples (10 ms), or a pulse pressure outside
15 to 150 mmHg in any 2-second block (flat or damped line, or flush artefact).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FS = 100                 # Hz, waveform sampling rate
WIN_S = 30               # seconds of waveform used as input
HIST_S = 300             # seconds of MAP history used as input
GAP_S = 300              # prediction horizon
OUT_S = 60               # outcome window length
STRIDE_S = 30            # one sample every 30 s
MAP_SMOOTH_S = 10        # MAP = mean pressure over the trailing 10 s
HYPO = 65.0              # mmHg


def clean(art: np.ndarray) -> np.ndarray:
    """Replace physiologically impossible values and jumps by NaN."""
    a = art.astype(np.float32).copy()
    a[(a < 20) | (a > 250)] = np.nan
    jump = np.abs(np.diff(a, prepend=a[0])) > 40
    a[jump] = np.nan
    return a


def map_1hz(a: np.ndarray) -> np.ndarray:
    """Mean arterial pressure once per second, averaged over the trailing 10 seconds.

    A second is NaN if more than 20% of the samples in its 10-s window are missing.
    """
    n_sec = len(a) // FS
    per_sec_sum = np.nansum(a[: n_sec * FS].reshape(n_sec, FS), axis=1)
    per_sec_cnt = np.sum(~np.isnan(a[: n_sec * FS].reshape(n_sec, FS)), axis=1)
    k = MAP_SMOOTH_S
    csum = np.concatenate([[0], np.cumsum(per_sec_sum)])
    ccnt = np.concatenate([[0], np.cumsum(per_sec_cnt)])
    out = np.full(n_sec, np.nan, dtype=np.float32)
    s, c = csum[k:] - csum[:-k], ccnt[k:] - ccnt[:-k]
    ok = c >= 0.8 * k * FS
    out[k - 1:][ok] = (s[ok] / c[ok]).astype(np.float32)
    return out


def good_window(w: np.ndarray) -> bool:
    """Quality check of a raw waveform window (see module docstring)."""
    if np.isnan(w).any():
        return False
    blocks = w[: len(w) // (2 * FS) * 2 * FS].reshape(-1, 2 * FS)
    pp = blocks.max(axis=1) - blocks.min(axis=1)
    return bool((pp > 15).all() and (pp < 150).all())


def samples_from_case(caseid: int, art: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (meta, waveforms, map_history) for one case.

    meta         one row per sample: caseid, t (s), map_now, y, hypo_now, and the lowest and
                 highest MAP in the outcome minute (used for the grey-zone sensitivity analysis)
    waveforms    float16 array (n, WIN_S * FS / 2), the 30-s input downsampled to 50 Hz
    map_history  float32 array (n, HIST_S), MAP at 1 Hz over the 5 min before t
    """
    a = clean(art)
    m = map_1hz(a)
    rows, waves, hists = [], [], []
    first_t = HIST_S + MAP_SMOOTH_S
    last_t = len(m) - (GAP_S + OUT_S)
    for t in range(first_t, last_t, STRIDE_S):
        hist = m[t - HIST_S:t]
        future = m[t + GAP_S:t + GAP_S + OUT_S]
        if np.isnan(hist).any() or np.isnan(future).any():
            continue
        w = a[(t - WIN_S) * FS:t * FS]
        if not good_window(w):
            continue
        rows.append((caseid, t, float(hist[-1]), int((future < HYPO).all()), int(hist[-1] < HYPO),
                     float(future.min()), float(future.max())))
        waves.append(w[::2].astype(np.float16))          # 50 Hz is enough for the pressure pulse
        hists.append(hist)
    meta = pd.DataFrame(rows, columns=["caseid", "t", "map_now", "y", "hypo_now", "future_min", "future_max"])
    if not rows:
        return meta, np.empty((0, WIN_S * FS // 2), np.float16), np.empty((0, HIST_S), np.float32)
    return meta, np.stack(waves), np.stack(hists)


def build(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metas, waves, hists = [], [], []
    files = sorted(in_dir.glob("art_*.npy"))
    for k, f in enumerate(files, 1):
        caseid = int(f.stem.split("_")[1])
        meta, w, h = samples_from_case(caseid, np.load(f))
        metas.append(meta); waves.append(w); hists.append(h)
        print(f"[{k}/{len(files)}] case {caseid}: {len(meta)} samples, {int(meta['y'].sum())} positive", flush=True)
    meta = pd.concat(metas, ignore_index=True)
    meta.to_csv(out_dir / "samples.csv", index=False)
    np.save(out_dir / "waveforms.npy", np.concatenate(waves))
    np.save(out_dir / "map_history.npy", np.concatenate(hists))
    print(f"{len(meta)} samples from {meta['caseid'].nunique()} cases; "
          f"{meta['y'].mean():.1%} positive; {meta['hypo_now'].mean():.1%} already hypotensive at t")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/vitaldb"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    a = ap.parse_args()
    build(a.inp, a.out)
