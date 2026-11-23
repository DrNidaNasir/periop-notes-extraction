"""Hand-crafted features from the arterial pressure input.

These are the quantities an anaesthetist reads from the monitor, plus their short-term
trend and variability. They feed the gradient-boosting models and keep the result
interpretable.

From the 30-s waveform (50 Hz):
    sbp, dbp        mean systolic and diastolic pressure over the detected beats
    pp              pulse pressure (sbp - dbp)
    hr              heart rate from the beat-to-beat intervals
    sbp_sd, pp_sd   beat-to-beat variability of systolic and pulse pressure
    rr_sd           variability of the beat intervals (irregular rhythm, ectopy)
    dpdt_max        steepest upstroke of the pulse (a rough index of contractility)
    wave_sd         standard deviation of the raw waveform

From the 5-min MAP history (1 Hz):
    map_now                  MAP at prediction time
    map_mean_1m/3m/5m        mean MAP over the last 1, 3 and 5 minutes
    map_min_5m               lowest MAP in the last 5 minutes
    map_sd_5m                MAP variability over 5 minutes
    map_slope_1m/5m          linear trend in mmHg per minute (negative = falling)
    map_drop_from_max        how far MAP is below its 5-minute maximum
    time_below_75            fraction of the last 5 minutes spent below 75 mmHg
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FS_WAVE = 50


def _slope_per_min(y: np.ndarray) -> float:
    x = np.arange(len(y)) / 60.0
    return float(np.polyfit(x, y, 1)[0])


def beat_features(w: np.ndarray) -> dict:
    """Beat-by-beat features of one 30-s waveform window (50 Hz)."""
    w = w.astype(np.float32)
    # systolic peaks: at least 0.3 s apart (max 200 beats/min) and prominent
    peaks, _ = find_peaks(w, distance=int(0.3 * FS_WAVE), prominence=10)
    out = {"wave_sd": float(w.std()), "dpdt_max": float(np.max(np.diff(w)) * FS_WAVE)}
    if len(peaks) < 5:      # too few beats to describe them; the model handles missing values
        out.update(dict(sbp=np.nan, dbp=np.nan, pp=np.nan, hr=np.nan, sbp_sd=np.nan, pp_sd=np.nan, rr_sd=np.nan))
        return out
    sbp = w[peaks]
    # diastolic value: minimum between consecutive systolic peaks
    dbp = np.array([w[a:b].min() for a, b in zip(peaks[:-1], peaks[1:])])
    rr = np.diff(peaks) / FS_WAVE
    pp = sbp[1:] - dbp
    out.update(dict(sbp=float(sbp.mean()), dbp=float(dbp.mean()), pp=float(pp.mean()), hr=float(60.0 / rr.mean()),
                    sbp_sd=float(sbp.std()), pp_sd=float(pp.std()), rr_sd=float(rr.std())))
    return out


def trend_features(h: np.ndarray) -> dict:
    """Features of the 5-min MAP history (1 Hz, last value = now)."""
    return dict(
        map_now=float(h[-1]),
        map_mean_1m=float(h[-60:].mean()), map_mean_3m=float(h[-180:].mean()), map_mean_5m=float(h.mean()),
        map_min_5m=float(h.min()), map_sd_5m=float(h.std()),
        map_slope_1m=_slope_per_min(h[-60:]), map_slope_5m=_slope_per_min(h),
        map_drop_from_max=float(h.max() - h[-1]),
        time_below_75=float((h < 75).mean()),
    )


def build_feature_table(waveforms: np.ndarray, map_history: np.ndarray) -> pd.DataFrame:
    rows = []
    for w, h in zip(waveforms, map_history):
        f = beat_features(w)
        f.update(trend_features(h))
        rows.append(f)
    return pd.DataFrame(rows)
