"""Unit tests of the waveform code on a synthetic pressure signal (no data needed)."""
import numpy as np

from periop.waveform.features import beat_features, trend_features
from periop.waveform.segments import FS, GAP_S, HYPO, OUT_S, clean, good_window, map_1hz, samples_from_case


def synthetic_art(minutes: float, mean: float = 90.0, hr: float = 75.0) -> np.ndarray:
    """A crude arterial pressure wave: a sine pulse of +/- 25 mmHg around a mean level."""
    t = np.arange(int(minutes * 60 * FS)) / FS
    return (mean + 25 * np.sin(2 * np.pi * hr / 60 * t)).astype(np.float32)


def test_clean_removes_impossible_values():
    a = synthetic_art(1)
    a[100:110] = 300      # flush artefact
    a[500:510] = 0        # disconnected
    c = clean(a)
    assert np.isnan(c[100:110]).all() and np.isnan(c[500:510]).all()


def test_map_matches_the_mean_level():
    m = map_1hz(clean(synthetic_art(2, mean=90)))
    assert abs(np.nanmean(m) - 90) < 1


def test_flat_line_is_rejected():
    assert not good_window(np.full(30 * FS, 80, dtype=np.float32))
    assert good_window(synthetic_art(0.5))


def test_heart_rate_and_pressures():
    f = beat_features(synthetic_art(0.5, mean=90, hr=75)[::2])
    assert abs(f["hr"] - 75) < 2
    assert abs(f["sbp"] - 115) < 2 and abs(f["dbp"] - 65) < 2


def test_trend_sign():
    falling = np.linspace(90, 70, 300).astype(np.float32)
    assert trend_features(falling)["map_slope_5m"] < 0
    assert trend_features(falling[::-1])["map_slope_5m"] > 0


def test_label_uses_only_the_future():
    """Normal pressure for 20 min, then low pressure: samples just before the drop must be positive."""
    a = np.concatenate([synthetic_art(20, mean=90), synthetic_art(10, mean=55)])
    meta, w, h = samples_from_case(1, a)
    drop = 20 * 60
    before = meta[meta["t"] < drop - GAP_S - OUT_S]
    lead = meta[(meta["t"] > drop - GAP_S + 15) & (meta["t"] <= drop)]
    assert (before["y"] == 0).all()
    assert len(lead) > 0 and (lead["y"] == 1).all() and (lead["map_now"] > HYPO).all()
    assert w.shape[1] == 1500 and h.shape[1] == 300
