"""
features.py
-----------
Implements Methodology 6.8 (Feature Extraction):

  EEG features
    - mean & SD of Delta/Theta/Alpha/Beta/Gamma per channel
    - band ratios: Beta/Alpha, Beta/(Alpha+Theta)            (McMahan et al., 2015)
    - engagement index: (Beta+Gamma) / (Alpha+Theta)
    - frontal asymmetry (AF7 vs AF8), temporal asymmetry (TP9 vs TP10)

  Motion features
    - accelerometer & gyroscope mean/SD, movement magnitude,
      movement variability, jerk (rate of change of movement)

Windows follow Methodology 6.7 ("segmentation into time windows, e.g. 5 or
10 second windows").
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(df: pd.DataFrame, window_seconds: int = config.WINDOW_SECONDS,
                  step_seconds: int = config.STEP_SECONDS) -> Iterator[pd.DataFrame]:
    """Yield successive (possibly overlapping) time-based slices of df."""
    if df.empty:
        return
    start = df["TimeStamp"].iloc[0]
    end = df["TimeStamp"].iloc[-1]
    win = pd.Timedelta(seconds=window_seconds)
    step = pd.Timedelta(seconds=step_seconds)

    t = start
    while t + win <= end:
        window = df[(df["TimeStamp"] >= t) & (df["TimeStamp"] < t + win)]
        if len(window) >= config.MIN_SAMPLES_PER_WINDOW:
            yield window
        t += step


# ---------------------------------------------------------------------------
# EEG features
# ---------------------------------------------------------------------------

def eeg_band_features(window: pd.DataFrame) -> dict:
    """Mean & SD of each band at each channel."""
    feats = {}
    for band in config.EEG_BANDS:
        for ch in config.EEG_CHANNELS:
            col = config.band_col(band, ch)
            if col in window.columns:
                feats[f"{col}_mean"] = window[col].mean()
                feats[f"{col}_std"] = window[col].std()
    return feats


def _band_channel_mean(window: pd.DataFrame, band: str) -> float:
    """Average a band's power across all four channels for a window."""
    cols = [config.band_col(band, ch) for ch in config.EEG_CHANNELS if config.band_col(band, ch) in window.columns]
    return window[cols].mean(axis=1).mean() if cols else np.nan


def band_ratio_features(window: pd.DataFrame) -> dict:
    """Beta/Alpha, Beta/(Alpha+Theta), and the (Beta+Gamma)/(Alpha+Theta) engagement index."""
    alpha = _band_channel_mean(window, "Alpha")
    beta = _band_channel_mean(window, "Beta")
    theta = _band_channel_mean(window, "Theta")
    gamma = _band_channel_mean(window, "Gamma")

    eps = 1e-9  # avoid divide-by-zero on near-flat segments
    return {
        "ratio_beta_alpha": beta / (alpha + eps),
        "ratio_beta_alpha_theta": beta / (alpha + theta + eps),
        "engagement_index": (beta + gamma) / (alpha + theta + eps),
    }


def asymmetry_features(window: pd.DataFrame) -> dict:
    """
    Frontal asymmetry (AF7 vs AF8) and temporal asymmetry (TP9 vs TP10),
    computed per band as (right - left) / (right + left), a standard
    normalised asymmetry index.
    """
    feats = {}
    eps = 1e-9

    for band in config.EEG_BANDS:
        left_f, right_f = config.FRONTAL_PAIR
        col_l, col_r = config.band_col(band, left_f), config.band_col(band, right_f)
        if col_l in window.columns and col_r in window.columns:
            l, r = window[col_l].mean(), window[col_r].mean()
            feats[f"frontal_asym_{band}"] = (r - l) / (abs(r) + abs(l) + eps)

        left_t, right_t = config.TEMPORAL_PAIR
        col_l, col_r = config.band_col(band, left_t), config.band_col(band, right_t)
        if col_l in window.columns and col_r in window.columns:
            l, r = window[col_l].mean(), window[col_r].mean()
            feats[f"temporal_asym_{band}"] = (r - l) / (abs(r) + abs(l) + eps)

    return feats


# ---------------------------------------------------------------------------
# Motion features
# ---------------------------------------------------------------------------

def motion_features(window: pd.DataFrame) -> dict:
    feats = {}

    for cols, prefix in [(config.ACC_COLS, "acc"), (config.GYRO_COLS, "gyro")]:
        available = [c for c in cols if c in window.columns]
        if not available:
            continue
        for c in available:
            feats[f"{c}_mean"] = window[c].mean()
            feats[f"{c}_std"] = window[c].std()

        magnitude = np.sqrt((window[available] ** 2).sum(axis=1))
        feats[f"{prefix}_magnitude_mean"] = magnitude.mean()
        feats[f"{prefix}_magnitude_std"] = magnitude.std()  # movement variability

        dt = window["TimeStamp"].diff().dt.total_seconds()
        jerk = magnitude.diff() / dt
        jerk = jerk.replace([np.inf, -np.inf], np.nan)
        feats[f"{prefix}_jerk_mean"] = jerk.abs().mean()

    return feats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_window_features(window: pd.DataFrame) -> dict:
    """Combine all feature groups for a single time window."""
    feats = {"window_start": window["TimeStamp"].iloc[0], "window_end": window["TimeStamp"].iloc[-1],
              "n_samples": len(window)}
    feats.update(eeg_band_features(window))
    feats.update(band_ratio_features(window))
    feats.update(asymmetry_features(window))
    feats.update(motion_features(window))
    return feats


def windowed_features(df: pd.DataFrame, window_seconds: int = config.WINDOW_SECONDS,
                       step_seconds: int = config.STEP_SECONDS) -> pd.DataFrame:
    """Run extract_window_features over every window in df."""
    rows = [extract_window_features(w) for w in make_windows(df, window_seconds, step_seconds)]
    return pd.DataFrame(rows)
