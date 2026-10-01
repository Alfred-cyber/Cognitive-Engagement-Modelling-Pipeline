"""
preprocessing.py
-----------------
Implements Methodology 6.7 (Data Preprocessing):
  - Removal of connection/event rows and missing values
  - Filtering of poor-quality EEG segments using HSI indicators
  - Bandpass filtering where raw EEG is available
  - Normalisation/standardisation of features (applied later, in dataset.py,
    AFTER features are extracted, to avoid leaking test-set statistics)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from . import config

logger = logging.getLogger(__name__)


def split_data_and_event_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mind Monitor interleaves genuine EEG/motion samples with pure event rows
    (connects, blinks, markers) that have TimeStamp + Elements populated and
    everything else blank.

    Returns (data_rows, event_rows). event_rows is kept separately so marker
    events survive for task labelling even though they're dropped from the
    numeric signal.
    """
    has_eeg = df[[config.band_col("Alpha", ch) for ch in config.EEG_CHANNELS]].notna().any(axis=1)
    data_rows = df[has_eeg].copy()
    event_rows = df[~has_eeg & df["Elements"].notna()].copy() if "Elements" in df.columns else df.iloc[0:0]
    return data_rows, event_rows


def filter_by_signal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop samples where too many channels report poor Horseshoe Signal
    Indicator (HSI) contact. Mind Monitor HSI: 1=good, 2=medium, 4=poor/off.
    """
    available = [c for c in config.HSI_COLS if c in df.columns]
    if not available:
        logger.warning("No HSI columns found; skipping signal-quality filtering.")
        return df

    good = (df[available] <= config.HSI_GOOD_MAX).mean(axis=1) >= config.MIN_GOOD_HSI_FRACTION
    n_dropped = (~good).sum()
    if n_dropped:
        logger.info("Dropping %d/%d samples for poor HSI signal quality", n_dropped, len(df))
    return df[good].copy()


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 1.0, high: float = 40.0,
                     order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter for a raw EEG channel."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    # filtfilt needs a signal longer than ~3x the filter order
    if len(signal) <= order * 3:
        return signal
    return filtfilt(b, a, signal)


def apply_bandpass_to_raw_eeg(df: pd.DataFrame, fs: float = 256.0) -> pd.DataFrame:
    """
    Apply a 1-40 Hz bandpass filter to the RAW_* channels where present.
    Mind Monitor's raw EEG stream is nominally ~256 Hz; adjust `fs` if your
    device/firmware differs.
    """
    df = df.copy()
    for col in config.RAW_EEG_COLS:
        if col in df.columns and df[col].notna().sum() > 0:
            series = df[col].interpolate(limit_direction="both")
            df[col] = bandpass_filter(series.to_numpy(), fs=fs)
    return df


def clean_participant_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full cleaning pipeline for one participant's raw export.

    Returns (clean_data_rows, event_rows_for_labelling).
    """
    data_rows, event_rows = split_data_and_event_rows(raw_df)
    data_rows = filter_by_signal_quality(data_rows)
    data_rows = apply_bandpass_to_raw_eeg(data_rows)
    data_rows = data_rows.reset_index(drop=True)
    return data_rows, event_rows
