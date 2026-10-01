"""
io_utils.py
-----------
Discovering and loading raw Mind Monitor exports, one file per participant.

Expected layout:

    data/raw/P01.csv
    data/raw/P02.csv
    data/raw/P03.csv
    ...

The participant ID is taken from the file name stem (e.g. "P01"). Rename
Mind Monitor's auto-generated export files to a consistent "<participant_id>.csv"
before placing them in data/raw/ (see README.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParticipantFile:
    participant_id: str
    path: Path


def discover_participant_files(raw_dir: Path) -> list[ParticipantFile]:
    """Find all participant CSVs in raw_dir, sorted by participant_id."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    return [ParticipantFile(participant_id=f.stem, path=f) for f in files]


def load_raw_csv(path: Path) -> pd.DataFrame:
    """
    Load a single Mind Monitor export.

    Mind Monitor exports contain a mix of data rows (EEG/motion samples) and
    pure event rows (device connect/disconnect, blinks, markers), which show
    up as a row with TimeStamp + Elements populated and almost every other
    column empty. Both row types are preserved here; filtering happens in
    preprocessing.py so that marker events in `Elements` are not lost before
    task labelling.
    """
    df = pd.read_csv(path, low_memory=False)

    if "TimeStamp" not in df.columns:
        raise ValueError(f"{path} does not look like a Mind Monitor export "
                          f"(no TimeStamp column found)")

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").reset_index(drop=True)
    return df


def iter_participants(raw_dir: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (participant_id, raw_dataframe) for every file in raw_dir."""
    for pf in discover_participant_files(raw_dir):
        logger.info("Loading %s", pf.path.name)
        yield pf.participant_id, load_raw_csv(pf.path)
