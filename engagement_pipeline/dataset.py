"""
dataset.py
----------
Ties preprocessing + labelling + feature extraction together across all
participants into a single "master dataset" (Methodology 6.9's modelling
target variable requires exactly this: one row per labelled time window,
with a participant_id column so that group-aware splitting is possible).

Also adds task/behavioural features (Methodology 6.8, "Task and behavioural
features": task type, self-reported engagement score).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config, features, io_utils, labeling, preprocessing

logger = logging.getLogger(__name__)


def _attach_task_metadata(window_df: pd.DataFrame, task_log: pd.DataFrame) -> pd.DataFrame:
    """Add task_type and self_report_score columns, looked up by window_start."""
    def lookup(ts):
        match = task_log[(task_log["start_time"] <= ts) & (ts < task_log["end_time"])]
        if match.empty:
            return pd.Series({"task_type": None, "self_report_score": pd.NA})
        row = match.iloc[0]
        return pd.Series({"task_type": row["task"], "self_report_score": row.get("self_report_score")})

    meta = window_df["window_start"].apply(lookup)
    return pd.concat([window_df, meta], axis=1)


def build_participant_dataset(participant_id: str, raw_df: pd.DataFrame,
                               task_log: pd.DataFrame) -> pd.DataFrame:
    """
    Full per-participant pipeline: clean -> window -> extract features ->
    label -> attach task metadata -> tag with participant_id.
    """
    clean_df, _event_rows = preprocessing.clean_participant_data(raw_df)
    if clean_df.empty:
        logger.warning("Participant %s has no usable samples after cleaning; skipping.", participant_id)
        return pd.DataFrame()

    window_feats = features.windowed_features(clean_df)
    if window_feats.empty:
        logger.warning("Participant %s produced no valid windows; skipping.", participant_id)
        return pd.DataFrame()

    labelled = labeling.label_windows(window_feats, task_log)
    if labelled.empty:
        logger.warning("Participant %s: no windows matched the task log; skipping.", participant_id)
        return pd.DataFrame()

    labelled = _attach_task_metadata(labelled, task_log)
    labelled.insert(0, "participant_id", participant_id)
    return labelled


def build_master_dataset(raw_dir: Path, task_log_dir: Path) -> pd.DataFrame:
    """
    Loop over every participant in raw_dir, matching each to its task log in
    task_log_dir by file stem (P01.csv <-> P01.csv), and stack the results.
    """
    raw_dir, task_log_dir = Path(raw_dir), Path(task_log_dir)
    frames = []

    for participant_id, raw_df in io_utils.iter_participants(raw_dir):
        log_path = task_log_dir / f"{participant_id}.csv"
        if not log_path.exists():
            logger.warning("No task log found for %s (expected %s); skipping participant.",
                            participant_id, log_path)
            continue

        task_log = labeling.load_task_log(log_path)
        participant_df = build_participant_dataset(participant_id, raw_df, task_log)
        if not participant_df.empty:
            frames.append(participant_df)

    if not frames:
        raise RuntimeError("No participant produced any labelled windows. Check task logs "
                            "and that each CSV's TimeStamp range overlaps its task log.")

    master = pd.concat(frames, ignore_index=True)
    logger.info("Master dataset: %d windows from %d participants",
                len(master), master["participant_id"].nunique())
    return master


def save_master_dataset(master: pd.DataFrame, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "master_dataset.csv"
    master.to_csv(path, index=False)
    logger.info("Saved master dataset to %s", path)
    return path
