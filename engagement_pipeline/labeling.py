"""
labeling.py
-----------
Assigns engagement labels (low / moderate / high) to time windows, per
Methodology 6.5 (experimental tasks) and 6.7 ("Labelling each time window
... based on task type, self-report and behavioural indicators").

Two sources of task timing are supported, and can be combined:

1. A per-participant task log CSV (recommended, most reliable):

       data/task_logs/P01.csv
       task,start_time,end_time,self_report_score
       resting_baseline,2026-08-13 14:52:18,2026-08-13 14:55:18,1
       passive_video,2026-08-13 14:55:20,2026-08-13 15:00:20,3
       ...

   start_time/end_time must be parseable timestamps covering the same clock
   time as the Mind Monitor recording. self_report_score (1-5 Likert, from
   Methodology 6.6 step 5) is optional but, if present, is blended with the
   task's default engagement level (see `apply_self_report_adjustment`).

2. In-app markers logged in Mind Monitor's `Elements` column (e.g.
   "/Marker/1"). If no task log is supplied, consecutive marker-to-marker
   spans are treated as task boundaries, and the analyst must map marker
   indices to task names manually in TASK_LOG template (see templates/).

A template task log is provided at templates/task_log_template.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_task_log(path: Path) -> pd.DataFrame:
    """Load a participant's task log CSV and parse timestamps."""
    log = pd.read_csv(path)
    required = {"task", "start_time", "end_time"}
    missing = required - set(log.columns)
    if missing:
        raise ValueError(f"Task log {path} is missing required columns: {missing}")

    log["start_time"] = pd.to_datetime(log["start_time"])
    log["end_time"] = pd.to_datetime(log["end_time"])
    if "self_report_score" not in log.columns:
        log["self_report_score"] = pd.NA
    return log


def default_label_for_task(task_name: str) -> str:
    """Map a free-text task name to its expected engagement level (6.5 table)."""
    name = task_name.strip().lower().replace("_", " ").replace("-", " ")
    for key, label in config.DEFAULT_TASK_ENGAGEMENT.items():
        normalized_key = key.replace("-", " ")
        if normalized_key in name:
            return label
    logger.warning("No default engagement mapping for task '%s'; defaulting to 'moderate'", task_name)
    return "moderate"


def apply_self_report_adjustment(default_label: str, self_report_score) -> str:
    """
    Blend the task's expected engagement level with the participant's own
    1-5 Likert rating collected after each task (Methodology 6.6, step 5).

    Rule: a strongly discordant self-report (<=2 on a task expected to be
    high/moderate, or >=4 on a task expected to be low) shifts the label one
    step toward the participant's report, rather than overriding it outright
    — keeping the task-design ground truth as the primary signal while still
    using the self-report as corroborating evidence, as the proposal specifies.
    """
    if pd.isna(self_report_score):
        return default_label

    score = float(self_report_score)
    order = config.ENGAGEMENT_LABELS  # low, moderate, high
    idx = order.index(default_label)

    if score <= 2 and idx > 0:
        idx -= 1
    elif score >= 4 and idx < len(order) - 1:
        idx += 1
    return order[idx]


def label_timestamp(ts: pd.Timestamp, task_log: pd.DataFrame) -> str | None:
    """Return the engagement label whose task window contains `ts`, or None."""
    match = task_log[(task_log["start_time"] <= ts) & (ts < task_log["end_time"])]
    if match.empty:
        return None
    row = match.iloc[0]
    base_label = default_label_for_task(row["task"])
    return apply_self_report_adjustment(base_label, row.get("self_report_score"))


def label_windows(window_summary: pd.DataFrame, task_log: pd.DataFrame) -> pd.DataFrame:
    """
    Attach an 'engagement_label' column to a DataFrame of windows that has a
    'window_start' timestamp column (as produced by features.windowed_features).
    Windows that fall outside every logged task span are dropped.
    """
    labels = window_summary["window_start"].apply(lambda ts: label_timestamp(ts, task_log))
    out = window_summary.copy()
    out["engagement_label"] = labels
    n_unlabelled = out["engagement_label"].isna().sum()
    if n_unlabelled:
        logger.info("Dropping %d/%d windows outside any logged task span", n_unlabelled, len(out))
    return out.dropna(subset=["engagement_label"]).reset_index(drop=True)


def extract_markers_from_events(event_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Pull out "/Marker/n" style rows from Mind Monitor's Elements column, for
    studies that rely on in-app markers instead of (or alongside) a manual
    task log. Returns a DataFrame with columns [marker, TimeStamp].
    """
    if event_rows.empty or "Elements" not in event_rows.columns:
        return pd.DataFrame(columns=["marker", "TimeStamp"])

    markers = event_rows[event_rows["Elements"].astype(str).str.contains("Marker", na=False)]
    return markers[["Elements", "TimeStamp"]].rename(columns={"Elements": "marker"}).reset_index(drop=True)
