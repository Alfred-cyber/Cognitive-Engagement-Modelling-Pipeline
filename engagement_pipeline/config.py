"""
config.py
---------
Central place for constants used across the pipeline: channel names, EEG
bands, windowing parameters, signal-quality thresholds and engagement
label mapping. Edit values here rather than inside the individual modules.

Corresponds to Methodology sections 6.4, 6.7, 6.8 of the research proposal.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (override via run_pipeline.py --raw-dir / --task-log-dir / --out-dir)
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path("data/raw")          # one Mind Monitor CSV per participant
TASK_LOG_DIR = Path("data/task_logs")    # one task-log CSV per participant (optional)
OUTPUT_DIR = Path("data/processed")      # cleaned data, master dataset, model outputs

# ---------------------------------------------------------------------------
# Muse / Mind Monitor channel layout
# ---------------------------------------------------------------------------
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EEG_BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

# Column name helper: BAND_CHANNEL, e.g. "Alpha_AF7"
def band_col(band: str, channel: str) -> str:
    return f"{band}_{channel}"

RAW_EEG_COLS = [f"RAW_{ch}" for ch in EEG_CHANNELS]
ACC_COLS = ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"]
GYRO_COLS = ["Gyro_X", "Gyro_Y", "Gyro_Z"]
HSI_COLS = [f"HSI_{ch}" for ch in EEG_CHANNELS]

# Frontal pair used for frontal asymmetry; temporal pair for temporal asymmetry
FRONTAL_PAIR = ("AF7", "AF8")
TEMPORAL_PAIR = ("TP9", "TP10")

# ---------------------------------------------------------------------------
# Signal quality (Methodology 6.7)
# ---------------------------------------------------------------------------
# Mind Monitor HSI values: 1 = good, 2 = medium, 4 = poor/no contact.
HSI_GOOD_MAX = 2
# Minimum fraction of samples in a window that must have good HSI on a
# channel for that channel (and therefore the window) to be kept.
MIN_GOOD_HSI_FRACTION = 0.7

# ---------------------------------------------------------------------------
# Windowing / segmentation (Methodology 6.7)
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 5
STEP_SECONDS = 5          # = WINDOW_SECONDS for non-overlapping windows
MIN_SAMPLES_PER_WINDOW = 3  # Mind Monitor EEG rows arrive roughly once/sec

# ---------------------------------------------------------------------------
# Engagement labels (Methodology 6.5, 6.9)
# ---------------------------------------------------------------------------
ENGAGEMENT_LABELS = ["low", "moderate", "high"]
LABEL_TO_INT = {"low": 0, "moderate": 1, "high": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Default task -> expected engagement mapping (Methodology 6.5 table).
# Used as a fallback when a participant's task log doesn't specify a label
# explicitly. Task names are matched case-insensitively, substring match.
DEFAULT_TASK_ENGAGEMENT = {
    "resting baseline": "low",
    "passive video": "moderate",
    "reading": "moderate",
    "comprehension": "moderate",
    "interactive quiz": "high",
    "problem-solving": "high",
    "repetitive": "low",
    "low-stimulation": "low",
}

# ---------------------------------------------------------------------------
# Modelling (Methodology 6.9, 6.10)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_CV_FOLDS = 5
