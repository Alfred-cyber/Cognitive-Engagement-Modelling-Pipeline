# Cognitive Engagement Modelling Pipeline

A modular Python pipeline implementing the Methodology of
*AI-Driven Cognitive Engagement Modelling in Smart Learning Environments
Using Wearable EEG and Multimodal Sensing*.

Takes one Mind Monitor CSV per participant plus a task log per participant,
and produces a labelled master dataset, trained/compared models
(Logistic Regression, Random Forest, SVM, XGBoost), evaluation metrics, and
feature-importance analysis (RF importance, permutation importance, SHAP).

## 1. Folder layout

```
engagement_pipeline/
├── run_pipeline.py                     # CLI entry point
├── requirements.txt
├── data/
│   ├── raw/                            # one Mind Monitor export per participant
│   │   ├── P01.csv
│   │   ├── P02.csv
│   │   └── ...
│   ├── task_logs/                      # one task log per participant (same ID)
│   │   ├── P01.csv
│   │   ├── P02.csv
│   │   └── ...
│   └── processed/                      # pipeline outputs (created automatically)
│       ├── master_dataset.csv
│       ├── model_comparison.csv
│       ├── feature_importance_rf.csv
│       ├── feature_importance_permutation.csv
│       └── feature_importance_shap.csv
└── engagement_pipeline/                # the Python package
    ├── config.py
    ├── io_utils.py
    ├── preprocessing.py
    ├── labeling.py
    ├── features.py
    ├── dataset.py
    ├── modeling.py
    ├── evaluation.py
    ├── explainability.py
    └── templates/
        └── task_log_template.csv
```

## 2. Adding a new participant

1. Export the recording from Mind Monitor and rename it to the participant's
   ID, e.g. `P07.csv`. Place it in `data/raw/`.
2. Copy `engagement_pipeline/templates/task_log_template.csv` to
   `data/task_logs/P07.csv` and fill in the actual task start/end times (from
   your session notes or in-app markers) and the 1-5 self-report score
   collected after each task (Methodology 6.6, step 5).
3. That's it — `run_pipeline.py` auto-discovers every participant by
   matching `data/raw/<id>.csv` to `data/task_logs/<id>.csv`. No code
   changes are needed to add participants.

**Important:** task log `start_time`/`end_time` must be real clock
timestamps overlapping the Mind Monitor recording's `TimeStamp` column —
these are what the pipeline uses to figure out which samples belong to
which task.

## 3. Running the pipeline

```bash
pip install -r requirements.txt
python run_pipeline.py --raw-dir data/raw --task-log-dir data/task_logs --out-dir data/processed
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--window-seconds` | 5 | Window length for segmentation (6.7) |
| `--test-size` | 0.25 | Fraction of **participants** (not rows) held out for testing |
| `--skip-shap` | off | Skip SHAP (can be slow with many windows) |
| `--allow-single-participant-demo` | off | Only needed with exactly 1 participant in `data/raw/`. Falls back to a row-level split so you can smoke-test that the pipeline runs end-to-end. **Not a valid evaluation** — metrics from this mode must not be reported as real model performance. With 2+ participants this flag does nothing. |

With fewer than 2 participants, the normal run stops with a clear error at
the train/test split step rather than a model you can't trust the numbers
from.

## 4. What each stage does

| Stage | Module | Proposal section |
|---|---|---|
| Load raw CSVs | `io_utils.py` | 6.4 |
| Clean, filter by HSI, bandpass raw EEG | `preprocessing.py` | 6.7 |
| Task-log / marker based labelling | `labeling.py` | 6.5, 6.6, 6.7 |
| Windowing + EEG/motion feature extraction | `features.py` | 6.7, 6.8 |
| Build multi-participant dataset | `dataset.py` | 6.8, 6.9 |
| Participant-aware split, train & tune models | `modeling.py` | 6.9 |
| Accuracy/precision/recall/F1/ROC-AUC/CV | `evaluation.py` | 6.10 |
| RF importance, permutation importance, SHAP | `explainability.py` | 6.11 |

## 5. Why splits are participant-aware

EEG signals are highly individual. A random row-level train/test split lets
windows from the same participant appear on both sides, which leaks
identity-specific signal and inflates performance. This pipeline always
splits and cross-validates by whole participants (`GroupShuffleSplit` /
`GroupKFold`), giving an honest estimate of how the model generalises to a
*new* learner — directly relevant to Research Question 4 (feasibility of
Muse as research infrastructure) and Research Objective 6.

## 6. Extending the pipeline

- **More participants**: just drop more files in `data/raw/` and
  `data/task_logs/` (step 2 above).
- **Different window length**: `--window-seconds 10`.
- **LSTM/temporal model** (optional, per 6.9): the master dataset is
  windowed and ordered by `window_start` per participant, so a sequence
  model can be built on top of `dataset.build_participant_dataset()`
  output without re-running preprocessing.
- **In-app markers instead of a task log**: `labeling.extract_markers_from_events()`
  pulls `/Marker/n` events out of the raw export's `Elements` column if you
  prefer marking task boundaries live during the session rather than
  writing a task log afterwards.
