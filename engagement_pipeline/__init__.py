"""
engagement_pipeline
--------------------
AI-driven cognitive engagement modelling from wearable Muse EEG and
multimodal sensing data (see the accompanying research proposal).

Modules:
    config          - shared constants
    io_utils        - discover & load raw Mind Monitor CSVs
    preprocessing   - cleaning, signal-quality filtering, bandpass filtering
    labeling        - task-log / marker based engagement labelling
    features        - windowing + EEG/motion feature extraction
    dataset         - build per-participant & master multi-participant datasets
    modeling        - participant-aware train/test split, model training
    evaluation      - accuracy/precision/recall/F1/ROC-AUC/cross-validation
    explainability  - RF importance, permutation importance, SHAP
"""

__version__ = "0.1.0"
