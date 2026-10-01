"""
modeling.py
-----------
Implements Methodology 6.9 (Machine Learning Modelling):
  - Logistic Regression (baseline), Random Forest, SVM, XGBoost
  - Train-test split and cross-validation
  - Hyperparameter tuning (light grid search; extend as needed)
  - Performance comparison

Splits are participant-aware (GroupKFold / group-based train-test split) so
that no participant's windows appear in both train and test — otherwise
performance is optimistically biased, since adjacent windows from one
person are highly correlated.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from . import config

logger = logging.getLogger(__name__)

NON_FEATURE_COLS = {
    "participant_id", "window_start", "window_end", "n_samples",
    "engagement_label", "task_type", "self_report_score",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_xy(master: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split the master dataset into X (numeric features), y (encoded labels),
    groups (participant_id, for group-aware CV). Rows with missing feature
    values are dropped (expected to be rare after preprocessing).
    """
    cols = feature_columns(master)
    df = master.dropna(subset=cols + ["engagement_label"]).copy()

    X = df[cols]
    y = df["engagement_label"].map(config.LABEL_TO_INT).to_numpy()
    groups = df["participant_id"].to_numpy()
    return X, y, groups


def participant_train_test_split(X, y, groups, test_size: float = 0.25,
                                  random_state: int = config.RANDOM_STATE,
                                  allow_single_participant_fallback: bool = False):
    """
    Hold out entire participants for testing, not individual rows.

    Requires at least 2 distinct participants (one full participant must be
    available for each side of the split). With 1 participant, there is no
    valid way to hold out "a participant" for testing — add more participant
    data, or pass allow_single_participant_fallback=True to fall back to a
    row-level stratified split FOR PIPELINE TESTING ONLY. That fallback does
    not test generalisation to a new person and must not be used to report
    real model performance (Methodology 6.9/6.10 call for genuine held-out
    evaluation).
    """
    n_participants = len(np.unique(groups))

    if n_participants < 2:
        if not allow_single_participant_fallback:
            raise ValueError(
                f"Only {n_participants} participant(s) found ({np.unique(groups)}). "
                "Participant-aware train/test splitting needs at least 2 participants "
                "so a whole participant can be held out for testing. Add more "
                "participant CSVs + task logs to data/raw/ and data/task_logs/, or "
                "re-run with --allow-single-participant-demo to smoke-test the "
                "pipeline on one participant (NOT a valid evaluation).")
        logger.warning(
            "Only 1 participant available: falling back to a row-level stratified "
            "split for pipeline smoke-testing. This does NOT evaluate generalisation "
            "to a new participant and should not be reported as model performance.")
        from sklearn.model_selection import train_test_split
        strat = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=strat)
        return (X.iloc[train_idx], X.iloc[test_idx],
                y[train_idx], y[test_idx],
                groups[train_idx], groups[test_idx])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx],
            y[train_idx], y[test_idx],
            groups[train_idx], groups[test_idx])


def build_models() -> dict:
    """
    Candidate models from Methodology 6.9. Each is wrapped with a
    StandardScaler for the scale-sensitive ones (SVM, LogReg); tree models
    don't need scaling but it's harmless to include for a consistent API.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=config.RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=config.RANDOM_STATE, class_weight="balanced"),
        "svm": SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE,
                    class_weight="balanced"),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, random_state=config.RANDOM_STATE,
            eval_metric="mlogloss", tree_method="hist")
    else:
        logger.warning("xgboost not installed; skipping it. `pip install xgboost` to enable.")
    return models


# Light hyperparameter grids (Methodology 6.9: "hyperparameter tuning where appropriate")
PARAM_GRIDS = {
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5]},
    "svm": {"C": [0.5, 1.0, 5.0], "gamma": ["scale", "auto"]},
    "xgboost": {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
}


def _cv_splitter(groups, n_folds: int = config.N_CV_FOLDS):
    """
    GroupKFold across participants when >=2 participants are present (the
    normal, correct case). Falls back to plain KFold — WITH A WARNING — when
    only 1 participant is available, since GroupKFold needs at least as many
    groups as folds; this fallback is for single-participant smoke-testing
    only (see modeling.participant_train_test_split).
    """
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        return GroupKFold(n_splits=min(n_folds, n_groups))
    logger.warning("Only 1 participant in this split: using plain KFold for cross-validation "
                    "(not participant-aware). Add more participants for a valid evaluation.")
    return KFold(n_splits=min(n_folds, len(groups)))


def train_with_tuning(name: str, model, X_train, y_train, groups_train, n_folds: int = config.N_CV_FOLDS):
    """GridSearchCV with GroupKFold so tuning itself doesn't leak participants."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    cv = _cv_splitter(groups_train, n_folds)
    grid = PARAM_GRIDS.get(name, {})

    if grid:
        search = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_scaled, y_train, groups=groups_train)
        logger.info("%s best params: %s", name, search.best_params_)
        best_model = search.best_estimator_
    else:
        best_model = model.fit(X_scaled, y_train)

    return best_model, scaler


def train_all_models(X_train, y_train, groups_train) -> dict:
    """Train & tune every candidate model. Returns {name: (model, scaler)}."""
    results = {}
    for name, model in build_models().items():
        logger.info("Training %s ...", name)
        results[name] = train_with_tuning(name, model, X_train, y_train, groups_train)
    return results
