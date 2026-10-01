"""
evaluation.py
-------------
Implements Methodology 6.10 (Model Evaluation): accuracy, precision, recall,
F1-score, confusion matrix, ROC-AUC, and cross-validation scores. Because
engagement classes may be imbalanced, macro-averaged F1 and per-class recall
are reported alongside accuracy, as the proposal specifies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score

from . import config, modeling

logger = logging.getLogger(__name__)


def evaluate_model(name: str, model, scaler, X_test, y_test) -> dict:
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_scaled)
            metrics["roc_auc_ovr"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except ValueError as e:
            logger.warning("ROC-AUC not computed for %s: %s", name, e)

    cm = confusion_matrix(y_test, y_pred, labels=list(config.INT_TO_LABEL.keys()))
    report = classification_report(
        y_test, y_pred, labels=list(config.INT_TO_LABEL.keys()),
        target_names=list(config.INT_TO_LABEL.values()), zero_division=0)

    return {"metrics": metrics, "confusion_matrix": cm, "classification_report": report}


def cross_validate_model(name: str, model, X, y, groups, n_folds: int = config.N_CV_FOLDS) -> np.ndarray:
    """Group-aware cross-validation score (macro F1) on the full (train) set."""
    cv = modeling._cv_splitter(groups, n_folds)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=-1)
    logger.info("%s cross-val f1_macro: %.3f +/- %.3f", name, scores.mean(), scores.std())
    return scores


def compare_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Build a summary comparison table across all trained models."""
    rows = []
    for name, (model, scaler) in trained_models.items():
        result = evaluate_model(name, model, scaler, X_test, y_test)
        rows.append(result["metrics"])
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
