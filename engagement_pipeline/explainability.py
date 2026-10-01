"""
explainability.py
------------------
Implements Methodology 6.11 (Explainability and Interpretation):
  - Random Forest feature importance
  - Permutation importance
  - SHAP values

Used to identify which EEG and movement features contribute most strongly
to engagement prediction (Research Objective 5 / Research Question 2).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

try:
    import shap
    logging.getLogger("shap").setLevel(logging.WARNING)
    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False


def rf_feature_importance(rf_model, feature_names: list[str]) -> pd.DataFrame:
    """Built-in Gini importance from a fitted RandomForestClassifier."""
    importances = rf_model.feature_importances_
    return (pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


def permutation_importance_analysis(model, scaler, X_test, y_test, feature_names: list[str],
                                     n_repeats: int = 20, random_state: int = 42) -> pd.DataFrame:
    """Model-agnostic permutation importance on the held-out test set."""
    X_scaled = scaler.transform(X_test)
    result = permutation_importance(model, X_scaled, y_test, n_repeats=n_repeats,
                                      random_state=random_state, scoring="f1_macro", n_jobs=-1)
    return (pd.DataFrame({
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def shap_analysis(model, scaler, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """
    Compute SHAP values for a fitted model. Works best with tree models
    (Random Forest, XGBoost) via shap.TreeExplainer; falls back to a
    model-agnostic KernelExplainer (slower) for others.

    Returns the shap.Explanation object, or None if shap isn't installed.
    """
    if not HAS_SHAP:
        logger.warning("shap is not installed; skipping SHAP analysis. `pip install shap` to enable.")
        return None

    X_bg_scaled = scaler.transform(X_background)
    X_explain_scaled = scaler.transform(X_explain)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_explain_scaled)
    except Exception:
        logger.info("Falling back to KernelExplainer for SHAP (model is not tree-based).")
        background_sample = shap.sample(X_bg_scaled, min(50, len(X_bg_scaled)))
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        shap_values = explainer(X_explain_scaled[: min(100, len(X_explain_scaled))])

    return shap_values
