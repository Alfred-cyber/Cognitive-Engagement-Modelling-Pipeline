#!/usr/bin/env python3
"""
run_pipeline.py
----------------
End-to-end entry point. Runs the full study pipeline described in the
research proposal's Methodology (sections 6.6-6.11):

    raw CSVs + task logs  -->  cleaned data  -->  windowed features
        -->  labelled master dataset  -->  model training & comparison
        -->  evaluation  -->  explainability

Usage:
    python run_pipeline.py \\
        --raw-dir data/raw \\
        --task-log-dir data/task_logs \\
        --out-dir data/processed

Each participant needs:
    data/raw/<participant_id>.csv          (Mind Monitor export)
    data/task_logs/<participant_id>.csv    (task log, see templates/task_log_template.csv)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from engagement_pipeline import config, dataset, evaluation, explainability, modeling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Cognitive engagement modelling pipeline")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    p.add_argument("--task-log-dir", type=Path, default=config.TASK_LOG_DIR)
    p.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--window-seconds", type=int, default=config.WINDOW_SECONDS)
    p.add_argument("--test-size", type=float, default=0.25,
                    help="Fraction of PARTICIPANTS (not rows) held out for testing")
    p.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis (can be slow)")
    p.add_argument("--allow-single-participant-demo", action="store_true",
                    help="Smoke-test the pipeline with only 1 participant using a row-level "
                         "split instead of a participant-level split. NOT a valid evaluation "
                         "— use only to check the pipeline runs before you have real multi-"
                         "participant data.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the master dataset across all participants ------------------
    logger.info("=== Step 1/5: Building master dataset ===")
    master = dataset.build_master_dataset(args.raw_dir, args.task_log_dir)
    dataset.save_master_dataset(master, args.out_dir)
    logger.info("Label distribution:\n%s", master["engagement_label"].value_counts())

    # 2. Prepare X/y/groups and split by participant ------------------------
    logger.info("=== Step 2/5: Train/test split (participant-aware) ===")
    X, y, groups = modeling.prepare_xy(master)
    X_train, X_test, y_train, y_test, groups_train, groups_test = (
        modeling.participant_train_test_split(
            X, y, groups, test_size=args.test_size,
            allow_single_participant_fallback=args.allow_single_participant_demo))
    logger.info("Train: %d windows / %d participants | Test: %d windows / %d participants",
                len(X_train), len(set(groups_train)), len(X_test), len(set(groups_test)))

    # 3. Train & tune all candidate models -----------------------------------
    logger.info("=== Step 3/5: Training models (Logistic Regression, RF, SVM, XGBoost) ===")
    trained = modeling.train_all_models(X_train, y_train, groups_train)

    # 4. Evaluate and compare -------------------------------------------------
    logger.info("=== Step 4/5: Evaluation ===")
    comparison = evaluation.compare_models(trained, X_test, y_test)
    comparison.to_csv(args.out_dir / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    for name, (model, scaler) in trained.items():
        cv_scores = evaluation.cross_validate_model(name, model, X_train, y_train, groups_train)
        result = evaluation.evaluate_model(name, model, scaler, X_test, y_test)
        logger.info("--- %s ---\nCV f1_macro: %.3f +/- %.3f\n%s",
                    name, cv_scores.mean(), cv_scores.std(), result["classification_report"])

    best_name = comparison.iloc[0]["model"]
    best_model, best_scaler = trained[best_name]
    logger.info("Best model by macro F1: %s", best_name)

    # 5. Explainability on the best model --------------------------------------
    logger.info("=== Step 5/5: Explainability (%s) ===", best_name)
    feature_names = modeling.feature_columns(master)

    if best_name == "random_forest":
        rf_importance = explainability.rf_feature_importance(best_model, feature_names)
        rf_importance.to_csv(args.out_dir / "feature_importance_rf.csv", index=False)
        logger.info("Top 10 RF features:\n%s", rf_importance.head(10).to_string(index=False))

    perm_importance = explainability.permutation_importance_analysis(
        best_model, best_scaler, X_test, y_test, feature_names)
    perm_importance.to_csv(args.out_dir / "feature_importance_permutation.csv", index=False)
    logger.info("Top 10 permutation-importance features:\n%s",
                perm_importance.head(10).to_string(index=False))

    if not args.skip_shap:
        shap_values = explainability.shap_analysis(best_model, best_scaler, X_train, X_test)
        if shap_values is not None:
            import numpy as np
            mean_abs_shap = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=(0, 2))
                                 if shap_values.values.ndim == 3
                                 else np.abs(shap_values.values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            mean_abs_shap.to_csv(args.out_dir / "feature_importance_shap.csv", index=False)
            logger.info("Top 10 SHAP features:\n%s", mean_abs_shap.head(10).to_string(index=False))

    logger.info("Done. Outputs written to %s", args.out_dir)


if __name__ == "__main__":
    main()
