"""Logistic regression model for attrition prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "age", "years_at_company", "salary", "employee_satisfaction",
    "last_performance_rating", "promotions_last_5_years",
    "years_since_last_promotion", "work_accidents", "training_hours",
    "manager_changes", "recent_rating", "rating_trend", "avg_rating",
    "rating_volatility", "promotion_gap_flag", "high_manager_change",
    "salary_below_avg", "salary_ratio", "satisfaction_below_avg",
    "engagement_score", "risk_score",
]


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in *df*."""
    return [c for c in FEATURE_COLS if c in df.columns]


def train_attrition_model(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a logistic regression classifier and return metrics.

    Args:
        df: DataFrame with engineered features and ``attrition`` target.
        config: ``model`` section from the YAML config.

    Returns:
        Dictionary containing model metrics, feature importances,
        and cross-validation scores.
    """
    model_config = config or {}
    test_size = model_config.get("test_size", 0.2)
    cv_folds = model_config.get("cv_folds", 5)
    random_state = model_config.get("random_state", 42)

    feature_cols = _select_features(df)
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame")

    X = df[feature_cols].values
    y = df["attrition"].values

    logger.info("Training model — samples: %d, features: %d", X.shape[0], X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv_folds, scoring="roc_auc")
    metrics["cross_validation"] = {
        "folds": cv_folds,
        "roc_auc_mean": round(float(cv_scores.mean()), 4),
        "roc_auc_std": round(float(cv_scores.std()), 4),
        "scores": [round(float(s), 4) for s in cv_scores],
    }

    importances = pd.Series(np.abs(model.coef_[0]), index=feature_cols)
    importances = importances.sort_values(ascending=False)
    metrics["feature_importance"] = {
        k: round(float(v), 4) for k, v in importances.items()
    }

    logger.info(
        "Model trained — accuracy: %.3f, ROC-AUC: %.3f, CV-AUC: %.3f ± %.3f",
        metrics["accuracy"],
        metrics["roc_auc"],
        metrics["cross_validation"]["roc_auc_mean"],
        metrics["cross_validation"]["roc_auc_std"],
    )
    return metrics
