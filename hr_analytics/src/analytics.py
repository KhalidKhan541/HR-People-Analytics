"""Descriptive analytics and equity analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descriptive analytics
# ---------------------------------------------------------------------------

def _attrition_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return attrition rate breakdown for a grouping column."""
    if col not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby(col)["attrition"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "attrition_rate", "sum": "attritions", "count": "headcount"})
        .sort_values("attrition_rate", ascending=False)
    )
    summary["attrition_rate"] = summary["attrition_rate"].round(4)
    return summary


def _salary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Return salary distribution statistics overall and per department."""
    stats_dict: dict[str, Any] = {
        "overall": {
            "mean": float(df["salary"].mean()),
            "median": float(df["salary"].median()),
            "std": float(df["salary"].std()),
            "min": float(df["salary"].min()),
            "max": float(df["salary"].max()),
        },
        "by_department": (
            df.groupby("department")["salary"]
            .agg(["mean", "median", "std", "count"])
            .round(0)
            .to_dict(orient="index")
        ),
    }
    return stats_dict


def _performance_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise performance rating distribution."""
    return {
        "mean": float(df["last_performance_rating"].mean()),
        "median": float(df["last_performance_rating"].median()),
        "std": float(df["last_performance_rating"].std()),
        "distribution": {
            "<2.0": int((df["last_performance_rating"] < 2.0).sum()),
            "2.0-3.0": int(((df["last_performance_rating"] >= 2.0) & (df["last_performance_rating"] < 3.0)).sum()),
            "3.0-4.0": int(((df["last_performance_rating"] >= 3.0) & (df["last_performance_rating"] < 4.0)).sum()),
            "4.0+": int((df["last_performance_rating"] >= 4.0).sum()),
        },
    }


def run_analytics(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a full analytics summary dictionary.

    Returns:
        Nested dictionary with sections: overview, by_department,
        by_level, by_gender, salary_stats, performance.
    """
    logger.info("Running analytics on DataFrame with %d rows", len(df))

    overview = {
        "total_employees": int(len(df)),
        "overall_attrition_rate": round(float(df["attrition"].mean()), 4),
        "avg_satisfaction": round(float(df["employee_satisfaction"].mean()), 2),
        "avg_tenure": round(float(df["years_at_company"].mean()), 2),
        "avg_salary": round(float(df["salary"].mean()), 0),
    }

    summary: dict[str, Any] = {
        "overview": overview,
        "attrition_by_department": _attrition_by_group(df, "department").to_dict(orient="index"),
        "attrition_by_level": _attrition_by_group(df, "level").to_dict(orient="index"),
        "attrition_by_gender": _attrition_by_group(df, "gender").to_dict(orient="index"),
        "attrition_by_tenure": _attrition_by_group(df, "tenure_band").to_dict(orient="index"),
        "salary_statistics": _salary_statistics(df),
        "performance_distribution": _performance_distribution(df),
    }

    logger.info("Analytics complete — overall attrition rate: %.1f%%", overview["overall_attrition_rate"] * 100)
    return summary


# ---------------------------------------------------------------------------
# Equity analysis
# ---------------------------------------------------------------------------

def _chi_square_test(
    df: pd.DataFrame,
    feature: str,
    target: str = "attrition",
) -> dict[str, Any]:
    """Run a chi-square test of independence."""
    if feature not in df.columns:
        return {"feature": feature, "error": "column not found"}
    contingency = pd.crosstab(df[feature], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "feature": feature,
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p), 6),
        "degrees_of_freedom": int(dof),
        "significant_at_005": bool(p < 0.05),
    }


def _bias_ratio(
    df: pd.DataFrame,
    group_col: str,
    positive_col: str = "attrition",
) -> dict[str, Any]:
    """Compute the four-fifths rule bias ratio.

    The ratio compares the highest-rate group to the lowest-rate group.
    A ratio below 0.80 indicates potential adverse impact.
    """
    rates = df.groupby(group_col)[positive_col].mean()
    if rates.min() == 0:
        ratio = float("inf")
    else:
        ratio = round(float(rates.max() / rates.min()), 3)

    return {
        "group_column": group_col,
        "rates_by_group": {k: round(float(v), 4) for k, v in rates.items()},
        "bias_ratio": ratio,
        "potential_adverse_impact": bool(ratio < 0.80),
    }


def compute_equity_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run statistical equity / fairness checks across demographic groups.

    Checks:
        - Chi-square tests for department, level, gender, tenure_band
        - Bias ratios (four-fifths rule) for gender and department

    Returns:
        Dictionary with ``chi_square_tests`` and ``bias_ratios`` lists.
    """
    logger.info("Running equity analysis")

    chi_features = ["department", "level", "gender", "tenure_band"]
    chi_results = [_chi_square_test(df, feat) for feat in chi_features]

    bias_results = [_bias_ratio(df, g) for g in ["gender", "department", "level"]]

    summary = {
        "chi_square_tests": chi_results,
        "bias_ratios": bias_results,
    }

    sig = [r["feature"] for r in chi_results if r.get("significant_at_005")]
    logger.info("Equity analysis complete — significant associations: %s", sig or "none")
    return summary
