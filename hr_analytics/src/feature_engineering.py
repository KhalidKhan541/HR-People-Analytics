"""
Feature Engineering Module for HR Attrition Prediction.

Creates SQL-engineered features for employee attrition risk modeling.
All operations are vectorized using pandas transform for group-level calculations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttritionFeatureEngineer:
    """Create SQL-engineered features for attrition prediction.

    Each method returns a new DataFrame with engineered features appended.
    All group-level calculations use .transform() for vectorized performance.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with employee DataFrame.

        Args:
            df: Raw employee data with columns like employee_id, department,
                performance_rating_1/2/3, salary, tenure_months, etc.
        """
        self.df = df.copy()
        logger.info("AttritionFeatureEngineer initialized with %d records", len(self.df))

    def recent_rating(self) -> pd.DataFrame:
        """Most recent performance rating (rating_3).

        SQL Equivalent:
            SELECT *, performance_rating_3 AS recent_rating
            FROM employees

        Returns:
            DataFrame with 'recent_rating' column added.
        """
        df = self.df.copy()
        df["recent_rating"] = df["performance_rating_3"]
        logger.debug("recent_rating: %s nulls", df["recent_rating"].isna().sum())
        return df

    def rating_trend(self) -> pd.DataFrame:
        """Rating trend: rating_3 - rating_1 (improving/declining).

        Positive = improving, Negative = declining, Zero = stable.

        SQL Equivalent:
            SELECT *,
                   performance_rating_3 - performance_rating_1 AS rating_trend
            FROM employees

        Returns:
            DataFrame with 'rating_trend' column added.
        """
        df = self.df.copy()
        r3 = df["performance_rating_3"]
        r1 = df["performance_rating_1"]
        df["rating_trend"] = r3 - r1
        df["rating_trend"] = df["rating_trend"].fillna(0.0)
        logger.debug("rating_trend: mean=%.3f, std=%.3f", df["rating_trend"].mean(), df["rating_trend"].std())
        return df

    def avg_rating(self) -> pd.DataFrame:
        """Average of last 3 ratings.

        SQL Equivalent:
            SELECT *,
                   (performance_rating_1 + performance_rating_2 + performance_rating_3) / 3.0
                       AS avg_rating
            FROM employees

        Returns:
            DataFrame with 'avg_rating' column added.
        """
        df = self.df.copy()
        rating_cols = ["performance_rating_1", "performance_rating_2", "performance_rating_3"]
        df["avg_rating"] = df[rating_cols].mean(axis=1)
        logger.debug("avg_rating: %.3f", df["avg_rating"].mean())
        return df

    def rating_volatility(self) -> pd.DataFrame:
        """Standard deviation of last 3 ratings.

        High volatility suggests inconsistent performance.

        SQL Equivalent:
            -- No direct SQL equivalent; requires array aggregation:
            SELECT *,
                   STDDEV(ARRAY[performance_rating_1,
                                performance_rating_2,
                                performance_rating_3]) AS rating_volatility
            FROM employees

        Returns:
            DataFrame with 'rating_volatility' column added.
        """
        df = self.df.copy()
        rating_cols = ["performance_rating_1", "performance_rating_2", "performance_rating_3"]
        df["rating_volatility"] = df[rating_cols].std(axis=1, ddof=1)
        df["rating_volatility"] = df["rating_volatility"].fillna(0.0)
        logger.debug("rating_volatility: mean=%.3f", df["rating_volatility"].mean())
        return df

    def promotion_gap(self) -> pd.DataFrame:
        """Months since last promotion. >24 months = high risk.

        SQL Equivalent:
            SELECT *,
                   DATEDIFF(MONTH, last_promotion_date, CURRENT_DATE) AS promotion_gap_months
            FROM employees

        Returns:
            DataFrame with 'promotion_gap_months' column added.
        """
        df = self.df.copy()
        df["promotion_gap_months"] = df["months_since_last_promotion"]
        df["promotion_gap_months"] = df["promotion_gap_months"].fillna(
            df["promotion_gap_months"].median()
        )
        logger.debug("promotion_gap_months: median=%.1f", df["promotion_gap_months"].median())
        return df

    def promotion_gap_flag(self, threshold: int = 24) -> pd.DataFrame:
        """Binary flag: 1 if promotion gap > threshold months.

        SQL Equivalent:
            SELECT *,
                   CASE WHEN DATEDIFF(MONTH, last_promotion_date, CURRENT_DATE) > 24
                        THEN 1 ELSE 0
                   END AS promotion_gap_flag
            FROM employees

        Args:
            threshold: Number of months to flag as high risk. Default 24.

        Returns:
            DataFrame with 'promotion_gap_flag' column added.
        """
        df = self.promotion_gap()
        df["promotion_gap_flag"] = (df["promotion_gap_months"] > threshold).astype(int)
        flagged = df["promotion_gap_flag"].sum()
        logger.debug("promotion_gap_flag: %d/%d flagged (> %d months)", flagged, len(df), threshold)
        return df

    def manager_change_count(self) -> pd.DataFrame:
        """Number of manager changes in last 2 years.

        SQL Equivalent:
            SELECT employee_id,
                   COUNT(DISTINCT manager_id) - 1 AS manager_changes_2yr
            FROM manager_history
            WHERE change_date >= DATEADD(YEAR, -2, CURRENT_DATE)
            GROUP BY employee_id

        Returns:
            DataFrame with 'manager_changes_2yr' column added.
        """
        df = self.df.copy()
        df["manager_changes_2yr"] = df["num_manager_changes_2yr"]
        df["manager_changes_2yr"] = df["manager_changes_2yr"].fillna(0).astype(int)
        logger.debug("manager_changes_2yr: mean=%.2f", df["manager_changes_2yr"].mean())
        return df

    def manager_change_flag(self) -> pd.DataFrame:
        """Binary flag: 1 if manager changed in last 2 years.

        SQL Equivalent:
            SELECT *,
                   CASE WHEN num_manager_changes_2yr > 0 THEN 1 ELSE 0 END
                       AS manager_change_flag
            FROM employees

        Returns:
            DataFrame with 'manager_change_flag' column added.
        """
        df = self.manager_change_count()
        df["manager_change_flag"] = (df["manager_changes_2yr"] > 0).astype(int)
        flagged = df["manager_change_flag"].sum()
        logger.debug("manager_change_flag: %d/%d flagged", flagged, len(df))
        return df

    def salary_vs_dept_avg(self) -> pd.DataFrame:
        """Salary relative to department average (ratio).

        Ratio < 1.0 means below department average.
        Ratio > 1.0 means above department average.

        SQL Equivalent:
            SELECT *,
                   salary / AVG(salary) OVER (PARTITION BY department) AS salary_vs_dept_avg
            FROM employees

        Returns:
            DataFrame with 'salary_vs_dept_avg' column added.
        """
        df = self.df.copy()
        dept_avg = df.groupby("department")["salary"].transform("mean")
        df["salary_vs_dept_avg"] = df["salary"] / dept_avg.replace(0, np.nan)
        df["salary_vs_dept_avg"] = df["salary_vs_dept_avg"].fillna(1.0)
        logger.debug("salary_vs_dept_avg: mean=%.3f", df["salary_vs_dept_avg"].mean())
        return df

    def salary_vs_level_avg(self) -> pd.DataFrame:
        """Salary relative to level average (ratio).

        Compares employee salary to the average salary for their job level,
        controlling for seniority differences.

        SQL Equivalent:
            SELECT *,
                   salary / AVG(salary) OVER (PARTITION BY job_level) AS salary_vs_level_avg
            FROM employees

        Returns:
            DataFrame with 'salary_vs_level_avg' column added.
        """
        df = self.df.copy()
        level_avg = df.groupby("job_level")["salary"].transform("mean")
        df["salary_vs_level_avg"] = df["salary"] / level_avg.replace(0, np.nan)
        df["salary_vs_level_avg"] = df["salary_vs_level_avg"].fillna(1.0)
        logger.debug("salary_vs_level_avg: mean=%.3f", df["salary_vs_level_avg"].mean())
        return df

    def salary_equity_ratio(self) -> pd.DataFrame:
        """Salary / expected salary based on level and department.

        Uses group-level medians as a proxy for expected salary.
        Values < 1.0 indicate potential pay inequity.

        SQL Equivalent:
            SELECT *,
                   salary / PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)
                       OVER (PARTITION BY department, job_level) AS salary_equity_ratio
            FROM employees

        Returns:
            DataFrame with 'salary_equity_ratio' column added.
        """
        df = self.df.copy()
        expected = df.groupby(["department", "job_level"])["salary"].transform("median")
        df["salary_equity_ratio"] = df["salary"] / expected.replace(0, np.nan)
        df["salary_equity_ratio"] = df["salary_equity_ratio"].fillna(1.0)
        logger.debug("salary_equity_ratio: mean=%.3f", df["salary_equity_ratio"].mean())
        return df

    def satisfaction_gap(self) -> pd.DataFrame:
        """Satisfaction score vs department average.

        Negative = below average satisfaction in department.
        Positive = above average.

        SQL Equivalent:
            SELECT *,
                   satisfaction_score -
                       AVG(satisfaction_score) OVER (PARTITION BY department) AS satisfaction_gap
            FROM employees

        Returns:
            DataFrame with 'satisfaction_gap' column added.
        """
        df = self.df.copy()
        dept_mean = df.groupby("department")["satisfaction_score"].transform("mean")
        df["satisfaction_gap"] = df["satisfaction_score"] - dept_mean
        df["satisfaction_gap"] = df["satisfaction_gap"].fillna(0.0)
        logger.debug("satisfaction_gap: mean=%.3f", df["satisfaction_gap"].mean())
        return df

    def workload_score(self) -> pd.DataFrame:
        """Composite: rating * satisfaction / tenure (proxy for workload).

        Higher values suggest productive, satisfied employees relative to tenure.
        Lower values may indicate disengagement or burnout.

        SQL Equivalent:
            SELECT *,
                   (performance_rating_3 * satisfaction_score) /
                       NULLIF(tenure_months, 0) AS workload_score
            FROM employees

        Returns:
            DataFrame with 'workload_score' column added.
        """
        df = self.df.copy()
        numerator = df["performance_rating_3"] * df["satisfaction_score"]
        denominator = df["tenure_months"].replace(0, np.nan)
        df["workload_score"] = numerator / denominator
        df["workload_score"] = df["workload_score"].fillna(df["workload_score"].median())
        logger.debug("workload_score: mean=%.3f", df["workload_score"].mean())
        return df

    def attrition_risk_score(self) -> pd.DataFrame:
        """Composite risk: weighted sum of risk factors (0-100).

        Risk factors and weights:
            - Promotion gap > 24 months:     25 points
            - Manager change in last 2yr:    15 points
            - Salary below dept avg:         15 points
            - Below avg satisfaction:        15 points
            - Low recent rating (< 3):       15 points
            - Declining rating trend:        10 points
            - High rating volatility:        5 points

        Scores are clipped to [0, 100].

        SQL Equivalent:
            SELECT *,
                   (CASE WHEN promotion_gap_months > 24 THEN 25 ELSE 0 END +
                    CASE WHEN manager_changes_2yr > 0 THEN 15 ELSE 0 END +
                    CASE WHEN salary_vs_dept_avg < 1.0 THEN 15 ELSE 0 END +
                    CASE WHEN satisfaction_gap < 0 THEN 15 ELSE 0 END +
                    CASE WHEN recent_rating < 3 THEN 15 ELSE 0 END +
                    CASE WHEN rating_trend < 0 THEN 10 ELSE 0 END +
                    CASE WHEN rating_volatility > 0.5 THEN 5 ELSE 0 END
                   ) AS attrition_risk_score
            FROM employees

        Returns:
            DataFrame with 'attrition_risk_score' column added (0-100 scale).
        """
        df = self.df.copy()

        # Ensure prerequisite features exist
        required_cols = [
            "promotion_gap_months", "manager_changes_2yr", "salary_vs_dept_avg",
            "satisfaction_gap", "recent_rating", "rating_trend", "rating_volatility",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning("Missing prerequisite columns for risk score: %s. Building them.", missing)
            builder = AttritionFeatureEngineer(df)
            for method_name in [
                "promotion_gap", "manager_change_count", "salary_vs_dept_avg",
                "satisfaction_gap", "recent_rating", "rating_trend", "rating_volatility",
            ]:
                built = getattr(builder, method_name)()
                # Update builder's internal df so subsequent calls use the new columns
                builder.df = built
                df = built.copy()

        # Compute each risk component (1 if risk present, 0 otherwise)
        risk_promo = (df["promotion_gap_months"] > 24).astype(float)
        risk_mgr = (df["manager_changes_2yr"] > 0).astype(float)
        risk_salary = (df["salary_vs_dept_avg"] < 1.0).astype(float)
        risk_sat = (df["satisfaction_gap"] < 0).astype(float)
        risk_rating = (df["recent_rating"] < 3).astype(float)
        risk_trend = (df["rating_trend"] < 0).astype(float)
        risk_vol = (df["rating_volatility"] > 0.5).astype(float)

        # Weighted composite
        raw_score = (
            25.0 * risk_promo
            + 15.0 * risk_mgr
            + 15.0 * risk_salary
            + 15.0 * risk_sat
            + 15.0 * risk_rating
            + 10.0 * risk_trend
            + 5.0 * risk_vol
        )

        df["attrition_risk_score"] = np.clip(raw_score, 0.0, 100.0)

        # Handle NaNs from missing underlying data
        df["attrition_risk_score"] = df["attrition_risk_score"].fillna(0.0)

        mean_score = df["attrition_risk_score"].mean()
        high_risk = (df["attrition_risk_score"] >= 50).sum()
        logger.info(
            "attrition_risk_score: mean=%.1f, high_risk(>=50)=%d/%d",
            mean_score, high_risk, len(df),
        )
        return df

    def build_all_features(self) -> pd.DataFrame:
        """Build all features and return enriched DataFrame.

        Applies each feature method sequentially, accumulating columns.

        Returns:
            DataFrame with all engineered features added.
        """
        methods = [
            self.recent_rating,
            self.rating_trend,
            self.avg_rating,
            self.rating_volatility,
            self.promotion_gap,
            self.promotion_gap_flag,
            self.manager_change_count,
            self.manager_change_flag,
            self.salary_vs_dept_avg,
            self.salary_vs_level_avg,
            self.salary_equity_ratio,
            self.satisfaction_gap,
            self.workload_score,
            self.attrition_risk_score,
        ]

        df = self.df.copy()
        builder = AttritionFeatureEngineer(df)
        for method in methods:
            result = method.__get__(builder, AttritionFeatureEngineer)()
            builder.df = result

        final = builder.df
        logger.info(
            "build_all_features complete: %d columns, %d new features",
            len(final.columns),
            len(final.columns) - len(self.df.columns),
        )
        return final

    def feature_summary(self) -> pd.DataFrame:
        """Summary of all engineered features.

        Returns:
            DataFrame with columns: feature_name, dtype, null_count,
            null_pct, mean, std, min, max.
        """
        df = self.build_all_features()
        original_cols = set(self.df.columns)
        new_cols = [c for c in df.columns if c not in original_cols]

        summary_rows = []
        for col in new_cols:
            series = df[col]
            summary_rows.append({
                "feature_name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_pct": round(series.isna().mean() * 100, 2),
                "mean": round(series.mean(), 4) if pd.api.types.is_numeric_dtype(series) else None,
                "std": round(series.std(), 4) if pd.api.types.is_numeric_dtype(series) else None,
                "min": series.min() if pd.api.types.is_numeric_dtype(series) else None,
                "max": series.max() if pd.api.types.is_numeric_dtype(series) else None,
            })

        summary = pd.DataFrame(summary_rows)
        logger.info("feature_summary: %d features engineered", len(summary))
        return summary
