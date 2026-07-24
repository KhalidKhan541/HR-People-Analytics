import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HRPreprocessor:
    """Clean and preprocess HR data.

    Supports fluent chaining::

        df = (
            HRPreprocessor(raw_df)
            .clean()
            .add_tenure_band()
            .add_salary_band()
            .add_rating_features()
            .add_tenure_at_company()
            .add_months_since_promotion()
            .to_df()
        )
    """

    TENURE_BANDS = ["0-1", "1-3", "3-5", "5-10", "10+"]
    TENURE_EDGES = [0, 1, 3, 5, 10, float("inf")]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info("HRPreprocessor initialized with %d rows", len(self.df))

    def to_df(self) -> pd.DataFrame:
        """Return the processed DataFrame."""
        return self.df

    def clean(self) -> "HRPreprocessor":
        """Clean data: handle nulls, fix types, dedup."""
        initial_len = len(self.df)

        self.df = self.df.drop_duplicates()
        if len(self.df) < initial_len:
            logger.info(
                "Removed %d duplicate rows", initial_len - len(self.df)
            )

        self.df.columns = (
            self.df.columns.str.strip().str.lower().str.replace(" ", "_")
        )

        datetime_cols = [
            "hire_date",
            "term_date",
            "last_promotion_date",
            "last_survey_date",
        ]
        for col in datetime_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        numeric_cols = [
            "salary",
            "tenure_years",
            "rating_1",
            "rating_2",
            "rating_3",
            "manager_changes",
            "satisfaction_score",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "is_active" in self.df.columns:
            self.df["is_active"] = self.df["is_active"].astype(bool)

        if "salary" in self.df.columns and "level" in self.df.columns:
            salary_median = self.df.groupby("level")["salary"].transform("median")
            null_salaries = self.df["salary"].isna()
            if null_salaries.any():
                self.df.loc[null_salaries, "salary"] = salary_median[null_salaries]
                logger.info(
                    "Filled %d null salaries with level median",
                    null_salaries.sum(),
                )

        if "satisfaction_score" in self.df.columns:
            median_sat = self.df["satisfaction_score"].median()
            self.df["satisfaction_score"] = self.df["satisfaction_score"].fillna(
                median_sat
            )

        if (
            "hire_date" in self.df.columns
            and "tenure_years" not in self.df.columns
        ):
            ref = pd.Timestamp("2025-01-01")
            self.df["tenure_years"] = (
                (ref - self.df["hire_date"]).dt.days / 365.25
            ).round(2)

        logger.info("Cleaning complete. Final shape: %s", self.df.shape)
        return self

    def add_tenure_band(self) -> "HRPreprocessor":
        """Add tenure bands: 0-1, 1-3, 3-5, 5-10, 10+ years."""
        if "tenure_years" not in self.df.columns:
            logger.warning(
                "tenure_years column not found; skipping add_tenure_band"
            )
            return self

        self.df["tenure_band"] = pd.cut(
            self.df["tenure_years"],
            bins=self.TENURE_EDGES,
            labels=self.TENURE_BANDS,
            right=True,
            include_lowest=True,
        )
        logger.info(
            "Added tenure_band column. Distribution:\n%s",
            self.df["tenure_band"].value_counts().to_string(),
        )
        return self

    def add_salary_band(self) -> "HRPreprocessor":
        """Add salary bands by percentile."""
        if "salary" not in self.df.columns:
            logger.warning(
                "salary column not found; skipping add_salary_band"
            )
            return self

        self.df["salary_band"] = pd.qcut(
            self.df["salary"],
            q=5,
            labels=["P0-20", "P20-40", "P40-60", "P60-80", "P80-100"],
            duplicates="drop",
        )
        logger.info(
            "Added salary_band column. Distribution:\n%s",
            self.df["salary_band"].value_counts().to_string(),
        )
        return self

    def add_rating_features(self) -> "HRPreprocessor":
        """Add average rating, rating trend, rating volatility."""
        rating_cols = [
            c
            for c in ["rating_1", "rating_2", "rating_3"]
            if c in self.df.columns
        ]

        if not rating_cols:
            logger.warning(
                "No rating columns found; skipping add_rating_features"
            )
            return self

        self.df["avg_rating"] = self.df[rating_cols].mean(axis=1).round(2)

        if len(rating_cols) >= 2:
            self.df["rating_trend"] = (
                self.df[rating_cols[-1]] - self.df[rating_cols[0]]
            )
        else:
            self.df["rating_trend"] = 0

        if len(rating_cols) >= 2:
            self.df["rating_volatility"] = (
                self.df[rating_cols].std(axis=1).round(2)
            )
        else:
            self.df["rating_volatility"] = 0.0

        logger.info(
            "Added rating features: avg_rating, rating_trend, rating_volatility. "
            "Mean avg_rating: %.2f",
            self.df["avg_rating"].mean(),
        )
        return self

    def add_tenure_at_company(self) -> "HRPreprocessor":
        """Calculate tenure in years and months."""
        if "hire_date" not in self.df.columns:
            logger.warning(
                "hire_date column not found; skipping add_tenure_at_company"
            )
            return self

        ref = pd.Timestamp("2025-01-01")
        ref_col = np.where(
            self.df["is_active"], ref, self.df["term_date"].fillna(ref)
        )
        ref_series = pd.to_datetime(ref_col)
        delta = ref_series - pd.to_datetime(self.df["hire_date"])

        self.df["tenure_years"] = (delta.dt.days / 365.25).round(2)
        self.df["tenure_months"] = (delta.dt.days // 30.44).astype(int)

        logger.info(
            "Added tenure_years and tenure_months. Mean tenure: %.2f years",
            self.df["tenure_years"].mean(),
        )
        return self

    def add_months_since_promotion(self) -> "HRPreprocessor":
        """Calculate months since last promotion."""
        if "last_promotion_date" not in self.df.columns:
            logger.warning(
                "last_promotion_date column not found; skipping add_months_since_promotion"
            )
            return self

        ref = pd.Timestamp("2025-01-01")
        delta = ref - pd.to_datetime(self.df["last_promotion_date"])
        self.df["months_since_promotion"] = (delta.dt.days // 30.44).astype(
            int
        )

        logger.info(
            "Added months_since_promotion. Mean: %.1f months",
            self.df["months_since_promotion"].mean(),
        )
        return self
