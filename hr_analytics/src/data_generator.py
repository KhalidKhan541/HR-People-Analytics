import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HRDataGenerator:
    """Generate realistic synthetic HR employee data."""

    DEPARTMENTS = [
        "Engineering",
        "Sales",
        "Marketing",
        "HR",
        "Finance",
        "Operations",
        "Product",
        "Legal",
    ]
    LEVELS = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director", "VP"]
    GENDERS = ["Male", "Female", "Non-binary"]

    # Weight distributions for realistic company structures
    LEVEL_WEIGHTS = [0.25, 0.28, 0.20, 0.12, 0.08, 0.05, 0.02]
    GENDER_WEIGHTS = [0.48, 0.48, 0.04]

    # Base salary ranges by level (min, max, mean, std)
    SALARY_PARAMS = {
        "Junior": {"min": 45000, "max": 80000, "mean": 62000, "std": 10000},
        "Mid": {"min": 70000, "max": 120000, "mean": 95000, "std": 13000},
        "Senior": {"min": 100000, "max": 160000, "mean": 128000, "std": 15000},
        "Lead": {"min": 120000, "max": 180000, "mean": 150000, "std": 15000},
        "Manager": {"min": 130000, "max": 190000, "mean": 158000, "std": 16000},
        "Director": {"min": 150000, "max": 220000, "mean": 182000, "std": 18000},
        "VP": {"min": 190000, "max": 250000, "mean": 220000, "std": 18000},
    }

    # Department salary multipliers
    DEPT_MULTIPLIERS = {
        "Engineering": 1.12,
        "Product": 1.08,
        "Finance": 1.04,
        "Legal": 1.02,
        "Marketing": 0.98,
        "Sales": 0.96,
        "Operations": 0.92,
        "HR": 0.90,
    }

    def __init__(self, n_employees: int = 10000, seed: int = 42):
        self.n_employees = n_employees
        self.rng = np.random.default_rng(seed)
        logger.info(
            "HRDataGenerator initialized with n_employees=%d, seed=%d",
            n_employees,
            seed,
        )

    def generate(self) -> pd.DataFrame:
        """Generate employees with columns:
        - employee_id: unique ID
        - hire_date: datetime (2015-2024)
        - term_date: datetime (null if still active)
        - department: Engineering, Sales, Marketing, HR, Finance, Operations, Product, Legal
        - level: Junior, Mid, Senior, Lead, Manager, Director, VP
        - salary: 45000-250000 (varies by level)
        - gender: Male, Female, Non-binary
        - tenure_years: calculated from hire_date
        - is_active: boolean
        - rating_1 through rating_3: last 3 performance ratings (1-5)
        - last_promotion_date: datetime
        - manager_id: reference to another employee
        - manager_changes: count of manager changes in last 2 years
        - satisfaction_score: 1-10
        - last_survey_date: datetime
        """
        logger.info("Generating %d employee records...", self.n_employees)

        df = pd.DataFrame()
        df["employee_id"] = np.arange(1, self.n_employees + 1)

        # Assign levels with realistic distribution
        df["level"] = self.rng.choice(
            self.LEVELS, size=self.n_employees, p=self.LEVEL_WEIGHTS
        )

        # Assign departments
        df["department"] = self.rng.choice(
            self.DEPARTMENTS, size=self.n_employees
        )

        # Assign genders
        df["gender"] = self.rng.choice(
            self.GENDERS, size=self.n_employees, p=self.GENDER_WEIGHTS
        )

        # Generate hire dates (2015-2024)
        start_date = datetime(2015, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range_days = (end_date - start_date).days
        df["hire_date"] = pd.to_datetime(start_date) + pd.to_timedelta(
            self.rng.integers(0, date_range_days, size=self.n_employees), unit="D"
        )

        # Generate attrition (~20% attrition rate)
        is_terminated = self.rng.random(self.n_employees) < 0.20
        df["is_active"] = ~is_terminated

        # Termination dates for terminated employees
        term_dates = pd.Series([pd.NaT] * self.n_employees, name="term_date")
        for idx in df.index[is_terminated]:
            hire = df.loc[idx, "hire_date"]
            min_term = hire + timedelta(days=30)
            max_term = datetime(2024, 12, 31)
            if min_term > max_term:
                term_dates.iloc[idx] = max_term
            else:
                term_dates.iloc[idx] = min_term + timedelta(
                    days=int(self.rng.integers(0, (max_term - min_term).days))
                )
        df["term_date"] = term_dates

        # Calculate tenure years
        reference_date = pd.to_datetime(datetime(2025, 1, 1))
        df["tenure_years"] = np.where(
            df["is_active"],
            (reference_date - df["hire_date"]).dt.days / 365.25,
            (df["term_date"] - df["hire_date"]).dt.days / 365.25,
        ).round(2)

        # Generate salaries based on level and department
        df["salary"] = df.apply(
            lambda row: self._generate_salary(row["level"], row["department"]),
            axis=1,
        )

        # Generate performance ratings (correlated with attrition)
        terminated_mask = ~df["is_active"]
        rating_base = np.where(
            terminated_mask, 2.8, 3.6  # Lower average ratings for terminated employees
        )
        df["rating_1"] = self._clamp_ratings(
            self.rng.normal(rating_base - 0.2, 0.9)
        )
        df["rating_2"] = self._clamp_ratings(
            self.rng.normal(rating_base - 0.1, 0.85)
        )
        df["rating_3"] = self._clamp_ratings(
            self.rng.normal(rating_base, 0.8)
        )

        # Generate last promotion date
        df["last_promotion_date"] = self._generate_promotion_dates(df)

        # Generate manager_id (reference to another employee)
        valid_managers = df["employee_id"].values
        df["manager_id"] = self.rng.choice(
            valid_managers, size=self.n_employees
        )
        # Ensure no self-references
        self_ref = df["manager_id"] == df["employee_id"]
        replacements = self.rng.choice(
            valid_managers, size=self_ref.sum()
        )
        df.loc[self_ref, "manager_id"] = replacements

        # Manager changes (correlated with attrition)
        df["manager_changes"] = self.rng.poisson(
            np.where(terminated_mask, 2.5, 0.8), size=self.n_employees
        )

        # Satisfaction score (correlated with attrition and ratings)
        satisfaction_base = np.where(
            terminated_mask, 5.5, 7.2
        )
        # Add rating correlation
        rating_bonus = (df["rating_3"].values - 3.0) * 0.5
        raw_sat = satisfaction_base + rating_bonus + self.rng.normal(0, 1.2, self.n_employees)
        df["satisfaction_score"] = np.clip(np.round(raw_sat), 1, 10).astype(int)

        # Last survey date
        df["last_survey_date"] = pd.to_datetime(
            reference_date - pd.to_timedelta(
                self.rng.integers(1, 365, size=self.n_employees), unit="D"
            )
        )

        # Round salary to nearest dollar
        df["salary"] = df["salary"].round(0).astype(int)

        logger.info("Generated %d records. Attrition rate: %.1f%%",
                     len(df), (1 - df["is_active"].mean()) * 100)

        return df

    def _generate_salary(self, level: str, department: str) -> float:
        """Salary based on level and department with realistic distributions."""
        params = self.SALARY_PARAMS[level]
        dept_mult = self.DEPT_MULTIPLIERS.get(department, 1.0)

        # Apply department multiplier to the mean
        adjusted_mean = params["mean"] * dept_mult
        salary = self.rng.normal(adjusted_mean, params["std"])

        # Clip to level bounds (after department adjustment)
        min_val = params["min"] * dept_mult
        max_val = params["max"] * dept_mult
        return float(np.clip(salary, min_val, max_val))

    def _clamp_ratings(self, raw: np.ndarray) -> np.ndarray:
        """Clamp ratings to integer 1-5 range."""
        return np.clip(np.round(raw), 1, 5).astype(int)

    def _generate_promotion_dates(self, df: pd.DataFrame) -> pd.Series:
        """Generate last promotion dates with realistic gaps."""
        promotion_dates = pd.Series([pd.NaT] * len(df), dtype="datetime64[ns]")

        for idx in df.index:
            hire = df.loc[idx, "hire_date"]
            term = df.loc[idx, "term_date"]
            ref = datetime(2025, 1, 1)

            # 15% never promoted
            if self.rng.random() < 0.15:
                promotion_dates.iloc[idx] = hire
                continue

            # Promotion gap: more senior = longer since last promotion on average
            level = df.loc[idx, "level"]
            gap_mean = {
                "Junior": 8,
                "Mid": 14,
                "Senior": 18,
                "Lead": 22,
                "Manager": 24,
                "Director": 30,
                "VP": 36,
            }.get(level, 12)

            months_gap = max(1, int(self.rng.exponential(gap_mean)))
            end = term if not pd.isna(term) else ref
            promo_date = end - pd.DateOffset(months=months_gap)
            promo_date = max(promo_date, hire)
            promotion_dates.iloc[idx] = promo_date

        return promotion_dates
