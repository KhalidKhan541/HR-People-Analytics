from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalaryEquityAnalyzer:
    """Analyze salary equity by gender and level."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate()

    def _validate(self) -> None:
        required = {"Gender", "JobLevel", "Salary"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def gender_pay_gap(self) -> pd.DataFrame:
        """Overall gender pay gap: mean salary by gender, gap % relative to highest."""
        agg = (
            self.df.groupby("Gender")["Salary"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )
        agg.columns = ["Gender", "MeanSalary", "MedianSalary", "StdSalary", "Count"]

        max_mean = agg["MeanSalary"].max()
        agg["GapPercent"] = ((max_mean - agg["MeanSalary"]) / max_mean * 100).round(2)

        return agg

    def gender_pay_gap_by_level(self) -> pd.DataFrame:
        """Gender pay gap at each job level."""
        agg = (
            self.df.groupby(["JobLevel", "Gender"])["Salary"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        agg.columns = ["JobLevel", "Gender", "MeanSalary", "MedianSalary", "Count"]

        pivoted = agg.pivot_table(
            index="JobLevel",
            columns="Gender",
            values=["MeanSalary", "MedianSalary", "Count"],
            aggfunc="first",
        )
        pivoted.columns = ["_".join(col) for col in pivoted.columns]
        pivoted = pivoted.reset_index()

        genders = [g for g in ["Male", "Female"] if f"MeanSalary_{g}" in pivoted.columns]
        if len(genders) == 2:
            high = pivoted[f"MeanSalary_{genders[0]}"].combine(
                pivoted[f"MeanSalary_{genders[1]}"], max
            )
            pivoted["GapPercent"] = (
                ((high - pivoted[f"MeanSalary_{genders[0]}"]) / high * 100).round(2)
            )
        else:
            pivoted["GapPercent"] = np.nan

        return pivoted

    def gender_pay_gap_by_department(self) -> pd.DataFrame:
        """Gender pay gap by department."""
        if "Department" not in self.df.columns:
            logger.warning("Department column not found; returning empty DataFrame.")
            return pd.DataFrame()

        agg = (
            self.df.groupby(["Department", "Gender"])["Salary"]
            .agg(["mean", "count"])
            .reset_index()
        )
        agg.columns = ["Department", "Gender", "MeanSalary", "Count"]

        pivoted = agg.pivot_table(
            index="Department",
            columns="Gender",
            values=["MeanSalary", "Count"],
            aggfunc="first",
        )
        pivoted.columns = ["_".join(col) for col in pivoted.columns]
        pivoted = pivoted.reset_index()

        genders = [g for g in ["Male", "Female"] if f"MeanSalary_{g}" in pivoted.columns]
        if len(genders) == 2:
            high = pivoted[f"MeanSalary_{genders[0]}"].combine(
                pivoted[f"MeanSalary_{genders[1]}"], max
            )
            pivoted["GapPercent"] = (
                ((high - pivoted[f"MeanSalary_{genders[0]}"]) / high * 100).round(2)
            )
        else:
            pivoted["GapPercent"] = np.nan

        return pivoted

    def salary_distribution(self) -> pd.DataFrame:
        """Salary distribution statistics: mean, median, std, min, max by gender and level."""
        agg = (
            self.df.groupby(["Gender", "JobLevel"])["Salary"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )
        agg.columns = [
            "Gender",
            "JobLevel",
            "MeanSalary",
            "MedianSalary",
            "StdSalary",
            "MinSalary",
            "MaxSalary",
            "Count",
        ]
        return agg

    def compensation_equity_score(self) -> pd.DataFrame:
        """Equity score: actual salary / expected salary (from level+dept model).
        Score < 0.95 = underpaid, > 1.05 = overpaid.
        """
        df = self.df.copy()

        dummy_cols: List[str] = []
        for col in ["JobLevel", "Department"]:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                dummy_cols.extend(dummies.columns.tolist())

        if not dummy_cols:
            logger.warning("No categorical columns found for expected salary model.")
            df["EquityScore"] = np.nan
            df["Status"] = "unknown"
            return df[["Gender", "JobLevel", "Salary", "EquityScore", "Status"]]

        X = df[dummy_cols].astype(float).values
        y = df["Salary"].values

        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.warning("OLS solve failed; returning NaN equity scores.")
            df["EquityScore"] = np.nan
            df["Status"] = "unknown"
            return df[["Gender", "JobLevel", "Salary", "EquityScore", "Status"]]

        expected = X_aug @ beta
        df["EquityScore"] = (df["Salary"] / expected).round(4)
        df["Status"] = pd.cut(
            df["EquityScore"],
            bins=[-np.inf, 0.95, 1.05, np.inf],
            labels=["underpaid", "fair", "overpaid"],
        )

        return df[["Gender", "JobLevel", "Salary", "EquityScore", "Status"]]

    def representation_analysis(self) -> pd.DataFrame:
        """Gender representation by level. Expected vs actual (chi-squared test)."""
        ct = pd.crosstab(self.df["Gender"], self.df["JobLevel"])

        if ct.shape[0] < 2 or ct.shape[1] < 2:
            logger.warning("Insufficient data for chi-squared test.")
            ct["Chi2Stat"] = np.nan
            ct["PValue"] = np.nan
            return ct

        chi2, p, dof, expected = stats.chi2_contingency(ct)

        expected_df = pd.DataFrame(
            expected, index=ct.index, columns=ct.columns
        )
        expected_df.columns = [f"Expected_{c}" for c in expected_df.columns]

        result = pd.concat([ct, expected_df], axis=1)
        result["Chi2Stat"] = round(chi2, 4)
        result["PValue"] = round(p, 4)
        result["Significant"] = result["PValue"] < 0.05

        return result

    def promotion_rate_by_gender(self) -> pd.DataFrame:
        """Promotion rate by gender (if promotion data exists)."""
        promo_col = None
        for col in ["Promoted", "Promotion", "LastPromotion", "YearsSincePromotion"]:
            if col in self.df.columns:
                promo_col = col
                break

        if promo_col is None:
            logger.info("No promotion column found; returning empty DataFrame.")
            return pd.DataFrame()

        df = self.df.copy()
        if promo_col in ("Promoted", "Promotion"):
            df["PromotedBinary"] = df[promo_col].astype(int)
        else:
            df["PromotedBinary"] = (df[promo_col] <= 2).astype(int)

        agg = (
            df.groupby("Gender")["PromotedBinary"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        )
        agg.columns = ["Gender", "PromotionRate", "PromotedCount", "TotalCount"]
        agg["PromotionRate"] = (agg["PromotionRate"] * 100).round(2)

        if len(agg) == 2:
            rates = agg["PromotionRate"].values
            n1, n2 = agg["TotalCount"].values
            p1, p2 = rates / 100
            p_pooled = (agg["PromotedCount"].sum()) / (agg["TotalCount"].sum())
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
            if se > 0:
                z = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                z, p_value = 0.0, 1.0
            agg["ZStat"] = round(z, 4)
            agg["PValue"] = round(p_value, 4)

        return agg

    def equity_report(self) -> Dict[str, pd.DataFrame]:
        """Full equity analysis report."""
        report: Dict[str, pd.DataFrame] = {
            "gender_pay_gap_overall": self.gender_pay_gap(),
            "gender_pay_gap_by_level": self.gender_pay_gap_by_level(),
            "salary_distribution": self.salary_distribution(),
            "equity_scores": self.compensation_equity_score(),
            "representation": self.representation_analysis(),
            "promotion_rate": self.promotion_rate_by_gender(),
        }

        dept_gap = self.gender_pay_gap_by_department()
        if not dept_gap.empty:
            report["gender_pay_gap_by_department"] = dept_gap

        logger.info("Equity report generated with %d sections.", len(report))
        return report

    def generate_insights(self) -> List[str]:
        """Auto-generate key findings (e.g., 'Female engineers earn 8% less than male engineers')."""
        insights: List[str] = []

        overall = self.gender_pay_gap()
        male = overall[overall["Gender"] == "Male"]
        female = overall[overall["Gender"] == "Female"]

        if not male.empty and not female.empty:
            m_sal = male["MeanSalary"].iloc[0]
            f_sal = female["MeanSalary"].iloc[0]
            if m_sal > 0:
                gap = ((m_sal - f_sal) / m_sal) * 100
                higher, lower = ("Male", "Female") if m_sal > f_sal else ("Female", "Male")
                if abs(gap) > 0.5:
                    insights.append(
                        f"{higher} employees earn {abs(gap):.1f}% more than {lower} employees on average."
                    )

        by_level = self.gender_pay_gap_by_level()
        if "GapPercent" in by_level.columns:
            for _, row in by_level.iterrows():
                gp = row.get("GapPercent")
                if pd.notna(gp) and abs(gp) > 1.0:
                    insights.append(
                        f"Level {row['JobLevel']}: {gp:.1f}% pay gap between genders."
                    )

        equity = self.compensation_equity_score()
        if not equity.empty and "Status" in equity.columns:
            status_counts = equity.groupby(["Gender", "Status"]).size().unstack(fill_value=0)
            for gender in status_counts.index:
                if "underpaid" in status_counts.columns:
                    under = status_counts.loc[gender, "underpaid"]
                    total = status_counts.loc[gender].sum()
                    if total > 0 and under / total > 0.1:
                        insights.append(
                            f"{under} of {total} {gender.lower()} employees are underpaid (equity score < 0.95)."
                        )

        promo = self.promotion_rate_by_gender()
        if not promo.empty and "PromotionRate" in promo.columns and len(promo) >= 2:
            rates = dict(zip(promo["Gender"], promo["PromotionRate"]))
            if len(rates) == 2:
                g1, g2 = list(rates.keys())
                diff = rates[g1] - rates[g2]
                if abs(diff) > 2.0:
                    higher_g = g1 if diff > 0 else g2
                    insights.append(
                        f"{higher_g} employees have a {abs(diff):.1f}pp higher promotion rate."
                    )

        rep = self.representation_analysis()
        if "PValue" in rep.columns:
            p = rep["PValue"].iloc[0]
            if pd.notna(p) and p < 0.05:
                insights.append(
                    "Gender representation across levels is statistically uneven (p < 0.05)."
                )

        if not insights:
            insights.append("No significant equity issues detected in the current dataset.")

        return insights
