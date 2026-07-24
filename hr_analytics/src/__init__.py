"""HR Analytics source modules."""

from hr_analytics.src.generate import generate_synthetic_data
from hr_analytics.src.preprocess import preprocess_data
from hr_analytics.src.features import engineer_features
from hr_analytics.src.analytics import run_analytics, compute_equity_analysis
from hr_analytics.src.model import train_attrition_model

__all__ = [
    "generate_synthetic_data",
    "preprocess_data",
    "engineer_features",
    "run_analytics",
    "compute_equity_analysis",
    "train_attrition_model",
]
