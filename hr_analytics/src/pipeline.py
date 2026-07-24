"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hr_analytics.src.analytics import compute_equity_analysis, run_analytics
from hr_analytics.src.features import engineer_features
from hr_analytics.src.generate import generate_synthetic_data
from hr_analytics.src.model import train_attrition_model
from hr_analytics.src.preprocess import preprocess_data

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the HR analytics pipeline.

    Attributes:
        config_path: Path to a YAML configuration file.
        n_employees: Override for number of employees to generate.
        seed: Random seed override.
        output_dir: Directory for all output artefacts.
        data_path: Path to an existing CSV (skips generation when set).
    """

    config_path: str = "hr_analytics/configs/default.yaml"
    n_employees: int | None = None
    seed: int | None = None
    output_dir: str | None = None
    data_path: str | None = None

    def load_yaml(self) -> dict[str, Any]:
        """Load and return the YAML configuration."""
        path = Path(self.config_path)
        if not path.exists():
            logger.warning("Config file not found at %s — using defaults", path)
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info("Loaded config from %s", path)
        return cfg

    def resolve_output_dir(self, cfg: dict[str, Any]) -> Path:
        """Determine the output directory from config, attribute, or default."""
        out = self.output_dir or cfg.get("output", {}).get("output_dir", "outputs")
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path


def _save_json(data: Any, path: Path) -> None:
    """Serialise *data* to JSON at *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved JSON → %s", path)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to CSV at *path*."""
    df.to_csv(path, index=False)
    logger.info("Saved CSV → %s (%d rows, %d cols)", path, len(df), df.shape[1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str = "hr_analytics/configs/default.yaml",
    n_employees: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    data_path: str | None = None,
) -> dict[str, Any]:
    """Execute the full analytics pipeline.

    Steps:
        1. Load configuration
        2. Generate or load data
        3. Preprocess
        4. Engineer features
        5. Run analytics & equity analysis
        6. Train model
        7. Persist outputs

    Args:
        config_path: Path to YAML config file.
        n_employees: Override for employee count.
        seed: Override for random seed.
        output_dir: Override for output directory.
        data_path: Path to an existing CSV to skip generation.

    Returns:
        Dictionary with keys ``config``, ``analytics``, ``equity``,
        ``model_metrics``, ``output_path``.
    """
    pipeline_cfg = PipelineConfig(
        config_path=config_path,
        n_employees=n_employees,
        seed=seed,
        output_dir=output_dir,
        data_path=data_path,
    )
    cfg = pipeline_cfg.load_yaml()
    out_dir = pipeline_cfg.resolve_output_dir(cfg)

    logger.info("=== HR Analytics Pipeline — start ===")

    # ---- Step 1: Data ----
    data_cfg = cfg.get("data", {})
    if pipeline_cfg.data_path and Path(pipeline_cfg.data_path).exists():
        logger.info("Loading existing data from %s", pipeline_cfg.data_path)
        raw_df = pd.read_csv(pipeline_cfg.data_path)
    else:
        n = pipeline_cfg.n_employees or data_cfg.get("n_employees", 10000)
        s = pipeline_cfg.seed or data_cfg.get("seed", 42)
        raw_df = generate_synthetic_data(n_employees=n, seed=s, config=cfg)

    # ---- Step 2: Preprocess ----
    processed_df = preprocess_data(raw_df)

    # ---- Step 3: Feature engineering ----
    enriched_df = engineer_features(processed_df, config=cfg.get("feature_engineering", {}))

    # ---- Step 4: Analytics ----
    analytics_summary = run_analytics(enriched_df)
    equity_summary = compute_equity_analysis(enriched_df)

    # ---- Step 5: Model ----
    model_metrics = train_attrition_model(enriched_df, config=cfg.get("model", {}))

    # ---- Step 6: Save outputs ----
    csv_path = out_dir / "hr_analytics_enriched.csv"
    _save_csv(enriched_df, csv_path)

    analytics_path = out_dir / "analytics_summary.json"
    _save_json(analytics_summary, analytics_path)

    equity_path = out_dir / "equity_analysis.json"
    _save_json(equity_summary, equity_path)

    model_path = out_dir / "model_metrics.json"
    _save_json(model_metrics, model_path)

    logger.info("=== HR Analytics Pipeline — complete ===")

    return {
        "config": cfg,
        "analytics": analytics_summary,
        "equity": equity_summary,
        "model_metrics": model_metrics,
        "output_dir": str(out_dir),
    }


def run_generate_only(
    n_employees: int = 10000,
    seed: int = 42,
    output_dir: str = "data",
    config_path: str = "hr_analytics/configs/default.yaml",
) -> pd.DataFrame:
    """Generate synthetic data without further analysis."""
    cfg_path = Path(config_path)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    df = generate_synthetic_data(n_employees=n_employees, seed=seed, config=cfg)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "hr_data.csv"
    _save_csv(df, csv_path)
    return df


def run_analytics_only(
    data_path: str,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """Run analytics on an existing CSV file."""
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    df = engineer_features(df)

    analytics_summary = run_analytics(df)
    equity_summary = compute_equity_analysis(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_json(analytics_summary, out / "analytics_summary.json")
    _save_json(equity_summary, out / "equity_analysis.json")

    return {"analytics": analytics_summary, "equity": equity_summary}


def run_model_only(
    data_path: str,
    output_dir: str = "outputs",
    config_path: str = "hr_analytics/configs/default.yaml",
) -> dict[str, Any]:
    """Train the attrition model on an existing CSV file."""
    cfg_path = Path(config_path)
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    df = engineer_features(df, config=cfg.get("feature_engineering", {}))

    model_metrics = train_attrition_model(df, config=cfg.get("model", {}))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_json(model_metrics, out / "model_metrics.json")

    return {"model_metrics": model_metrics}
