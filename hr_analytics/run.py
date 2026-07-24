"""Command-line interface for HR People Analytics."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from hr_analytics.src.pipeline import (
    run_analytics_only,
    run_generate_only,
    run_model_only,
    run_pipeline,
)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE, stream=sys.stderr)
    # Quiet noisy libraries
    for name in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hr-analytics",
        description="HR People Analytics & Attrition Prediction CLI",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "-c", "--config",
        default="hr_analytics/configs/default.yaml",
        help="Path to YAML configuration file (default: hr_analytics/configs/default.yaml).",
    )

    sub = parser.add_subparsers(dest="command", help="Available sub-commands")

    # ---- full ----
    p_full = sub.add_parser("full", help="Run the end-to-end pipeline.")
    p_full.add_argument("-n", "--n-employees", type=int, default=None, help="Number of employees to generate.")
    p_full.add_argument("-s", "--seed", type=int, default=None, help="Random seed.")
    p_full.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory.")
    p_full.add_argument("-d", "--data-path", type=str, default=None, help="Existing CSV to analyse (skips generation).")

    # ---- generate ----
    p_gen = sub.add_parser("generate", help="Generate synthetic HR data only.")
    p_gen.add_argument("-n", "--n-employees", type=int, default=10000, help="Number of employees.")
    p_gen.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")
    p_gen.add_argument("-o", "--output-dir", type=str, default="data", help="Directory to write CSV.")

    # ---- analyze ----
    p_ana = sub.add_parser("analyze", help="Run analytics on existing data.")
    p_ana.add_argument("data_path", help="Path to CSV file.")
    p_ana.add_argument("-o", "--output-dir", type=str, default="outputs", help="Output directory.")

    # ---- train ----
    p_mod = sub.add_parser("train", help="Train the attrition model.")
    p_mod.add_argument("data_path", help="Path to CSV file.")
    p_mod.add_argument("-o", "--output-dir", type=str, default="outputs", help="Output directory.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry-point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(verbose=getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "full":
            result = run_pipeline(
                config_path=args.config,
                n_employees=args.n_employees,
                seed=args.seed,
                output_dir=args.output_dir,
                data_path=args.data_path,
            )
            print(f"Pipeline complete. Outputs in: {result['output_dir']}")

        elif args.command == "generate":
            df = run_generate_only(
                n_employees=args.n_employees,
                seed=args.seed,
                output_dir=args.output_dir,
                config_path=args.config,
            )
            print(f"Generated {len(df)} employee records.")

        elif args.command == "analyze":
            result = run_analytics_only(
                data_path=args.data_path,
                output_dir=args.output_dir,
            )
            print("Analytics complete.")
            for key in result:
                print(f"  {key}: {type(result[key]).__name__}")

        elif args.command == "train":
            result = run_model_only(
                data_path=args.data_path,
                output_dir=args.output_dir,
                config_path=args.config,
            )
            metrics = result.get("model_metrics", {})
            print(f"Model trained — accuracy: {metrics.get('accuracy', 'N/A')}, "
                  f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")

        else:
            parser.print_help()
            return 1

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logging.exception("Unhandled exception")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
