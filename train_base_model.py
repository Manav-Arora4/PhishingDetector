"""CLI entry point for base training."""

from __future__ import annotations

import argparse

from app.ml.training.train_base_model import train_base_model


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for base training."""

    parser = argparse.ArgumentParser(description="Train the phishing detector on local datasets.")
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Limit the number of rows loaded from each CSV file.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Portion of normalized samples used for training.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Portion of normalized samples used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Portion of normalized samples used for held-out testing.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Force a specific backend, such as token_naive_bayes or transformer.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(
        train_base_model(
            max_rows_per_file=args.max_rows_per_file,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            force_backend=args.backend,
        )
    )
