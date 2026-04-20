"""CLI entry point for base training."""

from __future__ import annotations

import argparse
import json

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
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable preprocessing cache reuse.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the preprocessing cache even if one already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training progress logging.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use during dataset preprocessing.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Number of samples to score per validation/test batch.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = train_base_model(
            max_rows_per_file=args.max_rows_per_file,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            force_backend=args.backend,
            use_preprocessing_cache=not args.no_cache,
            rebuild_preprocessing_cache=args.rebuild_cache,
            log_progress=not args.quiet,
            workers=args.workers,
            evaluation_batch_size=args.eval_batch_size,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
