"""CLI entry point for incremental retraining."""

from app.ml.retraining.retrain_incremental import run_incremental_retraining


if __name__ == "__main__":
    print(run_incremental_retraining())
