"""Incremental retraining pipeline using synthetic and feedback data."""

from __future__ import annotations

from typing import Any

from app.ml.synthetic.generator import load_synthetic_training_samples, save_generated_samples
from app.ml.training.load_local_datasets import load_unified_dataset
from app.ml.training.model_backends import (
    dump_json,
    evaluate_model,
    load_or_bootstrap_model,
)
from app.ml.training.train_base_model import train_base_model
from app.services.feedback_service import FeedbackService
from app.utils.config import Settings, ensure_runtime_dirs


def run_incremental_retraining(
    *,
    settings: Settings | None = None,
    synthetic_count: int = 64,
    max_base_rows_per_file: int = 20,
) -> dict[str, Any]:
    """Fine-tune the saved model using synthetic data and analyst feedback."""

    effective = ensure_runtime_dirs(settings)
    if not effective.model_artifact_path.exists():
        train_base_model(settings=effective, max_rows_per_file=max_base_rows_per_file)

    model = load_or_bootstrap_model(effective.model_artifact_path, settings=effective)
    before_snapshot = model.snapshot_weights()

    base_dataset = load_unified_dataset(effective.dataset_dir, max_rows_per_file=max_base_rows_per_file)
    evaluation_samples = list(zip(base_dataset.texts, base_dataset.labels, strict=False))

    feedback_service = FeedbackService(effective.feedback_db_path)
    feedback_samples = feedback_service.get_feedback_samples(limit=synthetic_count)
    synthetic_samples = load_synthetic_training_samples(effective, count=synthetic_count)
    save_generated_samples(settings=effective, count=synthetic_count)

    before_metrics = evaluate_model(model, evaluation_samples)
    incremental_samples = synthetic_samples + feedback_samples
    model.partial_fit(incremental_samples)
    after_metrics = evaluate_model(model, evaluation_samples)
    after_snapshot = model.snapshot_weights()
    model.save(effective.model_artifact_path)

    changed_weight_keys = sorted(
        key for key in set(before_snapshot) | set(after_snapshot) if before_snapshot.get(key) != after_snapshot.get(key)
    )
    metrics = {
        "before": before_metrics,
        "after": after_metrics,
        "synthetic_samples": len(synthetic_samples),
        "feedback_samples": len(feedback_samples),
        "changed_weight_keys": changed_weight_keys,
        "model_path": str(effective.model_artifact_path),
    }
    dump_json(effective.retraining_metrics_path, metrics)
    return metrics


if __name__ == "__main__":
    print(run_incremental_retraining())
