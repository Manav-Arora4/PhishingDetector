"""Base training pipeline for phishing detection."""

from __future__ import annotations

import json
from typing import Any

from app.ml.training.load_local_datasets import UnifiedDataset, load_unified_dataset
from app.ml.training.model_backends import (
    dump_json,
    evaluate_model,
    select_training_backend,
    train_validation_test_split,
)
from app.utils.config import Settings, ensure_runtime_dirs


def train_base_model(
    *,
    settings: Settings | None = None,
    max_rows_per_file: int | None = None,
    force_backend: str | None = None,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train the base phishing detector using local datasets only."""

    effective = ensure_runtime_dirs(settings)
    dataset: UnifiedDataset = load_unified_dataset(
        effective.dataset_dir,
        max_rows_per_file=max_rows_per_file,
    )
    records = dataset.records
    samples = [(record.text, record.label) for record in records]
    if len(samples) < 3:
        raise RuntimeError("At least two normalized samples are required for training.")

    indexed_samples = list(enumerate(samples))
    train_indexed, validation_indexed, test_indexed = train_validation_test_split(
        indexed_samples,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=random_seed,
    )
    train_indices = [index for index, _sample in train_indexed]
    validation_indices = [index for index, _sample in validation_indexed]
    test_indices = [index for index, _sample in test_indexed]
    train_samples = [samples[index] for index in train_indices]
    validation_samples = [samples[index] for index in validation_indices]
    test_samples = [samples[index] for index in test_indices]
    model = select_training_backend(effective, force_backend=force_backend)
    model.fit(train_samples)
    validation_metrics = evaluate_model(model, validation_samples)
    test_metrics = evaluate_model(model, test_samples)
    training_metrics = evaluate_model(model, train_samples[: min(len(train_samples), 128)])
    model.save(effective.model_artifact_path)

    metrics = {
        "backend": model.backend_name,
        "dataset": {
            "raw_rows": dataset.raw_rows,
            "normalized_rows": len(dataset.records),
            "files_loaded": len(dataset.text_columns),
            "text_columns": dataset.text_columns,
            "split_counts": {
                "train": len(train_samples),
                "validation": len(validation_samples),
                "test": len(test_samples),
            },
        },
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model_path": str(effective.model_artifact_path),
            "metrics_path": str(effective.base_metrics_path),
            "test_samples_path": str(effective.test_samples_path),
        },
    }
    dump_json(effective.base_metrics_path, metrics)
    with effective.test_samples_path.open("w", encoding="utf-8") as handle:
        for index in test_indices:
            record = records[index]
            payload = {
                "text": record.text,
                "label": record.label,
                "source_file": record.source_file,
                "subject": record.subject,
                "urls": record.urls,
            }
            handle.write(json.dumps(payload) + "\n")
    return metrics


if __name__ == "__main__":
    result = train_base_model()
    print(result)
