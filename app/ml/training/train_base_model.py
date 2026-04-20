"""Base training pipeline for phishing detection."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from app.ml.training.load_local_datasets import (
    UnifiedDataset,
    build_preprocessing_cache_path,
    load_preprocessed_dataset,
    load_unified_dataset,
    save_preprocessed_dataset,
)
from app.ml.training.model_backends import (
    dump_json,
    evaluate_model,
    grouped_train_validation_test_split,
    select_training_backend,
)
from app.utils.config import Settings, ensure_runtime_dirs
from app.utils.text import SUSPICIOUS_KEYWORDS


def _label_distribution(records) -> dict[str, int]:
    counter = Counter(record.label for record in records)
    return {"benign": int(counter.get(0, 0)), "phishing": int(counter.get(1, 0))}


def _split_diagnostics(records, train_indices: list[int], validation_indices: list[int], test_indices: list[int]) -> dict[str, Any]:
    train_records = [records[index] for index in train_indices]
    validation_records = [records[index] for index in validation_indices]
    test_records = [records[index] for index in test_indices]

    def _fingerprints(items) -> set[str]:
        return {record.fingerprint for record in items}

    def _groups(items) -> set[str]:
        return {record.group_id for record in items}

    train_fingerprints = _fingerprints(train_records)
    validation_fingerprints = _fingerprints(validation_records)
    test_fingerprints = _fingerprints(test_records)
    train_groups = _groups(train_records)
    validation_groups = _groups(validation_records)
    test_groups = _groups(test_records)

    return {
        "label_distribution": {
            "train": _label_distribution(train_records),
            "validation": _label_distribution(validation_records),
            "test": _label_distribution(test_records),
        },
        "group_counts": {
            "train": len(train_groups),
            "validation": len(validation_groups),
            "test": len(test_groups),
        },
        "group_overlap": {
            "train_validation": len(train_groups & validation_groups),
            "train_test": len(train_groups & test_groups),
            "validation_test": len(validation_groups & test_groups),
        },
        "fingerprint_overlap": {
            "train_validation": len(train_fingerprints & validation_fingerprints),
            "train_test": len(train_fingerprints & test_fingerprints),
            "validation_test": len(validation_fingerprints & test_fingerprints),
        },
    }


def _overfitting_summary(training_metrics: dict[str, Any], validation_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> dict[str, Any]:
    validation_gap = round(float(training_metrics["f1"]) - float(validation_metrics["f1"]), 4)
    test_gap = round(float(training_metrics["f1"]) - float(test_metrics["f1"]), 4)
    suspicious = validation_gap > 0.08 or test_gap > 0.08
    return {
        "training_validation_f1_gap": validation_gap,
        "training_test_f1_gap": test_gap,
        "possible_overfitting": suspicious,
    }


def _token_feature_views(records, indices: list[int]) -> tuple[list[dict[str, int]], list[int]]:
    token_counts = [records[index].token_counts for index in indices]
    keyword_hit_counts = [len(set(token_counts_map).intersection(SUSPICIOUS_KEYWORDS)) for token_counts_map in token_counts]
    return token_counts, keyword_hit_counts


def _split_error_report(
    records,
    indices: list[int],
    probabilities: list[float],
    *,
    threshold: float = 0.5,
    max_examples: int = 10,
) -> dict[str, Any]:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    per_source_totals: dict[str, dict[str, int]] = {}

    for index, probability in zip(indices, probabilities, strict=False):
        record = records[index]
        predicted = 1 if probability >= threshold else 0
        source_bucket = per_source_totals.setdefault(
            record.source_file,
            {"samples": 0, "correct": 0, "false_positives": 0, "false_negatives": 0},
        )
        source_bucket["samples"] += 1
        if predicted == record.label:
            source_bucket["correct"] += 1
            continue

        example = {
            "source_file": record.source_file,
            "subject": record.subject or "(no subject)",
            "probability": round(float(probability), 4),
            "expected_label": "phishing" if record.label == 1 else "benign",
            "predicted_label": "phishing" if predicted == 1 else "benign",
            "text_preview": record.text[:220],
        }
        if predicted == 1 and record.label == 0:
            source_bucket["false_positives"] += 1
            if len(false_positives) < max_examples:
                false_positives.append(example)
        elif predicted == 0 and record.label == 1:
            source_bucket["false_negatives"] += 1
            if len(false_negatives) < max_examples:
                false_negatives.append(example)

    per_source_metrics = {}
    for source_file, bucket in per_source_totals.items():
        samples = max(bucket["samples"], 1)
        per_source_metrics[source_file] = {
            **bucket,
            "accuracy": round(bucket["correct"] / samples, 4),
        }
    return {
        "false_positive_examples": false_positives,
        "false_negative_examples": false_negatives,
        "per_source_metrics": per_source_metrics,
    }


def train_base_model(
    *,
    settings: Settings | None = None,
    max_rows_per_file: int | None = None,
    force_backend: str | None = None,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    use_preprocessing_cache: bool = True,
    rebuild_preprocessing_cache: bool = False,
    log_progress: bool = True,
    workers: int = 1,
    evaluation_batch_size: int = 64,
) -> dict[str, Any]:
    """Train the base phishing detector using local datasets only."""

    effective = ensure_runtime_dirs(settings)
    progress_callback = print if log_progress else None
    cache_path = build_preprocessing_cache_path(
        effective.dataset_dir,
        effective.cache_dir,
        max_rows_per_file=max_rows_per_file,
    )
    if use_preprocessing_cache and cache_path.exists() and not rebuild_preprocessing_cache:
        if progress_callback:
            progress_callback(f"Loading preprocessed dataset cache: {cache_path}")
        dataset = load_preprocessed_dataset(cache_path)
    else:
        if progress_callback:
            progress_callback("Building preprocessed dataset from local CSV files...")
        dataset = load_unified_dataset(
            effective.dataset_dir,
            max_rows_per_file=max_rows_per_file,
            progress_callback=progress_callback,
            workers=workers,
        )
        if use_preprocessing_cache:
            save_preprocessed_dataset(cache_path, dataset)
            if progress_callback:
                progress_callback(f"Saved preprocessing cache: {cache_path}")
    records = dataset.records
    samples = [(record.text, record.label) for record in records]
    if len(samples) < 3:
        raise RuntimeError("At least two normalized samples are required for training.")
    if progress_callback:
        progress_callback(
            f"Prepared {len(samples)} normalized samples. Starting grouped split with train={train_ratio:.0%}, validation={validation_ratio:.0%}, test={test_ratio:.0%}."
        )

    grouped_indices = [(index, record.group_id) for index, record in enumerate(records)]
    train_indices, validation_indices, test_indices = grouped_train_validation_test_split(
        grouped_indices,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=random_seed,
    )
    train_samples = [samples[index] for index in train_indices]
    validation_samples = [samples[index] for index in validation_indices]
    test_samples = [samples[index] for index in test_indices]
    validation_token_counts, validation_keyword_hits = _token_feature_views(records, validation_indices)
    test_token_counts, test_keyword_hits = _token_feature_views(records, test_indices)
    training_snapshot_indices = train_indices[: min(len(train_indices), 128)]
    training_snapshot_samples = [samples[index] for index in training_snapshot_indices]
    training_snapshot_token_counts, training_snapshot_keyword_hits = _token_feature_views(records, training_snapshot_indices)
    if progress_callback:
        progress_callback(
            f"Split ready: train={len(train_samples)}, validation={len(validation_samples)}, test={len(test_samples)}"
        )
    model = select_training_backend(effective, force_backend=force_backend)
    if progress_callback:
        progress_callback(f"Training backend selected: {model.backend_name}")
    model.fit(train_samples)
    if progress_callback:
        progress_callback(f"Training complete. Evaluating validation set ({len(validation_samples)} samples)...")
    validation_metrics = evaluate_model(
        model,
        validation_samples,
        progress_callback=progress_callback,
        stage_name="validation",
        batch_size=evaluation_batch_size,
        token_counts_sequence=validation_token_counts,
        keyword_hit_counts=validation_keyword_hits,
    )
    if progress_callback:
        progress_callback(f"Validation evaluation complete. Evaluating test set ({len(test_samples)} samples)...")
    test_metrics = evaluate_model(
        model,
        test_samples,
        progress_callback=progress_callback,
        stage_name="test",
        batch_size=evaluation_batch_size,
        token_counts_sequence=test_token_counts,
        keyword_hit_counts=test_keyword_hits,
    )
    if progress_callback:
        progress_callback("Test evaluation complete. Evaluating training snapshot...")
    training_metrics = evaluate_model(
        model,
        training_snapshot_samples,
        progress_callback=progress_callback,
        stage_name="training_snapshot",
        batch_size=evaluation_batch_size,
        token_counts_sequence=training_snapshot_token_counts,
        keyword_hit_counts=training_snapshot_keyword_hits,
    )
    leakage_checks = _split_diagnostics(records, train_indices, validation_indices, test_indices)
    overfitting_checks = _overfitting_summary(training_metrics, validation_metrics, test_metrics)
    validation_probabilities = model.predict_proba_from_token_counts(
        validation_token_counts,
        keyword_hit_counts=validation_keyword_hits,
    )
    test_probabilities = model.predict_proba_from_token_counts(
        test_token_counts,
        keyword_hit_counts=test_keyword_hits,
    )
    validation_error_report = _split_error_report(records, validation_indices, validation_probabilities)
    test_error_report = _split_error_report(records, test_indices, test_probabilities)
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
            "group_split": True,
            "workers": workers,
        },
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "leakage_checks": leakage_checks,
        "overfitting_checks": overfitting_checks,
        "validation_error_report": validation_error_report,
        "test_error_report": test_error_report,
        "artifacts": {
            "model_path": str(effective.model_artifact_path),
            "metrics_path": str(effective.base_metrics_path),
            "test_samples_path": str(effective.test_samples_path),
            "preprocessing_cache_path": str(cache_path) if use_preprocessing_cache else None,
        },
    }
    dump_json(effective.base_metrics_path, metrics)
    if progress_callback:
        progress_callback(f"Saved training metrics: {effective.base_metrics_path}")
    with effective.test_samples_path.open("w", encoding="utf-8") as handle:
        for index in test_indices:
            record = records[index]
            payload = {
                "text": record.text,
                "label": record.label,
                "source_file": record.source_file,
                "subject": record.subject,
                "urls": record.urls,
                "fingerprint": record.fingerprint,
                "group_id": record.group_id,
            }
            handle.write(json.dumps(payload) + "\n")
    if progress_callback:
        progress_callback(f"Saved demo test samples: {effective.test_samples_path}")
    return metrics


if __name__ == "__main__":
    result = train_base_model()
    print(result)
