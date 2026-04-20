"""Base training tests."""

from __future__ import annotations

import json

from app.ml.training.train_base_model import train_base_model
from tests.conftest import create_test_settings


def test_base_training_runs(tmp_path):
    settings = create_test_settings(tmp_path)
    metrics = train_base_model(settings=settings, max_rows_per_file=12, force_backend="token_naive_bayes")
    assert metrics["backend"] == "token_naive_bayes"
    assert metrics["dataset"]["normalized_rows"] > 0
    assert metrics["dataset"]["split_counts"]["train"] > 0
    assert metrics["dataset"]["split_counts"]["test"] > 0
    assert "test_metrics" in metrics
    assert metrics["leakage_checks"]["group_overlap"]["train_test"] == 0
    assert metrics["leakage_checks"]["fingerprint_overlap"]["train_test"] == 0


def test_model_saved_after_training(tmp_path):
    settings = create_test_settings(tmp_path)
    train_base_model(settings=settings, max_rows_per_file=12, force_backend="token_naive_bayes")
    assert settings.model_artifact_path.exists()
    assert settings.test_samples_path.exists()
    saved_metrics = json.loads(settings.base_metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["artifacts"]["model_path"].endswith("phishing_detector.pt")


def test_preprocessing_cache_created_and_reused(tmp_path):
    settings = create_test_settings(tmp_path)
    first_metrics = train_base_model(
        settings=settings,
        max_rows_per_file=6,
        force_backend="token_naive_bayes",
        log_progress=False,
    )
    cache_files = list(settings.cache_dir.glob("preprocessed_dataset_*.json"))
    assert cache_files
    second_metrics = train_base_model(
        settings=settings,
        max_rows_per_file=6,
        force_backend="token_naive_bayes",
        log_progress=False,
    )
    assert first_metrics["dataset"]["normalized_rows"] == second_metrics["dataset"]["normalized_rows"]
