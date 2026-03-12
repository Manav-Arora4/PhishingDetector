"""Incremental retraining tests."""

from __future__ import annotations

from app.ml.retraining.retrain_incremental import run_incremental_retraining
from app.ml.training.model_backends import load_saved_model
from app.ml.training.train_base_model import train_base_model
from tests.conftest import create_test_settings


def test_incremental_retraining_updates_weights(tmp_path):
    settings = create_test_settings(tmp_path)
    train_base_model(settings=settings, max_rows_per_file=10, force_backend="token_naive_bayes")
    before = load_saved_model(settings.model_artifact_path).snapshot_weights()
    run_incremental_retraining(settings=settings, synthetic_count=8, max_base_rows_per_file=10)
    after = load_saved_model(settings.model_artifact_path).snapshot_weights()
    assert before != after


def test_retraining_metrics_created(tmp_path):
    settings = create_test_settings(tmp_path)
    train_base_model(settings=settings, max_rows_per_file=10, force_backend="token_naive_bayes")
    run_incremental_retraining(settings=settings, synthetic_count=8, max_base_rows_per_file=10)
    assert settings.retraining_metrics_path.exists()
