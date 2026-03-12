"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from app.main import create_app
from app.services.container import build_service_container
from app.utils.config import DATASET_DIR, Settings


def create_test_settings(tmp_path: Path) -> Settings:
    models_dir = tmp_path / "models"
    synthetic_dir = tmp_path / "synthetic"
    return Settings(
        dataset_dir=DATASET_DIR,
        models_dir=models_dir,
        synthetic_dir=synthetic_dir,
        model_artifact_path=models_dir / "phishing_detector.pt",
        base_metrics_path=models_dir / "base_training_metrics.json",
        retraining_metrics_path=models_dir / "retraining_metrics.json",
        test_samples_path=models_dir / "demo_test_samples.jsonl",
        feedback_db_path=models_dir / "phishing_runtime.db",
        enable_remote_fetch=False,
        allow_model_download=False,
    )


def count_raw_rows(dataset_dir: Path, max_rows_per_file: int | None = None) -> int:
    total = 0
    for path in dataset_dir.glob("*.csv"):
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, _row in enumerate(reader):
                if max_rows_per_file is not None and index >= max_rows_per_file:
                    break
                total += 1
    return total


def create_test_app(tmp_path: Path):
    settings = create_test_settings(tmp_path)
    container = build_service_container(settings)
    return create_app(container), container


def post_json(app, path: str, payload: dict):
    if hasattr(app, "dispatch"):
        return asyncio.run(app.dispatch("POST", path, payload))
    from fastapi.testclient import TestClient  # pragma: no cover

    client = TestClient(app)
    return client.post(path, json=payload)
