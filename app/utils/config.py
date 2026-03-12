"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = APP_DIR / "models"
SYNTHETIC_DIR = APP_DIR / "ml" / "synthetic" / "generated_data"


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the phishing detection system."""

    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = DATASET_DIR
    models_dir: Path = MODELS_DIR
    synthetic_dir: Path = SYNTHETIC_DIR
    model_artifact_path: Path = MODELS_DIR / "phishing_detector.pt"
    base_metrics_path: Path = MODELS_DIR / "base_training_metrics.json"
    retraining_metrics_path: Path = MODELS_DIR / "retraining_metrics.json"
    test_samples_path: Path = MODELS_DIR / "demo_test_samples.jsonl"
    feedback_db_path: Path = MODELS_DIR / "phishing_runtime.db"
    default_model_name: str = os.getenv("PHISHING_MODEL_NAME", "distilbert-base-uncased")
    allow_model_download: bool = os.getenv("PHISHING_ALLOW_MODEL_DOWNLOAD", "0") == "1"
    enable_remote_fetch: bool = os.getenv("PHISHING_ENABLE_REMOTE_FETCH", "0") == "1"
    request_timeout_seconds: float = float(os.getenv("PHISHING_REQUEST_TIMEOUT", "3.0"))
    max_fetch_bytes: int = int(os.getenv("PHISHING_MAX_FETCH_BYTES", "131072"))
    max_text_length: int = int(os.getenv("PHISHING_MAX_TEXT_LENGTH", "12000"))
    rate_limit_requests: int = int(os.getenv("PHISHING_RATE_LIMIT_REQUESTS", "120"))
    rate_limit_window_seconds: int = int(os.getenv("PHISHING_RATE_LIMIT_WINDOW_SECONDS", "60"))
    domain_cache_ttl_seconds: int = int(os.getenv("PHISHING_DOMAIN_CACHE_TTL", "300"))


def ensure_runtime_dirs(settings: Settings | None = None) -> Settings:
    """Create required runtime directories if they do not exist."""

    effective = settings or Settings()
    effective.dataset_dir.mkdir(parents=True, exist_ok=True)
    effective.models_dir.mkdir(parents=True, exist_ok=True)
    effective.synthetic_dir.mkdir(parents=True, exist_ok=True)
    return effective
