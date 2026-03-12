"""Dataset loader tests."""

from __future__ import annotations

from app.ml.training.load_local_datasets import discover_dataset_files, load_unified_dataset
from app.utils.config import DATASET_DIR
from tests.conftest import count_raw_rows


def test_all_datasets_loaded():
    files = discover_dataset_files(DATASET_DIR)
    dataset = load_unified_dataset(DATASET_DIR, max_rows_per_file=3)
    assert len(files) == 7
    assert len(dataset.text_columns) == len(files)


def test_dataset_merge_has_expected_rows():
    dataset = load_unified_dataset(DATASET_DIR, max_rows_per_file=2)
    expected_raw_rows = count_raw_rows(DATASET_DIR, max_rows_per_file=2)
    assert dataset.raw_rows == expected_raw_rows
    assert 0 < len(dataset.records) <= expected_raw_rows


def test_text_column_detected():
    dataset = load_unified_dataset(DATASET_DIR, max_rows_per_file=1)
    assert dataset.text_columns["phishing_email.csv"] == "text_combined"
    assert dataset.text_columns["CEAS_08.csv"] == "body"
    assert dataset.text_columns["Enron.csv"] == "body"


def test_labels_generated_correctly():
    dataset = load_unified_dataset(DATASET_DIR, max_rows_per_file=5)
    labels = set(dataset.labels)
    assert labels == {0, 1}
