"""Local dataset loading and normalization."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.utils.config import Settings
from app.utils.text import clean_text, extract_urls, tokenize


csv.field_size_limit(10**8)

TEXT_COLUMN_CANDIDATES = ("text", "body", "email", "content", "message", "text_combined")
LABEL_COLUMN_CANDIDATES = ("label", "target", "class", "is_phishing", "spam")
URL_COLUMN_CANDIDATES = ("url", "urls", "link", "links")
POSITIVE_LABEL_VALUES = {"1", "phishing", "spam", "malicious", "fraud", "true", "yes"}
NEGATIVE_LABEL_VALUES = {"0", "benign", "ham", "legitimate", "safe", "false", "no"}
CACHE_SCHEMA_VERSION = 2


@dataclass(slots=True)
class DatasetRecord:
    """Normalized dataset record."""

    text: str
    label: int
    source_file: str
    subject: str = ""
    sender: str = ""
    receiver: str = ""
    urls: list[str] = field(default_factory=list)
    fingerprint: str = ""
    group_id: str = ""
    sender_domain: str = ""
    token_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UnifiedDataset:
    """Container for normalized phishing training samples."""

    records: list[DatasetRecord]
    text_columns: dict[str, str]
    raw_rows: int

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.records), 9)

    @property
    def texts(self) -> list[str]:
        return [record.text for record in self.records]

    @property
    def labels(self) -> list[int]:
        return [record.label for record in self.records]

    def to_records(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], text_columns: dict[str, str], raw_rows: int) -> "UnifiedDataset":
        return cls(records=[DatasetRecord(**record) for record in records], text_columns=text_columns, raw_rows=raw_rows)


def _normalize_column_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", "_").split())


def _detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _detect_text_column(columns: list[str]) -> str | None:
    detected = _detect_column(columns, TEXT_COLUMN_CANDIDATES)
    if detected:
        return detected
    normalized = {_normalize_column_name(column): column for column in columns}
    if "body" in normalized and "subject" in normalized:
        return normalized["body"]
    return None


def _detect_label_column(columns: list[str]) -> str | None:
    return _detect_column(columns, LABEL_COLUMN_CANDIDATES)


def _detect_url_column(columns: list[str]) -> str | None:
    return _detect_column(columns, URL_COLUMN_CANDIDATES)


def _infer_label(raw_label: Any, source_file: str) -> int:
    if raw_label is None:
        lowered = source_file.lower()
        if any(marker in lowered for marker in ("phishing", "fraud", "nazario")):
            return 1
        return 0
    value = str(raw_label).strip().lower()
    if value in POSITIVE_LABEL_VALUES:
        return 1
    if value in NEGATIVE_LABEL_VALUES:
        return 0
    try:
        return 1 if float(value) >= 1.0 else 0
    except ValueError:
        lowered = source_file.lower()
        if any(marker in lowered for marker in ("phishing", "fraud", "nazario")):
            return 1
        return 0


def _compose_text(row: dict[str, Any], text_column: str | None) -> str:
    parts: list[str] = []
    subject = str(row.get("subject", "") or "")
    if subject.strip():
        parts.append(subject.strip())
    if text_column and row.get(text_column):
        parts.append(str(row[text_column]))
    elif row.get("body"):
        parts.append(str(row["body"]))
    return clean_text(" ".join(parts), max_length=12000)


def _canonical_template_text(value: str, token_limit: int = 80) -> str:
    lowered = clean_text(value).lower()
    lowered = re.sub(r"https?://\S+|www\.\S+", " urltoken ", lowered)
    lowered = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " emailtoken ", lowered)
    lowered = re.sub(r"\d+", " numtoken ", lowered)
    tokens = tokenize(lowered)
    return " ".join(tokens[:token_limit])


def _fingerprint_text(value: str) -> str:
    canonical = _canonical_template_text(value, token_limit=160)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sender_domain(sender: str) -> str:
    if "@" not in sender:
        return ""
    return sender.rsplit("@", 1)[-1].strip(" >").lower()


def _group_record(*, source_file: str, subject: str, sender: str, text: str) -> str:
    subject_signature = _canonical_template_text(subject, token_limit=12)
    body_signature = _canonical_template_text(text, token_limit=30)
    domain = _sender_domain(sender)
    raw = f"{source_file}|{domain}|{subject_signature}|{body_signature}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_row_payload(job: tuple[dict[str, Any], str, str | None, str | None, str | None]) -> dict[str, Any] | None:
    row, source_file, normalized_text_column, label_key, normalized_url_column = job
    normalized_row = {_normalize_column_name(key): value for key, value in row.items() if key}
    text = _compose_text(normalized_row, normalized_text_column)
    if not text:
        return None
    label = _infer_label(normalized_row.get(label_key) if label_key else None, source_file)
    urls_field = str(normalized_row.get(normalized_url_column, "") or "") if normalized_url_column else ""
    urls = [candidate.strip() for candidate in urls_field.split() if candidate.strip()]
    urls.extend(extract_urls(text))
    sender = str(normalized_row.get("sender", "") or "")
    subject = str(normalized_row.get("subject", "") or "")
    fingerprint = _fingerprint_text(text)
    token_counts = dict(Counter(tokenize(text)))
    return {
        "text": text,
        "label": label,
        "source_file": source_file,
        "subject": subject,
        "sender": sender,
        "receiver": str(normalized_row.get("receiver", "") or ""),
        "urls": sorted(set(urls)),
        "fingerprint": fingerprint,
        "group_id": _group_record(source_file=source_file, subject=subject, sender=sender, text=text),
        "sender_domain": _sender_domain(sender),
        "token_counts": token_counts,
    }


def discover_dataset_files(dataset_dir: Path) -> list[Path]:
    """Return every CSV file located in the dataset directory."""

    return sorted(path for path in dataset_dir.glob("*.csv") if path.is_file())


def build_preprocessing_cache_path(
    dataset_dir: Path,
    cache_dir: Path,
    *,
    max_rows_per_file: int | None,
) -> Path:
    """Build a deterministic cache filename based on dataset file metadata and preprocessing settings."""

    files = discover_dataset_files(dataset_dir)
    signature_parts = [
        f"schema={CACHE_SCHEMA_VERSION}",
        f"max_rows={max_rows_per_file if max_rows_per_file is not None else 'all'}",
    ]
    for path in files:
        stat = path.stat()
        signature_parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    digest = hashlib.sha256("|".join(signature_parts).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"preprocessed_dataset_{digest}.json"


def save_preprocessed_dataset(path: Path, dataset: UnifiedDataset) -> None:
    """Persist the normalized dataset to a local cache file."""

    payload = {
        "raw_rows": dataset.raw_rows,
        "text_columns": dataset.text_columns,
        "records": dataset.to_records(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_preprocessed_dataset(path: Path) -> UnifiedDataset:
    """Load a normalized dataset from a local cache file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return UnifiedDataset.from_records(
        records=list(payload.get("records", [])),
        text_columns=dict(payload.get("text_columns", {})),
        raw_rows=int(payload.get("raw_rows", 0)),
    )


def load_unified_dataset(
    dataset_dir: Path | None = None,
    *,
    drop_duplicates: bool = True,
    max_rows_per_file: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    workers: int = 1,
) -> UnifiedDataset:
    """Read, normalize, and merge every CSV file under the dataset directory."""

    settings = Settings()
    target_dir = dataset_dir or settings.dataset_dir
    files = discover_dataset_files(target_dir)
    records: list[DatasetRecord] = []
    text_columns: dict[str, str] = {}
    raw_rows = 0
    seen_fingerprints: set[str] = set()
    worker_count = max(1, min(int(workers), os.cpu_count() or 1))

    for path in files:
        if progress_callback:
            progress_callback(f"Scanning {path.name}...")
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            text_column = _detect_text_column(columns)
            label_column = _detect_label_column(columns)
            url_column = _detect_url_column(columns)
            if text_column is None and "subject" not in {_normalize_column_name(column) for column in columns}:
                continue
            text_columns[path.name] = _normalize_column_name(text_column or "subject_body")
            normalized_text_column = _normalize_column_name(text_column) if text_column else None
            label_key = _normalize_column_name(label_column) if label_column else None
            normalized_url_column = _normalize_column_name(url_column) if url_column else None
            row_iter = reader if max_rows_per_file is None else itertools.islice(reader, max_rows_per_file)
            jobs = (
                (row, path.name, normalized_text_column, label_key, normalized_url_column)
                for row in row_iter
            )
            payload_iter = jobs
            executor = None
            if worker_count > 1:
                try:
                    executor = ProcessPoolExecutor(max_workers=worker_count)
                    payload_iter = executor.map(_normalize_row_payload, jobs, chunksize=64)
                except Exception:
                    executor = None
                    payload_iter = jobs
                    if progress_callback:
                        progress_callback(
                            f"Multiprocessing unavailable for {path.name}; falling back to single-process preprocessing."
                        )
            processed_for_file = 0
            try:
                for payload in payload_iter:
                    raw_rows += 1
                    if executor is None and not isinstance(payload, dict):
                        payload = _normalize_row_payload(payload)
                    if payload is None:
                        continue
                    processed_for_file += 1
                    fingerprint = str(payload["fingerprint"])
                    if drop_duplicates and fingerprint in seen_fingerprints:
                        continue
                    seen_fingerprints.add(fingerprint)
                    records.append(DatasetRecord(**payload))
                    if progress_callback and processed_for_file % 1000 == 0:
                        progress_callback(
                            f"{path.name}: processed {processed_for_file} normalized rows, total kept = {len(records)}"
                        )
            finally:
                if executor is not None:
                    executor.shutdown()
        if progress_callback:
            progress_callback(f"Finished {path.name}: normalized records so far = {len(records)}")

    return UnifiedDataset(records=records, text_columns=text_columns, raw_rows=raw_rows)
