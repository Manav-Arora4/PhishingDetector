"""Local dataset loading and normalization."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.utils.config import Settings
from app.utils.text import clean_text, extract_urls


csv.field_size_limit(10**8)

TEXT_COLUMN_CANDIDATES = ("text", "body", "email", "content", "message", "text_combined")
LABEL_COLUMN_CANDIDATES = ("label", "target", "class", "is_phishing", "spam")
URL_COLUMN_CANDIDATES = ("url", "urls", "link", "links")
POSITIVE_LABEL_VALUES = {"1", "phishing", "spam", "malicious", "fraud", "true", "yes"}
NEGATIVE_LABEL_VALUES = {"0", "benign", "ham", "legitimate", "safe", "false", "no"}


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
        return (len(self.records), 6)

    @property
    def texts(self) -> list[str]:
        return [record.text for record in self.records]

    @property
    def labels(self) -> list[int]:
        return [record.label for record in self.records]

    def to_records(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records]


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


def discover_dataset_files(dataset_dir: Path) -> list[Path]:
    """Return every CSV file located in the dataset directory."""

    return sorted(path for path in dataset_dir.glob("*.csv") if path.is_file())


def load_unified_dataset(
    dataset_dir: Path | None = None,
    *,
    drop_duplicates: bool = True,
    max_rows_per_file: int | None = None,
) -> UnifiedDataset:
    """Read, normalize, and merge every CSV file under the dataset directory."""

    settings = Settings()
    target_dir = dataset_dir or settings.dataset_dir
    files = discover_dataset_files(target_dir)
    records: list[DatasetRecord] = []
    text_columns: dict[str, str] = {}
    raw_rows = 0
    seen_texts: set[str] = set()

    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            text_column = _detect_text_column(columns)
            label_column = _detect_label_column(columns)
            url_column = _detect_url_column(columns)
            if text_column is None and "subject" not in {_normalize_column_name(column) for column in columns}:
                continue
            text_columns[path.name] = _normalize_column_name(text_column or "subject_body")

            for index, row in enumerate(reader):
                if max_rows_per_file is not None and index >= max_rows_per_file:
                    break
                raw_rows += 1
                normalized_row = {_normalize_column_name(key): value for key, value in row.items() if key}
                text = _compose_text(normalized_row, _normalize_column_name(text_column) if text_column else None)
                if not text:
                    continue
                label_key = _normalize_column_name(label_column) if label_column else ""
                label = _infer_label(normalized_row.get(label_key) if label_key else None, path.name)
                urls_field = ""
                if url_column:
                    urls_field = str(normalized_row.get(_normalize_column_name(url_column), "") or "")
                urls = [candidate.strip() for candidate in urls_field.split() if candidate.strip()]
                urls.extend(extract_urls(text))
                dedupe_key = clean_text(text).lower()
                if drop_duplicates and dedupe_key in seen_texts:
                    continue
                seen_texts.add(dedupe_key)
                records.append(
                    DatasetRecord(
                        text=text,
                        label=label,
                        source_file=path.name,
                        subject=str(normalized_row.get("subject", "") or ""),
                        sender=str(normalized_row.get("sender", "") or ""),
                        receiver=str(normalized_row.get("receiver", "") or ""),
                        urls=sorted(set(urls)),
                    )
                )

    return UnifiedDataset(records=records, text_columns=text_columns, raw_rows=raw_rows)
