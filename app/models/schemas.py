"""Validation and response schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ValidationError(ValueError):
    """Raised when request payloads fail validation."""


def _require_non_empty_string(payload: dict[str, Any], key: str, max_length: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"'{key}' must be a string.")
    trimmed = value.strip()
    if not trimmed:
        raise ValidationError(f"'{key}' cannot be empty.")
    if len(trimmed) > max_length:
        raise ValidationError(f"'{key}' exceeds the maximum allowed length of {max_length}.")
    return trimmed


@dataclass(slots=True)
class TextAnalysisRequest:
    """Request payload for text-only phishing analysis."""

    text: str

    @classmethod
    def validate(cls, payload: dict[str, Any], max_length: int = 12000) -> "TextAnalysisRequest":
        return cls(text=_require_non_empty_string(payload, "text", max_length))


@dataclass(slots=True)
class URLAnalysisRequest:
    """Request payload for URL-only phishing analysis."""

    url: str

    @classmethod
    def validate(cls, payload: dict[str, Any], max_length: int = 4000) -> "URLAnalysisRequest":
        return cls(url=_require_non_empty_string(payload, "url", max_length))


@dataclass(slots=True)
class FullAnalysisRequest:
    """Request payload for full phishing analysis."""

    text: str
    url: str
    html_content: str | None = None

    @classmethod
    def validate(cls, payload: dict[str, Any], max_text_length: int = 12000) -> "FullAnalysisRequest":
        html_content = payload.get("html_content")
        if html_content is not None and not isinstance(html_content, str):
            raise ValidationError("'html_content' must be a string when provided.")
        return cls(
            text=_require_non_empty_string(payload, "text", max_text_length),
            url=_require_non_empty_string(payload, "url", 4000),
            html_content=html_content,
        )


@dataclass(slots=True)
class FeedbackRequest:
    """Feedback payload used for continuous learning simulations."""

    prediction_id: int | None = None
    text: str | None = None
    url: str | None = None
    user_label: int = 0
    notes: str | None = None

    @classmethod
    def validate(cls, payload: dict[str, Any], max_text_length: int = 12000) -> "FeedbackRequest":
        prediction_id = payload.get("prediction_id")
        if prediction_id is not None and not isinstance(prediction_id, int):
            raise ValidationError("'prediction_id' must be an integer when provided.")
        text = payload.get("text")
        if text is not None:
            text = _require_non_empty_string(payload, "text", max_text_length)
        url = payload.get("url")
        if url is not None:
            url = _require_non_empty_string(payload, "url", 4000)
        user_label = payload.get("user_label")
        if user_label not in (0, 1):
            raise ValidationError("'user_label' must be 0 or 1.")
        notes = payload.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValidationError("'notes' must be a string when provided.")
        if prediction_id is None and text is None and url is None:
            raise ValidationError("Feedback requires 'prediction_id' or raw 'text'/'url' context.")
        return cls(
            prediction_id=prediction_id,
            text=text,
            url=url,
            user_label=user_label,
            notes=notes,
        )


@dataclass(slots=True)
class NLPAnalysisResponse:
    phishing_probability: float
    explanation_keywords: list[str]
    backend: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class URLAnalysisResponse:
    url: str
    domain: str
    final_risk_score: float
    explanation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FullAnalysisResponse:
    final_risk_score: float
    decision: str
    explanation: dict[str, Any]
    prediction_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
