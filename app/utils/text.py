"""Text and tokenization helpers used across the platform."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


URL_PATTERN = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9@._'-]+")
BASE64_LIKE_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{16,}={0,2})")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

SUSPICIOUS_KEYWORDS = {
    "verify",
    "urgent",
    "suspend",
    "suspended",
    "account",
    "password",
    "credential",
    "invoice",
    "security",
    "confirm",
    "click",
    "login",
    "limited",
    "immediately",
}


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_data(self) -> str:
        return " ".join(self.parts)


def strip_html(raw_html: str) -> str:
    """Strip HTML tags while preserving text content."""

    if "<" not in raw_html and ">" not in raw_html:
        return raw_html
    stripper = _HTMLStripper()
    stripper.feed(raw_html)
    return stripper.get_data()


def normalize_whitespace(value: str) -> str:
    """Collapse whitespace into a single space."""

    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: str, max_length: int | None = None) -> str:
    """Normalize text for both training and inference."""

    normalized = html.unescape(value or "")
    normalized = strip_html(normalized)
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = normalize_whitespace(normalized.replace("\x00", " "))
    if max_length is not None:
        return normalized[:max_length]
    return normalized


def tokenize(value: str) -> list[str]:
    """Tokenize text using a lightweight regex-based tokenizer."""

    return [token.lower() for token in TOKEN_PATTERN.findall(clean_text(value))]


def extract_urls(value: str) -> list[str]:
    """Extract URLs embedded in a text body."""

    return [match.group(0) for match in URL_PATTERN.finditer(value or "")]


def contains_base64_like(value: str) -> bool:
    """Detect long encoded-looking strings often used in phishing URLs."""

    return bool(BASE64_LIKE_PATTERN.search(value or ""))


def suspicious_keyword_hits(value: str) -> list[str]:
    """Return suspicious keywords present in the provided text."""

    tokens = set(tokenize(value))
    return sorted(tokens.intersection(SUSPICIOUS_KEYWORDS))
