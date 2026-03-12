"""Structural webpage analysis for phishing detection."""

from __future__ import annotations

import asyncio
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from app.utils.cache import TTLCache
from app.utils.config import Settings


BRAND_KEYWORDS = {"paypal", "microsoft", "google", "okta", "amazon", "docusign", "office365"}


class _FeatureParser(HTMLParser):
    def __init__(self, page_domain: str) -> None:
        super().__init__()
        self.page_domain = page_domain
        self.form_count = 0
        self.iframe_count = 0
        self.external_scripts = 0
        self.brand_hits: set[str] = set()
        self.password_fields = 0
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        if tag == "form":
            self.form_count += 1
        if tag == "iframe":
            self.iframe_count += 1
        if tag == "script":
            source = attributes.get("src", "")
            if source:
                host = urlsplit(source).netloc.lower()
                if host and self.page_domain and host != self.page_domain:
                    self.external_scripts += 1
        if tag == "input" and attributes.get("type", "").lower() == "password":
            self.password_fields += 1

    def handle_data(self, data: str) -> None:
        self._text_chunks.append(data.lower())

    def finalize(self) -> None:
        joined = " ".join(self._text_chunks)
        self.brand_hits = {keyword for keyword in BRAND_KEYWORDS if keyword in joined}


@dataclass(slots=True)
class StructuralRiskReport:
    """Structured report describing webpage HTML risk features."""

    final_risk_score: float
    explanation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuralWebAnalyzer:
    """Safely fetch and inspect webpage structure without executing active content."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._cache = TTLCache[StructuralRiskReport](ttl_seconds=self.settings.domain_cache_ttl_seconds)

    async def _fetch_html(self, url: str) -> str | None:
        if not self.settings.enable_remote_fetch:
            return None

        def _blocking_fetch() -> str | None:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "PhishingDetector/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                raw = response.read(self.settings.max_fetch_bytes)
                return raw.decode("utf-8", errors="ignore")

        try:
            return await asyncio.to_thread(_blocking_fetch)
        except Exception:
            return None

    async def analyze(self, url: str, html_content: str | None = None) -> StructuralRiskReport:
        """Inspect webpage HTML for phishing-oriented structural patterns."""

        cache_key = f"{url}|{hash(html_content) if html_content else 'fetch'}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        html = html_content or await self._fetch_html(url)
        if not html:
            report = StructuralRiskReport(
                final_risk_score=0.05,
                explanation={
                    "fetched": False,
                    "form_count": 0,
                    "iframe_count": 0,
                    "external_scripts": 0,
                    "brand_keywords": [],
                },
            )
            self._cache.set(cache_key, report)
            return report

        page_domain = urlsplit(url).netloc.lower()
        parser = _FeatureParser(page_domain)
        parser.feed(html)
        parser.finalize()
        inline_obfuscation = len(re.findall(r"atob\(|fromCharCode\(", html))

        risk_score = 0.0
        risk_score += min(parser.form_count * 0.18, 0.36)
        risk_score += min(parser.iframe_count * 0.12, 0.24)
        risk_score += min(parser.external_scripts * 0.08, 0.24)
        risk_score += 0.18 if parser.brand_hits else 0.0
        risk_score += 0.18 if parser.password_fields else 0.0
        risk_score += min(inline_obfuscation * 0.06, 0.12)
        risk_score = round(min(risk_score, 1.0), 4)

        report = StructuralRiskReport(
            final_risk_score=risk_score,
            explanation={
                "fetched": True,
                "form_count": parser.form_count,
                "iframe_count": parser.iframe_count,
                "external_scripts": parser.external_scripts,
                "brand_keywords": sorted(parser.brand_hits),
                "password_fields": parser.password_fields,
                "inline_obfuscation_hits": inline_obfuscation,
            },
        )
        self._cache.set(cache_key, report)
        return report
