"""URL risk analysis for phishing detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from app.utils.cache import TTLCache
from app.utils.config import Settings
from app.utils.text import contains_base64_like


SUSPICIOUS_TLDS = {".zip", ".mov", ".click", ".work", ".support", ".top", ".gq", ".tk", ".cf", ".ml"}
KNOWN_BRANDS = {"paypal", "microsoft", "google", "amazon", "docusign", "okta", "office365", "outlook"}
HOST_RISK_TERMS = {"login", "secure", "verify", "update", "account", "signin", "confirm"}
CONFUSABLE_MAP = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "3": "e",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "|": "l",
        "I": "l",
    }
)


@dataclass(slots=True)
class URLRiskReport:
    """Structured URL risk assessment."""

    url: str
    domain: str
    final_risk_score: float
    explanation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class URLAnalysisEngine:
    """Analyze phishing characteristics present in URLs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._cache = TTLCache[URLRiskReport](ttl_seconds=self.settings.domain_cache_ttl_seconds)

    @staticmethod
    def _raw_host(url: str) -> str:
        netloc = urlsplit(url).netloc
        return netloc.split("@")[-1].split(":")[0]

    @staticmethod
    def _normalized_host(host: str) -> str:
        return host.translate(CONFUSABLE_MAP).lower()

    @staticmethod
    def _detect_brand_impersonation(host: str) -> bool:
        lowered = host.lower()
        normalized = URLAnalysisEngine._normalized_host(host)
        return any(
            (brand in normalized and brand not in lowered)
            or (normalized.startswith(brand) and lowered != normalized)
            for brand in KNOWN_BRANDS
        )

    @staticmethod
    def _detect_excessive_subdomains(host: str) -> bool:
        return len([segment for segment in host.split(".") if segment]) >= 4

    @staticmethod
    def _detect_suspicious_tld(host: str) -> bool:
        parts = host.lower().rsplit(".", 1)
        if len(parts) != 2:
            return False
        return f".{parts[-1]}" in SUSPICIOUS_TLDS

    @staticmethod
    def _detect_encoded_query(url: str) -> bool:
        query = urlsplit(url).query
        if contains_base64_like(query):
            return True
        return any(contains_base64_like(value) for values in parse_qs(query).values() for value in values)

    async def analyze(self, url: str) -> URLRiskReport:
        """Analyze a URL and return a structured risk report."""

        cached = self._cache.get(url)
        if cached is not None:
            return cached

        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            report = URLRiskReport(
                url=url,
                domain="",
                final_risk_score=1.0,
                explanation={"invalid_scheme": True, "risk_factors": ["Unsupported URL scheme."]},
            )
            self._cache.set(url, report)
            return report

        host = self._raw_host(url)
        suspicious_tld = self._detect_suspicious_tld(host)
        excessive_subdomains = self._detect_excessive_subdomains(host)
        homoglyph_detected = self._detect_brand_impersonation(host)
        encoded_query = self._detect_encoded_query(url)
        path_depth = len([segment for segment in parsed.path.split("/") if segment])
        deep_path = path_depth >= 3
        host_keyword_hits = sorted(term for term in HOST_RISK_TERMS if term in host.lower())

        risk_score = 0.0
        risk_score += 0.38 if homoglyph_detected else 0.0
        risk_score += 0.18 if suspicious_tld else 0.0
        risk_score += 0.16 if excessive_subdomains else 0.0
        risk_score += 0.16 if encoded_query else 0.0
        risk_score += 0.06 if deep_path else 0.0
        risk_score += min(len(host_keyword_hits) * 0.12, 0.24)
        risk_score = round(min(risk_score, 1.0), 4)

        risk_factors = [
            label
            for label, enabled in (
                ("homoglyph_brand_impersonation", homoglyph_detected),
                ("suspicious_tld", suspicious_tld),
                ("excessive_subdomains", excessive_subdomains),
                ("encoded_query_string", encoded_query),
                ("deep_path", deep_path),
                ("host_risk_terms", bool(host_keyword_hits)),
            )
            if enabled
        ]

        report = URLRiskReport(
            url=url,
            domain=host.lower(),
            final_risk_score=risk_score,
            explanation={
                "homoglyph_detected": homoglyph_detected,
                "suspicious_tld": suspicious_tld,
                "excessive_subdomains": excessive_subdomains,
                "encoded_query_string": encoded_query,
                "path_depth": path_depth,
                "host_keyword_hits": host_keyword_hits,
                "risk_factors": risk_factors,
            },
        )
        self._cache.set(url, report)
        return report
