"""URL analyzer tests."""

from __future__ import annotations

import asyncio

from app.security.url_analyzer import URLAnalysisEngine


def test_detects_homoglyph_domain():
    engine = URLAnalysisEngine()
    report = asyncio.run(engine.analyze("http://paypaI-secure-login.com/update"))
    assert report.explanation["homoglyph_detected"] is True


def test_detects_base64_encoded_params():
    engine = URLAnalysisEngine()
    report = asyncio.run(
        engine.analyze("https://example.com/login?continue=ZXZlbnQ9dmVyaWZ5JnRva2VuPWFsZXJ0&id=42")
    )
    assert report.explanation["encoded_query_string"] is True


def test_safe_url_low_risk():
    engine = URLAnalysisEngine()
    report = asyncio.run(engine.analyze("https://example.com/about"))
    assert report.final_risk_score < 0.2
