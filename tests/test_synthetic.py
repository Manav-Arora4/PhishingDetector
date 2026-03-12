"""Synthetic phishing data tests."""

from __future__ import annotations

from app.ml.synthetic.generator import generate_synthetic_text, generate_synthetic_urls


def test_generate_synthetic_text_returns_samples():
    samples = generate_synthetic_text(count=6, seed=7)
    assert len(samples) == 6
    assert all(sample["label"] == 1 for sample in samples)


def test_generated_urls_contain_homoglyph_patterns():
    urls = generate_synthetic_urls(count=8, seed=11)
    assert any("paypaI" in url or "micros0ft" in url or "g00gle" in url for url in urls)
