"""API tests."""

from __future__ import annotations

from tests.conftest import create_test_app, post_json


def test_analyze_text_endpoint_returns_json(tmp_path):
    app, _container = create_test_app(tmp_path)
    response = post_json(app, "/analyze/text", {"text": "Please verify your account immediately."})
    body = response.json()
    assert response.status_code == 200
    assert "phishing_probability" in body
    assert "explanation_keywords" in body


def test_invalid_input_returns_400(tmp_path):
    app, _container = create_test_app(tmp_path)
    response = post_json(app, "/analyze/text", {"text": ""})
    assert response.status_code == 400
