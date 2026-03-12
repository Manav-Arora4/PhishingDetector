"""Full pipeline integration test."""

from __future__ import annotations

from tests.conftest import create_test_app, post_json


def test_full_pipeline_blocks_known_phishing_pattern(tmp_path):
    app, _container = create_test_app(tmp_path)
    response = post_json(
        app,
        "/analyze/full",
        {
            "text": "Your account has been suspended, verify immediately",
            "url": "http://paypaI-secure-login.com/update",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["decision"] == "BLOCK"
    assert body["final_risk_score"] > 0.8
