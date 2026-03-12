"""NLP phishing model tests."""

from __future__ import annotations

from app.ml.training.model_backends import TokenNaiveBayesClassifier


def test_benign_message_returns_low_score():
    model = TokenNaiveBayesClassifier.bootstrap_default()
    score = model.predict_proba(["Can we reschedule the project review to Friday morning?"])[0]
    assert score < 0.5


def test_phishing_message_returns_high_score():
    model = TokenNaiveBayesClassifier.bootstrap_default()
    score = model.predict_proba(["Your account has been suspended, verify immediately to restore access."])[0]
    assert score > 0.7
