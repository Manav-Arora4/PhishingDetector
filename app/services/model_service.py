"""Model inference service."""

from __future__ import annotations

from app.ml.training.model_backends import load_or_bootstrap_model
from app.utils.config import Settings


class ModelService:
    """Lazy model loader for text-based phishing inference."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_or_bootstrap_model(self.settings.model_artifact_path, settings=self.settings)
        return self._model

    def analyze_text(self, text: str) -> dict[str, object]:
        probability = self.model.predict_proba([text])[0]
        return {
            "phishing_probability": probability,
            "explanation_keywords": self.model.explain(text),
            "backend": self.model.backend_name,
        }

    def refresh(self) -> None:
        self._model = None
