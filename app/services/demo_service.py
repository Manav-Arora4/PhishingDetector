"""Demo data and inference helpers for the local showcase frontend."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.model_service import ModelService
from app.security.url_analyzer import URLAnalysisEngine
from app.utils.config import Settings


@dataclass(slots=True)
class DemoSample:
    """Serializable held-out sample used by the local demo UI."""

    sample_id: int
    text: str
    label: int
    source_file: str
    subject: str
    urls: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "label": self.label,
            "source_file": self.source_file,
            "subject": self.subject,
            "urls": self.urls,
        }


class DemoService:
    """Load held-out samples and score them for the local showcase UI."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_service: ModelService | None = None,
        url_analyzer: URLAnalysisEngine | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.model_service = model_service or ModelService(self.settings)
        self.url_analyzer = url_analyzer or URLAnalysisEngine(self.settings)

    def load_test_samples(self) -> list[DemoSample]:
        """Load persisted test-set samples from the latest training run."""

        path = self.settings.test_samples_path
        if not path.exists():
            raise FileNotFoundError(
                f"Test samples artifact not found at {path}. Run base training first."
            )
        samples: list[DemoSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                payload = json.loads(line)
                samples.append(
                    DemoSample(
                        sample_id=index,
                        text=str(payload.get("text", "")),
                        label=int(payload.get("label", 0)),
                        source_file=str(payload.get("source_file", "")),
                        subject=str(payload.get("subject", "")),
                        urls=list(payload.get("urls", [])),
                    )
                )
        return samples

    def get_random_sample(self) -> dict[str, Any]:
        """Select a random held-out sample for demonstration."""

        samples = self.load_test_samples()
        sample = random.choice(samples)
        return sample.as_dict()

    async def infer_sample(self, sample_id: int) -> dict[str, Any]:
        """Run inference for a held-out sample and compare it with the ground truth label."""

        samples = self.load_test_samples()
        if sample_id < 0 or sample_id >= len(samples):
            raise IndexError("Sample id is out of range.")
        sample = samples[sample_id]
        nlp = self.model_service.analyze_text(sample.text)
        predicted_label = 1 if float(nlp["phishing_probability"]) >= 0.5 else 0
        url_report = None
        if sample.urls:
            url_report = (await self.url_analyzer.analyze(sample.urls[0])).as_dict()
        return {
            "sample": sample.as_dict(),
            "prediction": {
                "predicted_label": predicted_label,
                "predicted_label_name": "phishing" if predicted_label == 1 else "benign",
                "actual_label_name": "phishing" if sample.label == 1 else "benign",
                "is_correct": predicted_label == sample.label,
                "phishing_probability": nlp["phishing_probability"],
                "explanation_keywords": nlp["explanation_keywords"],
                "backend": nlp["backend"],
                "url_analysis": url_report,
            },
        }
