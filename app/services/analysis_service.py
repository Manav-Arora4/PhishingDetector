"""End-to-end phishing analysis orchestration."""

from __future__ import annotations

from app.services.feedback_service import FeedbackService
from app.services.model_service import ModelService
from app.services.scoring_service import ScoringService
from app.security.url_analyzer import URLAnalysisEngine
from app.security.web_analyzer import StructuralWebAnalyzer


class AnalysisService:
    """Coordinate text, URL, and structural analyzers."""

    def __init__(
        self,
        *,
        model_service: ModelService,
        url_analyzer: URLAnalysisEngine,
        web_analyzer: StructuralWebAnalyzer,
        scoring_service: ScoringService,
        feedback_service: FeedbackService,
    ) -> None:
        self.model_service = model_service
        self.url_analyzer = url_analyzer
        self.web_analyzer = web_analyzer
        self.scoring_service = scoring_service
        self.feedback_service = feedback_service

    async def analyze_text(self, text: str) -> dict[str, object]:
        return self.model_service.analyze_text(text)

    async def analyze_url(self, url: str) -> dict[str, object]:
        report = await self.url_analyzer.analyze(url)
        return report.as_dict()

    async def analyze_full(self, *, text: str, url: str, html_content: str | None = None) -> dict[str, object]:
        nlp = self.model_service.analyze_text(text)
        url_report = await self.url_analyzer.analyze(url)
        structural_report = await self.web_analyzer.analyze(url, html_content=html_content)
        fused = self.scoring_service.combine(
            nlp_score=float(nlp["phishing_probability"]),
            url_score=url_report.final_risk_score,
            structural_score=structural_report.final_risk_score,
        )
        explanation = {
            **fused["explanation"],
            "nlp": nlp,
            "url": url_report.as_dict(),
            "structural": structural_report.as_dict(),
        }
        prediction_id = self.feedback_service.log_prediction(
            text=text,
            url=url,
            decision=str(fused["decision"]),
            risk_score=float(fused["final_risk_score"]),
            explanation=explanation,
        )
        return {
            "final_risk_score": fused["final_risk_score"],
            "decision": fused["decision"],
            "explanation": explanation,
            "prediction_id": prediction_id,
        }
