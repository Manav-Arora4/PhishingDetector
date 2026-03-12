"""Route registration for phishing analysis APIs."""

from __future__ import annotations

from app.models.schemas import FeedbackRequest, FullAnalysisRequest, TextAnalysisRequest, URLAnalysisRequest, ValidationError
from app.services.analysis_service import AnalysisService
from app.services.feedback_service import FeedbackService
from app.utils.api_compat import HTTPException
from app.utils.config import Settings


def register_routes(app, *, analysis_service: AnalysisService, feedback_service: FeedbackService, settings: Settings) -> None:
    """Bind all API routes to the provided application instance."""

    @app.post("/analyze/text")
    async def analyze_text(payload: dict, client_host: str = "127.0.0.1") -> dict:
        del client_host
        try:
            request = TextAnalysisRequest.validate(payload, max_length=settings.max_text_length)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await analysis_service.analyze_text(request.text)

    @app.post("/analyze/url")
    async def analyze_url(payload: dict, client_host: str = "127.0.0.1") -> dict:
        del client_host
        try:
            request = URLAnalysisRequest.validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await analysis_service.analyze_url(request.url)

    @app.post("/analyze/full")
    async def analyze_full(payload: dict, client_host: str = "127.0.0.1") -> dict:
        del client_host
        try:
            request = FullAnalysisRequest.validate(payload, max_text_length=settings.max_text_length)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await analysis_service.analyze_full(
            text=request.text,
            url=request.url,
            html_content=request.html_content,
        )

    @app.post("/feedback")
    async def submit_feedback(payload: dict, client_host: str = "127.0.0.1") -> dict:
        del client_host
        try:
            request = FeedbackRequest.validate(payload, max_text_length=settings.max_text_length)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        feedback_id = feedback_service.record_feedback(
            prediction_id=request.prediction_id,
            text=request.text,
            url=request.url,
            user_label=request.user_label,
            notes=request.notes,
        )
        return {"feedback_id": feedback_id, "status": "accepted"}
