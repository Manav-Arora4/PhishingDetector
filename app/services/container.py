"""Dependency injection container for the phishing platform."""

from __future__ import annotations

from dataclasses import dataclass

from app.security.rate_limiter import SlidingWindowRateLimiter
from app.security.url_analyzer import URLAnalysisEngine
from app.security.web_analyzer import StructuralWebAnalyzer
from app.services.analysis_service import AnalysisService
from app.services.feedback_service import FeedbackService
from app.services.model_service import ModelService
from app.services.scoring_service import ScoringService
from app.utils.config import Settings, ensure_runtime_dirs


@dataclass(slots=True)
class ServiceContainer:
    """Resolved service dependencies."""

    settings: Settings
    rate_limiter: SlidingWindowRateLimiter
    feedback_service: FeedbackService
    model_service: ModelService
    url_analyzer: URLAnalysisEngine
    web_analyzer: StructuralWebAnalyzer
    scoring_service: ScoringService
    analysis_service: AnalysisService


def build_service_container(settings: Settings | None = None) -> ServiceContainer:
    """Instantiate the project's service graph."""

    effective = ensure_runtime_dirs(settings)
    feedback_service = FeedbackService(effective.feedback_db_path)
    model_service = ModelService(effective)
    url_analyzer = URLAnalysisEngine(effective)
    web_analyzer = StructuralWebAnalyzer(effective)
    scoring_service = ScoringService()
    analysis_service = AnalysisService(
        model_service=model_service,
        url_analyzer=url_analyzer,
        web_analyzer=web_analyzer,
        scoring_service=scoring_service,
        feedback_service=feedback_service,
    )
    return ServiceContainer(
        settings=effective,
        rate_limiter=SlidingWindowRateLimiter(
            max_requests=effective.rate_limit_requests,
            window_seconds=effective.rate_limit_window_seconds,
        ),
        feedback_service=feedback_service,
        model_service=model_service,
        url_analyzer=url_analyzer,
        web_analyzer=web_analyzer,
        scoring_service=scoring_service,
        analysis_service=analysis_service,
    )
