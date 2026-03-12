"""Application entry point."""

from __future__ import annotations

from app.api.routes import register_routes
from app.services.container import ServiceContainer, build_service_container
from app.utils.api_compat import build_api_app


def create_app(container: ServiceContainer | None = None):
    """Create the phishing detection API application."""

    resolved = container or build_service_container()
    app = build_api_app(
        title="Real-Time AI/ML-Based Phishing Detection and Prevention System",
        rate_limiter=resolved.rate_limiter,
    )
    register_routes(
        app,
        analysis_service=resolved.analysis_service,
        feedback_service=resolved.feedback_service,
        settings=resolved.settings,
    )
    return app


app = create_app()
