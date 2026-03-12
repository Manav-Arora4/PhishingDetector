"""Compatibility helpers for FastAPI and offline fallback dispatch."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.security.rate_limiter import RateLimitExceeded, SlidingWindowRateLimiter


try:  # pragma: no cover - exercised only when FastAPI is installed.
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by fallback tests.
    FASTAPI_AVAILABLE = False

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    @dataclass(slots=True)
    class JSONResponse:
        content: dict[str, Any]
        status_code: int = 200


@dataclass(slots=True)
class DispatchResponse:
    """Minimal response wrapper used by the offline API shim."""

    status_code: int
    body: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.body


class SimpleFastAPI:
    """Fallback API container used when FastAPI is unavailable."""

    def __init__(self, title: str, rate_limiter: SlidingWindowRateLimiter) -> None:
        self.title = title
        self.rate_limiter = rate_limiter
        self.routes: dict[tuple[str, str], Callable[..., Awaitable[dict[str, Any]]]] = {}

    def post(self, path: str) -> Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]]:
        def decorator(func: Callable[..., Awaitable[dict[str, Any]]]) -> Callable[..., Awaitable[dict[str, Any]]]:
            self.routes[("POST", path)] = func
            return func

        return decorator

    async def dispatch(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        client_host: str = "127.0.0.1",
    ) -> DispatchResponse:
        handler = self.routes.get((method.upper(), path))
        if handler is None:
            return DispatchResponse(status_code=404, body={"detail": "Not found."})
        try:
            self.rate_limiter.check_or_raise(client_host)
            kwargs = {"payload": payload or {}}
            if "client_host" in inspect.signature(handler).parameters:
                kwargs["client_host"] = client_host
            result = await handler(**kwargs)
            return DispatchResponse(status_code=200, body=result)
        except HTTPException as exc:
            return DispatchResponse(status_code=exc.status_code, body={"detail": exc.detail})
        except RateLimitExceeded as exc:
            return DispatchResponse(status_code=429, body={"detail": str(exc)})


def build_api_app(title: str, rate_limiter: SlidingWindowRateLimiter) -> FastAPI | SimpleFastAPI:
    """Create a real FastAPI app when available, otherwise fall back to a small shim."""

    if not FASTAPI_AVAILABLE:
        return SimpleFastAPI(title=title, rate_limiter=rate_limiter)

    app = FastAPI(title=title)

    @app.middleware("http")  # pragma: no cover - depends on FastAPI.
    async def rate_limit_middleware(request, call_next):
        try:
            client_host = request.client.host if request.client else "127.0.0.1"
            rate_limiter.check_or_raise(client_host)
        except RateLimitExceeded as exc:
            return JSONResponse(content={"detail": str(exc)}, status_code=429)
        return await call_next(request)

    return app
