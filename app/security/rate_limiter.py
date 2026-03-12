"""Rate limiting helpers."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


class RateLimitExceeded(RuntimeError):
    """Raised when a client exceeds the configured request quota."""


@dataclass(slots=True)
class SlidingWindowRateLimiter:
    """Simple in-memory sliding window limiter."""

    max_requests: int
    window_seconds: int
    _events: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def check_or_raise(self, client_id: str) -> None:
        now = time.time()
        bucket = self._events[client_id]
        while bucket and bucket[0] <= now - self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            raise RateLimitExceeded("Rate limit exceeded.")
        bucket.append(now)
