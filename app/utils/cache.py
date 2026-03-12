"""Simple in-memory TTL cache utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(slots=True)
class TTLCache(Generic[T]):
    """Tiny TTL cache suitable for lightweight inference memoization."""

    ttl_seconds: int
    _store: dict[str, tuple[float, T]] = field(default_factory=dict)

    def get(self, key: str) -> T | None:
        record = self._store.get(key)
        if record is None:
            return None
        expires_at, value = record
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: T) -> None:
        self._store[key] = (time.time() + self.ttl_seconds, value)

    def clear(self) -> None:
        self._store.clear()
