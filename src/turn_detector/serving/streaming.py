"""Isolated state for incremental microphone inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
import time
import uuid

from turn_detector.models import StreamingWhisperFeatureBuffer, WaveformFeatureExtractor


@dataclass
class StreamingSession:
    session_id: str
    features: StreamingWhisperFeatureBuffer
    created_at: float
    last_used_at: float
    lock: Lock = field(default_factory=Lock)


class StreamingSessionStore:
    def __init__(
        self,
        feature_extractor: WaveformFeatureExtractor,
        *,
        capacity: int = 32,
        ttl_seconds: float = 300.0,
    ):
        self.feature_extractor = feature_extractor
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._lock = Lock()
        self._sessions: dict[str, StreamingSession] = {}

    def _purge_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_used_at > self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def create(self) -> StreamingSession:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            if len(self._sessions) >= self.capacity:
                raise RuntimeError("stream capacity reached")
            session_id = uuid.uuid4().hex
            session = StreamingSession(
                session_id=session_id,
                features=StreamingWhisperFeatureBuffer(self.feature_extractor),
                created_at=now,
                last_used_at=now,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> StreamingSession | None:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_used_at = now
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def snapshot(self) -> dict[str, float | int]:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            return {
                "active": len(self._sessions),
                "capacity": self.capacity,
                "ttl_seconds": self.ttl_seconds,
            }
