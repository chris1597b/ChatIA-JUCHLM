"""Session State (§9). Controlado por backend, no por el texto del LLM."""
from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field

from core.models import ConversationState


@dataclass
class Session:
    session_id: str
    state: ConversationState = ConversationState.START
    context: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class InMemorySessionManager:
    """MVP: memoria local thread-safe con TTL. Migrar a Redis/Postgres sin cambiar interfaz."""

    def __init__(self, ttl_seconds: int = 1800):
        self._store: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            self._purge_locked()
            if session_id and session_id in self._store:
                s = self._store[session_id]
                s.updated_at = time.time()
                return s
            sid = session_id or uuid.uuid4().hex[:16]
            s = Session(session_id=sid, state=ConversationState.START)
            self._store[sid] = s
            return s

    def set_state(self, session_id: str, state: ConversationState) -> Session:
        s = self.get_or_create(session_id)
        with self._lock:
            s.state = state
            s.updated_at = time.time()
        return s

    def update_context(self, session_id: str, **kwargs) -> Session:
        s = self.get_or_create(session_id)
        with self._lock:
            s.context.update(kwargs)
            s.updated_at = time.time()
        return s

    def push_history(self, session_id: str, role: str, text: str):
        s = self.get_or_create(session_id)
        with self._lock:
            s.history.append({"role": role, "text": text[:1000]})
            s.history = s.history[-20:]
            s.updated_at = time.time()

    def reset(self, session_id: str) -> Session:
        with self._lock:
            sid = session_id or uuid.uuid4().hex[:16]
            s = Session(session_id=sid, state=ConversationState.START)
            self._store[sid] = s
            return s

    def _purge_locked(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v.updated_at > self._ttl]
        for k in expired:
            del self._store[k]


# Singleton de aplicación (inyectable en tests)
session_manager = InMemorySessionManager()
