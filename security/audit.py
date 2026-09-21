"""Auditoría (§23). Hash de parámetros sensibles, nunca plaintext innecesario."""
from __future__ import annotations
import hashlib
import json
import logging
import time

audit_log = logging.getLogger("audit")


def _hash_params(params: dict) -> str:
    try:
        raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "n/a"


def audit(session_id: str, action: str, capability: str = "", params: dict | None = None, status: str = "ok"):
    audit_log.info(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": session_id,
        "action": action,
        "capability": capability,
        "parameters_hash": _hash_params(params or {}),
        "result_status": status,
    }, ensure_ascii=False))
