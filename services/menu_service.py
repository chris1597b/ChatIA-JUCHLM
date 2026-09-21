"""Capability Registry (§6) + Menús multinivel (§7). Backend expone, frontend renderiza."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

CAPABILITIES_PATH = Path(__file__).resolve().parent.parent / "config" / "capabilities.json"


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    try:
        return json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"capabilities": []}


def get_capabilities() -> list[dict]:
    return [c for c in _load_registry().get("capabilities", []) if c.get("enabled", True)]


def get_main_menu() -> list[dict]:
    """Solo top-level para MVP: junta + predio. Trámites se muestran tras predio (§16)."""
    caps = get_capabilities()
    return [c for c in caps if c["id"] in ("informacion_junta", "informacion_predio")]


def get_tramite_menu() -> list[dict]:
    return [c for c in get_capabilities() if c.get("type") == "tramite"]


def find_capability(cap_id: str) -> dict | None:
    if not cap_id:
        return None
    for c in get_capabilities():
        if c.get("id") == cap_id:
            return c
        for child in c.get("children", []) or []:
            if child.get("id") == cap_id or child.get("action") == cap_id:
                return child
    return None


def is_valid_action(cap_id: str) -> bool:
    return find_capability(cap_id) is not None
