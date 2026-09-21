"""TrámiteService (§17). Whitelist estricta; jamás rutas del usuario."""
from __future__ import annotations
from pathlib import Path

from core.exceptions import TramiteNotFoundError, DomainValidationError
from security.validation import validate_capability_id

BASE_DIR = Path(__file__).resolve().parent.parent
TRAMITES_DIR = BASE_DIR / "documents" / "tramites"

# id capacidad -> archivo permitido (única fuente de verdad filesystem)
WHITELIST = {
    "tramite_constancia_usuario": "constancia_usuario.pdf",
    "descargar_constancia_usuario": "constancia_usuario.pdf",
    "constancia_usuario": "constancia_usuario.pdf",
}


def resolve_tramite(action_id: str) -> Path:
    cap = validate_capability_id(action_id)
    fname = WHITELIST.get(cap)
    if not fname:
        raise DomainValidationError("Trámite no disponible.")
    # Resolución confinada al directorio (anti path-traversal §21)
    base = TRAMITES_DIR.resolve()
    target = (base / fname).resolve()
    if base not in target.parents and target != base:
        raise DomainValidationError("Ruta no autorizada.")
    if not target.is_file():
        raise TramiteNotFoundError(f"{fname} no disponible.")
    return target


def list_tramites() -> list[dict]:
    return [{"id": "tramite_constancia_usuario", "label": "📄 Constancia de usuario", "type": "download",
             "available": (TRAMITES_DIR / "constancia_usuario.pdf").is_file()}]
