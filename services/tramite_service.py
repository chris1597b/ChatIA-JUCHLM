"""TrámiteService (§17). Whitelist estricta; jamás rutas del usuario.

El menú crece solo con config/capabilities.json (type=tramite): agregar la
capability + su entrada aquí + el PDF en documents/tramites/.
"""
from __future__ import annotations
import re
from pathlib import Path

from core.exceptions import TramiteNotFoundError, DomainValidationError
from security.validation import validate_capability_id

BASE_DIR = Path(__file__).resolve().parent.parent
TRAMITES_DIR = BASE_DIR / "documents" / "tramites"

# id capacidad / acción / alias -> archivo permitido (fuente de verdad filesystem)
WHITELIST = {
    "tramite_constancia_usuario": "constancia_usuario.pdf",
    "descargar_constancia_usuario": "constancia_usuario.pdf",
    "constancia_usuario": "constancia_usuario.pdf",
    "tramite_constancia_no_adeudo": "constancia_no_adeudo.pdf",
    "descargar_constancia_no_adeudo": "constancia_no_adeudo.pdf",
    "constancia_no_adeudo": "constancia_no_adeudo.pdf",
}

_SLUG_RE = re.compile(r"^[a-z0-9-]{3,80}$")


def _slug_of(filename: str) -> str:
    return Path(filename).stem.replace("_", "-")


def is_tramite(action_id: str) -> bool:
    try:
        return validate_capability_id(action_id) in WHITELIST
    except DomainValidationError:
        return False


def _label_for(action_id: str, filename: str) -> str:
    try:
        import services.menu_service as Menu
        for c in Menu.get_capabilities():
            if c.get("id") == action_id or c.get("action") == action_id:
                return c.get("label", filename)
            for child in c.get("children", []) or []:
                if child.get("id") == action_id or child.get("action") == action_id:
                    return child.get("label", filename)
    except Exception:
        pass
    return "📄 " + _slug_of(filename).replace("-", " ")


def describe(action_id: str) -> dict:
    """Info para el orchestrator: etiqueta, archivo, slug y URL de descarga."""
    cap = validate_capability_id(action_id)
    fname = WHITELIST.get(cap)
    if not fname:
        raise DomainValidationError("Trámite no disponible.")
    slug = _slug_of(fname)
    return {"cap_id": cap, "label": _label_for(cap, fname), "file": fname,
            "slug": slug, "download_url": f"/api/tramites/{slug}"}


def capability_for_slug(slug: str) -> dict | None:
    s = (slug or "").strip().lower()
    if not _SLUG_RE.match(s):
        return None
    try:
        import services.menu_service as Menu
        for c in Menu.get_capabilities():
            if c.get("type") == "tramite" and _slug_of(c.get("file", "")) == s:
                return c
    except Exception:
        pass
    return None


def is_available(action_id: str) -> bool:
    """¿Existe el PDF estático del trámite? Para avisar antes de pedir datos."""
    try:
        resolve_tramite(action_id)
        return True
    except (TramiteNotFoundError, DomainValidationError):
        return False


def esta_disponible(action_id: str) -> bool:
    """Disponible si hay PDF estático O si se genera al momento
    (constancia_pdf.TRAMITES). Sin esto, la guardia bloquearía el flujo
    generado aunque no necesite ningún archivo en disco."""
    if is_available(action_id):
        return True
    try:
        from services import constancia_pdf as CP
        return validate_capability_id(action_id) in CP.TRAMITES
    except (DomainValidationError, ImportError):
        return False


def resolve_tramite(action_id: str) -> Path:
    cap = validate_capability_id(action_id)
    fname = WHITELIST.get(cap)
    if not fname:
        raise DomainValidationError("Trámite no disponible.")
    return _resolve_file(fname)


def resolve_slug(slug: str) -> Path:
    s = (slug or "").strip().lower()
    if not _SLUG_RE.match(s):
        raise DomainValidationError("Trámite no disponible.")
    for fname in sorted(set(WHITELIST.values())):
        if _slug_of(fname) == s:
            return _resolve_file(fname)
    raise DomainValidationError("Trámite no disponible.")


def _resolve_file(fname: str) -> Path:
    # Resolución confinada al directorio (anti path-traversal §21)
    base = TRAMITES_DIR.resolve()
    target = (base / fname).resolve()
    if base not in target.parents and target != base:
        raise DomainValidationError("Ruta no autorizada.")
    if not target.is_file():
        raise TramiteNotFoundError(f"{fname} no disponible.")
    return target


def list_tramites() -> list[dict]:
    try:
        import services.menu_service as Menu
        out = []
        for c in Menu.get_tramite_menu():
            fname = WHITELIST.get(c.get("id", ""), "")
            out.append({"id": c.get("id"), "label": c.get("label"), "type": "download",
                        "available": bool(fname) and (TRAMITES_DIR / fname).is_file()})
        return out
    except Exception:
        return []
