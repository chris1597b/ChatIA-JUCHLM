"""PredioService (§11,14,15,26). Lógica determinista; LLM jamás altera cifras."""
from __future__ import annotations
import logging

from core.exceptions import DatabaseUnavailableError
from security.validation import validate_nombre_completo

log = logging.getLogger("application")


def _formatear_predio(row: dict) -> str:
    def g(*keys):
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return row[k]
        return "—"
    nombre = g("nombre", "nombrecompleto", "nombres", "usuario")
    codigo = g("codigo_predio", "codigopredio", "codigo", "cod_predio")
    sector = g("sector", "zona", "ubicacion")
    area = g("area", "area_ha", "hectareas")
    estado = g("estado", "situacion")
    try:
        area_txt = f"{float(area):.2f} ha" if area != "—" else "—"
    except Exception:
        area_txt = str(area)
    # Plantilla estructurada backend (§15). Sin LLM.
    return (
        "🏠 Información del predio\n\n"
        f"👤 Usuario:\n{nombre}\n\n"
        f"📍 Sector:\n{sector}\n\n"
        f"🏷️ Código de predio:\n{codigo}\n\n"
        f"📐 Área:\n{area_txt}\n\n"
        f"✅ Estado:\n{estado}"
    )


def consultar(nombre: str, repository=None) -> dict:
    """Casos 0/1/N (§14). repository inyectable para tests."""
    nombre_ok = validate_nombre_completo(nombre)
    repo = repository or __import__("database.repositories.predio_repository",
                                    fromlist=["consultar_por_nombre"]).consultar_por_nombre
    try:
        filas = repo(nombre_ok)
    except DatabaseUnavailableError:
        raise
    except Exception:
        log.exception("predio consultar falló")
        raise DatabaseUnavailableError("Error consultando predio.")
    if not filas:
        return {"found": False, "count": 0, "data": None,
                "message": "No encontré información registrada para los datos proporcionados."}
    if len(filas) == 1:
        row = filas[0]
        return {"found": True, "count": 1,
                "data": {"nombre": row.get("nombre", nombre_ok),
                         "codigo_predio": str(row.get("codigo_predio", row.get("codigo", "—"))),
                         "sector": str(row.get("sector", "—")),
                         "area": row.get("area"), "estado": str(row.get("estado", "—")),
                         "raw": row},
                "message": _formatear_predio(row)}
    return {"found": True, "count": len(filas), "data": filas[:5],
            "message": ("Encontré más de un registro con ese nombre.\n"
                        "Necesito un dato adicional para identificar correctamente el predio "
                        "(p. ej. código de predio o sector).")}
