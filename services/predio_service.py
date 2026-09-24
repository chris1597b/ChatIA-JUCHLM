"""PredioService (§11,14,15,26). Lógica determinista; LLM jamás altera cifras."""
from __future__ import annotations
import logging

from core.exceptions import DatabaseUnavailableError
from security.validation import validate_codigo_riego, validate_dni, validate_nombre_completo

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


# ---------- Padrón por DNI (flujo real) ----------

def _mostrar(v) -> str:
    """El SP usa 'No encontrado' como centinela de NULL; en pantalla se ve '—'."""
    if v is None:
        return "—"
    s = str(v).strip()
    return "—" if (not s or s.lower() == "no encontrado") else s


def _titular(fila: dict) -> str:
    partes = [_mostrar(fila.get(k)) for k in ("nombres", "apellido paterno", "apellido materno")]
    return " ".join(p for p in partes if p != "—") or "—"


def _formatear_predio_dni(fila: dict, idx: int) -> str:
    area = fila.get("area")
    try:
        area_txt = f"{float(area):.2f} ha" if area is not None else "—"
    except (TypeError, ValueError):
        area_txt = _mostrar(area)
    estado = _mostrar(fila.get("estado"))
    punto = "🟢" if estado.lower() == "activo" else ("🔴" if estado.lower() == "desactivo" else "⚪")
    # Tarjeta por predio: un campo por fila.
    return (
        f"🌾 Predio {idx} — {_mostrar(fila.get('nombre del predio'))}\n"
        f"🏷️ Código de riego: {_mostrar(fila.get('codigo de riego'))}\n"
        f"📐 Área: {area_txt}\n"
        f"🚰 Canal: {_mostrar(fila.get('canal'))}\n"
        f"🏛️ Comisión: {_mostrar(fila.get('comision'))}\n"
        f"📋 Régimen: {_mostrar(fila.get('regimen'))}\n"
        f"🔖 UC: {_mostrar(fila.get('uc_actual'))}\n"
        f"{punto} Estado: {estado}"
    )


def _listar_predios(filas: list, linea_id: str, data: dict, sin_resultados: str) -> dict:
    """N filas = N predios del titular: se listan todos (no es ambigüedad)."""
    if not filas:
        return {"found": False, "count": 0, "data": None, "message": sin_resultados}
    titular = _titular(filas[0])
    data = {**data, "titular": titular, "predios": filas}
    bloque = "\n\n".join(_formatear_predio_dni(f, i) for i, f in enumerate(filas, 1))
    titulo = "Predio registrado (1)" if len(filas) == 1 else f"Predios registrados ({len(filas)})"
    return {"found": True, "count": len(filas), "data": data,
            "message": f"🏠 {titulo}\n\n👤 Titular:\n{titular}\n{linea_id}\n\n{bloque}"}


def _ejecutar_consulta(valor_ok: str, repo, etiqueta: str) -> list:
    try:
        return repo(valor_ok)
    except DatabaseUnavailableError:
        raise
    except Exception:
        log.exception(f"predio {etiqueta} falló")
        raise DatabaseUnavailableError("Error consultando predio.")


def consultar_por_dni(dni: str, repository=None) -> dict:
    """Casos 0 / 1..N. repository inyectable para tests."""
    dni_ok = validate_dni(dni)
    repo = repository or __import__("database.repositories.predio_repository",
                                    fromlist=["consultar_por_dni"]).consultar_por_dni
    filas = _ejecutar_consulta(dni_ok, repo, "consultar_por_dni")
    return _listar_predios(filas, f"🪪 DNI: {dni_ok}", {"dni": dni_ok},
                           "No encontré predios registrados para ese DNI.")


def consultar_por_codigo(codigo: str, repository=None) -> dict:
    """Casos 0 / 1..N por código de riego. repository inyectable para tests."""
    cod_ok = validate_codigo_riego(codigo)
    repo = repository or __import__("database.repositories.predio_repository",
                                    fromlist=["consultar_por_codigo"]).consultar_por_codigo
    filas = _ejecutar_consulta(cod_ok, repo, "consultar_por_codigo")
    return _listar_predios(filas, f"🏷️ Código: {cod_ok}", {"codigo": cod_ok},
                           "No encontré predios registrados para ese código de riego.")
