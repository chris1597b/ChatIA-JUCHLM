"""PredioRepository (§11-13). Abstrae SP; devuelve dicts, nunca strings SQL."""
from __future__ import annotations
import logging

from database.procedures import SP_CONSULTAR_PREDIO_POR_NOMBRE, SP_OBTENER_PREDIOS_POR_DNI
from database.sqlserver import get_connection

log = logging.getLogger("sql")


def consultar_por_nombre(nombre: str) -> list[dict]:
    """Parametrizado. Nunca f-string. Retorna lista de filas como dicts."""
    cn = get_connection()
    try:
        cur = cn.cursor()
        # pyodbc parametriza el ? — el nombre viaja como parámetro, no como SQL
        cur.execute(SP_CONSULTAR_PREDIO_POR_NOMBRE, (nombre,))
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall() if cols else []
        out = []
        for r in rows:
            d = {cols[i]: r[i] for i in range(len(cols))}
            out.append({k.lower(): v for k, v in d.items()})
        return out
    except Exception:
        log.exception("SP consultar_predio falló")
        raise
    finally:
        try:
            cn.close()
        except Exception:
            pass


def consultar_por_dni(dni: str) -> list[dict]:
    """Padrón real por DNI. Parametrizado (?), nunca f-string."""
    return _ejecutar_padron(dni, None)


def consultar_por_codigo(codigo: str) -> list[dict]:
    """Padrón real por código de riego (@Dni va NULL)."""
    return _ejecutar_padron(None, codigo)


def _ejecutar_padron(dni: str | None, codigo: str | None) -> list[dict]:
    """Una fila por predio; columnas con espacios ('codigo de riego')."""
    cn = get_connection()
    try:
        cur = cn.cursor()
        cur.execute(SP_OBTENER_PREDIOS_POR_DNI, (dni, codigo))
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall() if cols else []
        out = []
        for r in rows:
            d = {cols[i]: r[i] for i in range(len(cols))}
            out.append({str(k).strip().lower(): v for k, v in d.items()})
        return out
    except Exception:
        log.exception("SP obtener_predios_por_dni falló")
        raise
    finally:
        try:
            cn.close()
        except Exception:
            pass
