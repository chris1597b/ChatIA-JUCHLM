"""Conexión SQL Server 2022 (§21). Solo parametrizado, usuario mínimos, read-only intent."""
from __future__ import annotations
import logging

from config.settings import get_settings
from core.exceptions import DatabaseUnavailableError

log = logging.getLogger("sql")


def get_connection():
    s = get_settings()
    if not s.sql_configured():
        raise DatabaseUnavailableError("SQL no configurado (.env).")
    try:
        import pyodbc
    except ImportError as e:
        log.error("pyodbc no instalado: %s", e)
        raise DatabaseUnavailableError("Driver SQL no disponible.")
    conn_str = (
        f"DRIVER={{{s.SQL_DRIVER}}};SERVER={s.SQL_SERVER};DATABASE={s.SQL_DATABASE};"
        f"UID={s.SQL_USERNAME};PWD={s.SQL_PASSWORD};"
        "Encrypt=yes;TrustServerCertificate=yes;"
        f"Connection Timeout={s.SQL_TIMEOUT};ApplicationIntent=ReadOnly;"
    )
    try:
        return pyodbc.connect(conn_str, timeout=s.SQL_TIMEOUT)
    except Exception as e:
        log.exception("SQL connect falló")
        raise DatabaseUnavailableError(str(e)[:200])


def health() -> dict:
    s = get_settings()
    if not s.sql_configured():
        return {"ok": False, "configured": False, "msg": "SQL no configurado (ver .env.example)"}
    try:
        cn = get_connection()
        cn.close()
        return {"ok": True, "configured": True}
    except DatabaseUnavailableError as e:
        return {"ok": False, "configured": True, "msg": str(e)[:200]}
