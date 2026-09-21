"""Observabilidad (§27): application/rag/sql/security/audit separados."""
from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGERS = ("application", "rag", "sql", "security", "audit")
_configured = False


def setup_logging(log_dir: str = "logs"):
    global _configured
    if _configured:
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for name in LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        if not lg.handlers:
            fh = RotatingFileHandler(Path(log_dir) / f"{name}.log", maxBytes=1_000_000,
                                     backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            lg.addHandler(fh)
            if name != "audit":  # audit no va a consola (privacidad)
                ch = logging.StreamHandler()
                ch.setFormatter(fmt)
                lg.addHandler(ch)
        lg.propagate = False
    # uvicorn hereda application
    logging.getLogger("uvicorn.error").handlers = logging.getLogger("application").handlers
    _configured = True
