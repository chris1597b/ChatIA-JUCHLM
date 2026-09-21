"""Validación de parámetros (§21). Toda entrada pasa por aquí antes de SQL/archivos."""
from __future__ import annotations
import re

from core.exceptions import DomainValidationError

_NOMBRE_RE = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'.\- ]{3,100}$")
_CAP_RE = re.compile(r"^[a-z0-9_]{3,64}$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")


def validate_nombre_completo(nombre: str) -> str:
    nombre = (nombre or "").strip()
    # collapse espacios
    nombre = re.sub(r"\s+", " ", nombre)
    if len(nombre) < 5 or len(nombre) > 100:
        raise DomainValidationError("Indícame tu nombre y apellido completo (5 a 100 caracteres).")
    if len(nombre.split()) < 2:
        raise DomainValidationError("Por favor indica nombre y apellido completos.")
    if not _NOMBRE_RE.match(nombre):
        raise DomainValidationError("El nombre contiene caracteres no permitidos.")
    return nombre


def validate_capability_id(cap_id: str) -> str:
    cap_id = (cap_id or "").strip()
    if not _CAP_RE.match(cap_id):
        raise DomainValidationError("Acción no válida.")
    return cap_id


def validate_safe_filename(name: str) -> str:
    if not name or ".." in name or "/" in name or "\\" in name:
        raise DomainValidationError("Nombre de archivo no permitido.")
    if not _SAFE_FILENAME_RE.match(name):
        raise DomainValidationError("Nombre de archivo no permitido.")
    return name
