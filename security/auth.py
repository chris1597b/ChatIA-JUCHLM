"""Auth stub (§22). MVP sin login; deja interfaz para JWT futuro."""
from __future__ import annotations
from fastapi import Header


def get_current_role(x_role: str | None = Header(default=None, alias="X-Role")) -> str:
    role = (x_role or "USUARIO").upper()
    return role if role in ("ADMIN", "FUNCIONARIO", "USUARIO") else "USUARIO"
