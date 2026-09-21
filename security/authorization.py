"""Autorización futura (§22). MVP: todo público; interfaz lista para roles."""
from __future__ import annotations

ROLES = ("ADMIN", "FUNCIONARIO", "USUARIO")

PERMISSIONS = {
    "can_query_junta": ["ADMIN", "FUNCIONARIO", "USUARIO"],
    "can_query_predio": ["ADMIN", "FUNCIONARIO", "USUARIO"],
    "can_download_tramites": ["ADMIN", "FUNCIONARIO", "USUARIO"],
    "can_manage_documents": ["ADMIN", "FUNCIONARIO"],
}

CAPABILITY_PERMISSION = {
    "informacion_junta": "can_query_junta",
    "consulta_documental": "can_query_junta",
    "informacion_predio": "can_query_predio",
    "tramite_constancia_usuario": "can_download_tramites",
}


def can_access(capability_id: str, role: str = "USUARIO") -> bool:
    perm = CAPABILITY_PERMISSION.get(capability_id)
    if not perm:
        return True  # capacidades no mapeadas: permitir si están enabled (MVP)
    return role in PERMISSIONS.get(perm, [])
