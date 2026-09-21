"""Tests Trámites (§29): whitelist, traversal bloqueado, faltante 404 seguro."""
import pytest
from services import tramite_service as T
from core.exceptions import DomainValidationError, TramiteNotFoundError


def test_traversal_bloqueado():
    with pytest.raises(DomainValidationError):
        T.resolve_tramite("../../etc/passwd")
    with pytest.raises(DomainValidationError):
        T.resolve_tramite("../../../windows/win.ini")


def test_id_desconocido_rechazado():
    with pytest.raises(DomainValidationError):
        T.resolve_tramite("tramite_falso_xyz")


def test_faltante_retorna_404_seguro(tmp_path, monkeypatch):
    # sin archivo real debe dar TramiteNotFoundError, nunca path interno al usuario
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)
    with pytest.raises(TramiteNotFoundError):
        T.resolve_tramite("constancia_usuario")
