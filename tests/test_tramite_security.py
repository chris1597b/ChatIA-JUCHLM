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


def test_menu_tramites_crece_con_registry():
    items = T.list_tramites()
    ids = [i["id"] for i in items]
    assert "tramite_constancia_usuario" in ids
    assert "tramite_constancia_no_adeudo" in ids


def test_slug_seguro_y_generico(tmp_path, monkeypatch):
    (tmp_path / "constancia_no_adeudo.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)
    assert T.resolve_slug("constancia-no-adeudo").name == "constancia_no_adeudo.pdf"
    info = T.describe("tramite_constancia_no_adeudo")
    assert info["download_url"] == "/api/tramites/constancia-no-adeudo"
    with pytest.raises(DomainValidationError):
        T.resolve_slug("../../etc/passwd")
    with pytest.raises(DomainValidationError):
        T.resolve_slug("inexistente")
    assert T.is_tramite("tramite_constancia_no_adeudo") is True
    assert T.is_tramite("tramite_falso_xyz") is False
