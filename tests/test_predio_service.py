"""Tests SQL (§29): 0 / 1 / N / parámetro inválido / DB Caída."""
import pytest
from services import predio_service as P
from core.exceptions import DomainValidationError, DatabaseUnavailableError


def test_cero_resultados():
    out = P.consultar("Juan Perez Lopez", repository=lambda n: [])
    assert out["count"] == 0 and out["found"] is False


def test_un_resultado_formatea_sin_llm():
    row = {"nombre": "Juan Pérez López", "codigo_predio": "P001245",
           "sector": "Santo Domingo", "area": 4.5, "estado": "Activo"}
    out = P.consultar("Juan Perez Lopez", repository=lambda n: [row])
    assert out["count"] == 1
    assert "P001245" in out["message"] and "Santo Domingo" in out["message"]
    assert "4.50 ha" in out["message"]


def test_multiples_pide_desambiguacion():
    rows = [{"nombre": "Juan Perez", "codigo_predio": "P1"}, {"nombre": "Juan Perez", "codigo_predio": "P2"}]
    out = P.consultar("Juan Perez Lopez", repository=lambda n: rows)
    assert out["count"] == 2
    assert "dato adicional" in out["message"]


def test_nombre_invalido():
    with pytest.raises(DomainValidationError):
        P.consultar("X", repository=lambda n: [])


def test_db_error_es_seguro():
    def boom(n):
        raise DatabaseUnavailableError("down")
    with pytest.raises(DatabaseUnavailableError):
        P.consultar("Juan Perez Lopez", repository=boom)
