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


# ---------- Padrón por DNI ----------

def _fila_dni(nombre="El Tablazo", codigo="CR-001", area=4.5):
    return {"apellido paterno": "Perez", "apellido materno": "Lopez",
            "nombres": "Juan", "nombre del predio": nombre, "area": area,
            "canal": "L1", "codigo de riego": codigo, "uc_actual": "UC-9",
            "regimen": "Licencia", "estado": "Activo", "comision": "Chancay"}


def test_dni_cero():
    out = P.consultar_por_dni("32104221", repository=lambda d: [])
    assert out["count"] == 0 and out["found"] is False


def test_dni_un_predio_formatea_sin_llm():
    out = P.consultar_por_dni("32104221", repository=lambda d: [_fila_dni()])
    assert out["count"] == 1
    assert "32104221" in out["message"] and "CR-001" in out["message"]
    assert "4.50 ha" in out["message"] and "Juan Perez Lopez" in out["message"]


def test_dni_multiples_lista_todos():
    out = P.consultar_por_dni("32104221",
                              repository=lambda d: [_fila_dni("A", "CR-1"), _fila_dni("B", "CR-2")])
    assert out["count"] == 2
    assert "Predios registrados (2)" in out["message"]
    assert "CR-1" in out["message"] and "CR-2" in out["message"]


def test_dni_invalido():
    with pytest.raises(DomainValidationError):
        P.consultar_por_dni("123", repository=lambda d: [])
    # tolera puntos/guiones al escribir
    out = P.consultar_por_dni("32.104.221", repository=lambda d: [])
    assert out["count"] == 0


def test_dni_db_error_es_seguro():
    def boom(d):
        raise DatabaseUnavailableError("down")
    with pytest.raises(DatabaseUnavailableError):
        P.consultar_por_dni("32104221", repository=boom)
