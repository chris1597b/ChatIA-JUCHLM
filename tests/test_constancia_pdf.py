"""Tests constancia generada: plantilla, datos, X del trámite, fecha."""
from datetime import datetime
from services import constancia_pdf as CP


def _datos():
    return {"nombre": "Segura Salinas Marithza Marleni", "dni": "32104221", "celular": "987654321"}


def _predio():
    return {"apellido paterno": "Segura", "apellido materno": "Salinas",
            "nombres": "Marithza Marleni", "nombre del predio": "EL COCO",
            "area": 0.5, "canal": "TAYMI/4 DE MAYO/PAÑO II", "codigo de riego": "FE4M2D36",
            "uc_actual": "1", "regimen": "LICENCIA", "estado": "Activo", "comision": "Mochumi"}


def _texto(pdf: bytes) -> str:
    import re
    from pypdf import PdfReader
    import io
    crudo = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf)).pages)
    return re.sub(r"\s+", " ", crudo)


def test_genera_pdf_con_datos():
    pdf = CP.generar_constancia_pdf(_datos(), _predio(), "tramite_constancia_usuario",
                                    datetime(2026, 9, 28))
    assert pdf[:4] == b"%PDF"
    from pypdf import PdfReader
    import io
    assert len(PdfReader(io.BytesIO(pdf)).pages) == 1, "el formato debe caber en una hoja"
    txt = _texto(pdf)
    assert "SEGURA SALINAS MARITHZA MARLENI" in txt
    assert "32104221" in txt and "987654321" in txt
    assert "EL COCO" in txt and "FE4M2D36" in txt
    assert "28 de septiembre de 2026" in txt
    assert "CONSTANCIA DE RED DE RIEGO" in txt


def test_x_marca_el_tramite_pedido():
    pdf_riego = CP.generar_constancia_pdf(_datos(), _predio(), "tramite_constancia_usuario")
    pdf_adeudo = CP.generar_constancia_pdf(_datos(), _predio(), "tramite_constancia_no_adeudo")
    assert "CONSTANCIA DE NO ADEUDO" in _texto(pdf_adeudo)
    assert pdf_riego != pdf_adeudo  # la X cambia de casilla


def test_tramite_invalido_rechazado():
    try:
        CP.generar_constancia_pdf(_datos(), _predio(), "tramite_falso")
        assert False, "debió fallar"
    except ValueError:
        pass
