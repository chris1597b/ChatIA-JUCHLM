"""Generación del formato institucional en PDF (reportlab, 100% local).

Amarillo (solicitante): apellidos y nombres, DNI, celular.
Verde (predio elegido de la consulta SQL): usuario, nombre, área, canal, código, comisión.
Fucsia: fecha actual de descarga. La X marca el trámite solicitado.
"""
from __future__ import annotations
import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, HRFlowable)

ANIO_FRASE = "Año de la recuperación y consolidación de la economía peruana"

CHECKLIST = [
    "CONSTANCIA DE RED DE RIEGO",
    "CONSTANCIA DE NO ADEUDO",
    "TARJETAS DE CONSUMO.",
    "CONSTANCIA DE USUARIO",
    "CONSTANCIA DE HABILIDAD",
]

# tramite_id -> constancia que nombra el cuerpo + casilla a marcar
TRAMITES = {
    "tramite_constancia_usuario": {"constancia": "CONSTANCIA DE RED DE RIEGO", "check": 0},
    "descargar_constancia_usuario": {"constancia": "CONSTANCIA DE RED DE RIEGO", "check": 0},
    "tramite_constancia_no_adeudo": {"constancia": "CONSTANCIA DE NO ADEUDO", "check": 1},
    "descargar_constancia_no_adeudo": {"constancia": "CONSTANCIA DE NO ADEUDO", "check": 1},
}

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

TRAMITE_POR_SLUG = {
    "constancia-usuario": "tramite_constancia_usuario",
    "constancia-no-adeudo": "tramite_constancia_no_adeudo",
}


def _limpio(fila: dict, clave: str, default: str = "—") -> str:
    v = fila.get(clave)
    s = "" if v is None else str(v).strip()
    return default if (not s or s.lower() == "no encontrado") else s


def _usuario_mayus(fila: dict) -> str:
    partes = [_limpio(fila, "apellido paterno", ""), _limpio(fila, "apellido materno", ""),
              _limpio(fila, "nombres", "")]
    nombre = " ".join(p for p in partes if p and p != "—")
    return nombre.upper() or "—"


def _area_txt(fila: dict) -> str:
    try:
        return f"{float(fila.get('area')):.2f} Ha"
    except (TypeError, ValueError):
        return _limpio(fila, "area")


def nombre_archivo(tramite_id: str, dni: str) -> str:
    base = {"tramite_constancia_usuario": "constancia_red_riego",
            "descargar_constancia_usuario": "constancia_red_riego",
            "tramite_constancia_no_adeudo": "constancia_no_adeudo",
            "descargar_constancia_no_adeudo": "constancia_no_adeudo"}.get(tramite_id, "constancia")
    d = "".join(c for c in str(dni or "") if c.isdigit()) or "sindni"
    return f"{base}_{d}.pdf"


def generar_constancia_pdf(datos: dict, predio: dict, tramite_id: str,
                           fecha: datetime | None = None) -> bytes:
    """datos: {nombre, dni, celular}. predio: fila SQL en minúsculas."""
    if tramite_id not in TRAMITES:
        raise ValueError("Trámite no soportado para generación.")
    cfg = TRAMITES[tramite_id]
    fecha = fecha or datetime.now()

    nombre_sol = str(datos.get("nombre", "")).strip().upper() or "—"
    dni_sol = "".join(c for c in str(datos.get("dni", "")) if c.isdigit()) or "—"
    cel_sol = str(datos.get("celular", "")).strip() or "—"

    titular = _usuario_mayus(predio)
    comision = _limpio(predio, "comision").upper()
    nombre_predio = _limpio(predio, "nombre del predio").upper()
    area = _area_txt(predio).upper().replace("HA", "Ha")
    canal = _limpio(predio, "canal").upper()
    codigo = _limpio(predio, "codigo de riego").upper()

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                          topMargin=2 * cm, bottomMargin=2 * cm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="m")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame])])

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=11, leading=15, alignment=TA_JUSTIFY)
    bold = ParagraphStyle("bold", parent=base, fontName="Helvetica-Bold")
    center = ParagraphStyle("center", parent=base, alignment=TA_CENTER)
    center_bi = ParagraphStyle("center_bi", parent=center, fontName="Helvetica-BoldOblique", fontSize=12)
    right = ParagraphStyle("right", parent=base, alignment=TA_RIGHT, fontName="Helvetica-Bold", fontSize=11)

    E = escape
    story = [
        Paragraph(f"\u201c<i><b>{E(ANIO_FRASE)}</b></i>\u201d", center_bi),
        Spacer(1, 0.8 * cm),
        Paragraph("SOLICITA:", right),
        Spacer(1, 0.15 * cm),
    ]

    filas_check = []
    for i, item in enumerate(CHECKLIST):
        pref = "- " + item
        if i == 0:
            pref = "- " + item
        filas_check.append([Paragraph(f"<b>{E(pref)}</b>", base),
                            Paragraph("<b>X</b>" if i == cfg["check"] else "", center)])
    t_check = Table(filas_check, colWidths=[11 * cm, 0.9 * cm], hAlign="RIGHT")
    t_check.setStyle(TableStyle([
        ("BOX", (1, 0), (1, -1), 1, "black"),
        ("INNERGRID", (1, 0), (1, -1), 0.7, "black"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 150),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))
    story += [t_check, Spacer(1, 0.6 * cm)]

    story += [
        Paragraph("Señor:", bold),
        Paragraph("ING. SANTOS D. FARIAS CABREJO", bold),
        Paragraph("Gerente JUSHMCHL CLASE A", base),
        Paragraph("<u>Ciudad</u>.-", base),
        Spacer(1, 0.5 * cm),
        Paragraph(f"Yo, <b>{E(nombre_sol)}</b>, agricultor(a) del Sub Sector Hidráulico: "
                  f"<b>{E(comision)}</b> identificado(a) con DNI. Nº <b>{E(dni_sol)}</b> ante Usted, "
                  f"con el debido respeto me presento y expongo:", base),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Que necesito realizar trámite documentario ante la Administración Local de Agua "
                  f"Chancay Lambayeque, por lo que solicito a su despacho me extienda "
                  f"{E(cfg['constancia'])} de la Junta de Usuarios Chancay Lambayeque.", base),
        Spacer(1, 0.4 * cm),
        Paragraph("Las características del predio son:", base),
        Spacer(1, 0.15 * cm),
    ]

    filas_predio = [
        ("Usuario", titular), ("Nombre del Predio", nombre_predio), ("Área", area),
        ("Canal", canal), ("Código", codigo),
    ]
    t_predio = Table([[Paragraph(f"<b>{E(k)}</b>", base), Paragraph(":", base), Paragraph(f"<b>{E(v)}</b>", base)]
                      for k, v in filas_predio],
                     colWidths=[4.2 * cm, 0.6 * cm, 10 * cm])
    t_predio.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t_predio, Spacer(1, 0.4 * cm),
              Paragraph("Es justicia que espero alcanzar.", base),
              Spacer(1, 0.8 * cm),
              Paragraph(f"Chiclayo, {fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}",
                        ParagraphStyle("fecha", parent=base, alignment=TA_RIGHT)),
              Spacer(1, 1.6 * cm),
              HRFlowable(width="60%", thickness=1, color="black", hAlign="CENTER"),
              Paragraph(f"<b>{E(nombre_sol)}</b>", center),
              Paragraph(f"<b>DNI.</b> {E(dni_sol)}", center),
              Spacer(1, 1.2 * cm),
              Paragraph(f"<b>CELULAR:</b> {E(cel_sol)}", base)]

    doc.build(story)
    return buf.getvalue()
