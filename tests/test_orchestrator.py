"""Tests Orchestrator (§29): botón->workflow, pregunta->routing, acción inválida, estado inválido."""
from core.models import ConversationState
from core.session_manager import InMemorySessionManager
from core.orchestrator import ConversationOrchestrator


def _orc():
    return ConversationOrchestrator(InMemorySessionManager(ttl_seconds=60))


def test_start_devuelve_menu():
    o = _orc()
    resp, sid = o.handle(None, "", None)
    assert resp.state == ConversationState.MAIN_MENU
    assert any(a.id == "informacion_junta" for a in resp.actions)
    assert any(a.id == "informacion_predio" for a in resp.actions)


def test_boton_predio_muestra_submenu():
    o = _orc()
    resp, sid = o.handle(None, "", None)
    resp2, _ = o.handle(sid, "", "informacion_predio")
    assert resp2.state == ConversationState.PREDIO_ELEGIR_METODO
    ids = [a.id for a in resp2.actions]
    assert "predio_por_dni" in ids and "predio_por_codigo" in ids and "volver_menu" in ids


def test_metodo_dni_pide_dni():
    o = _orc()
    _, sid = o.handle(None, "", None)
    r, _ = o.handle(sid, "", "predio_por_dni")
    assert r.state == ConversationState.PREDIO_REQUEST_NAME
    assert "dni" in r.message.lower()


def test_metodo_codigo_pide_codigo():
    o = _orc()
    _, sid = o.handle(None, "", None)
    r, _ = o.handle(sid, "", "predio_por_codigo")
    assert r.state == ConversationState.PREDIO_REQUEST_NAME
    assert "digo de riego" in r.message.lower()


def test_escribir_dni_directo_resuelve():
    o = _orc()
    _, sid = o.handle(None, "", None)
    o.handle(sid, "", "informacion_predio")
    # simular respuesta del servicio sin DB
    import services.predio_service as P
    P_cons = P.consultar_por_dni
    P.consultar_por_dni = lambda dni, repository=None: {"found": True, "count": 1,
        "data": {"dni": "32104221", "titular": "T", "predios": [{}]}, "message": "M"}
    try:
        r, _ = o.handle(sid, "32104221", None)
        assert r.state == ConversationState.ASK_TRAMITE
    finally:
        P.consultar_por_dni = P_cons


def test_accion_invalida_rechazada():
    o = _orc()
    resp, sid = o.handle(None, "", None)
    resp2, _ = o.handle(sid, "", "capacidad_inexistente_xyz")
    assert "aún no está disponible" in resp2.message


def test_tramite_pide_datos_en_orden(tmp_path, monkeypatch):
    import services.tramite_service as T
    (tmp_path / "constancia_no_adeudo.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)
    o = _orc()
    _, sid = o.handle(None, "", None)
    r, _ = o.handle(sid, "", "tramite_constancia_no_adeudo")
    assert r.state == ConversationState.TRAMITE_PIDE_NOMBRE
    r, _ = o.handle(sid, "Perez Lopez Juan", None)
    assert r.state == ConversationState.TRAMITE_PIDE_DNI
    r, _ = o.handle(sid, "abc", None)
    assert r.state == ConversationState.TRAMITE_PIDE_DNI  # error conserva estado
    assert any(a.id == "volver_menu" for a in r.actions)
    r, _ = o.handle(sid, "32104221", None)
    assert r.state == ConversationState.TRAMITE_PIDE_CELULAR
    r, _ = o.handle(sid, "987654321", None)
    assert r.state == ConversationState.TRAMITE_REVISAR
    assert any(a.id == "confirmar_tramite" for a in r.actions)
    r, _ = o.handle(sid, "", "confirmar_tramite")
    assert r.state == ConversationState.CONSTANCIA_USUARIO
    assert r.data["download_url"] == "/api/tramites/constancia-no-adeudo"


def test_tramite_sin_pdf_avisa_amable(tmp_path, monkeypatch):
    import services.tramite_service as T
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)  # vacío: sin PDFs
    o = _orc()
    _, sid = o.handle(None, "", None)
    r, _ = o.handle(sid, "", "tramite_constancia_usuario")
    assert r.state == ConversationState.TRAMITE_MENU
    assert "aún no está disponible" in r.message
    assert any(a.id == "tramite_constancia_usuario" for a in r.actions)


def _ctx_predios(o, sid, predios):
    o.sessions.update_context(
        sid, tramite_id="tramite_constancia_no_adeudo",
        tramite_datos={"nombre": "X Y", "dni": "32104221", "celular": "987654321"},
        predio={"dni": "32104221", "titular": "T", "predios": predios})


def test_confirmar_con_varios_pide_elegir(tmp_path, monkeypatch):
    import services.tramite_service as T
    (tmp_path / "constancia_no_adeudo.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)
    o = _orc()
    _, sid = o.handle(None, "", None)
    _ctx_predios(o, sid, [{"nombre del predio": "A", "codigo de riego": "C1"},
                          {"nombre del predio": "B", "codigo de riego": "C2"}])
    r, _ = o.handle(sid, "", "confirmar_tramite")
    assert r.state == ConversationState.TRAMITE_ELEGIR_PREDIO
    assert [a.id for a in r.actions] == ["elegir_predio_0", "elegir_predio_1", "volver_menu"]
    r2, _ = o.handle(sid, "2", None)
    assert r2.state == ConversationState.CONSTANCIA_USUARIO
    assert "Predio incluido" in r2.message and "B" in r2.message
    o.sessions.set_state(sid, ConversationState.TRAMITE_ELEGIR_PREDIO)
    r4, _ = o.handle(sid, "", "elegir_predio_0")
    assert "Predio incluido" in r4.message and "A" in r4.message


def test_confirmar_con_uno_descarga_directo(tmp_path, monkeypatch):
    import services.tramite_service as T
    (tmp_path / "constancia_no_adeudo.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)
    o = _orc()
    _, sid = o.handle(None, "", None)
    _ctx_predios(o, sid, [{"nombre del predio": "Solo", "codigo de riego": "C9"}])
    r, _ = o.handle(sid, "", "confirmar_tramite")
    assert r.state == ConversationState.CONSTANCIA_USUARIO
    assert "Solo" in r.message


def test_menu_tramite_no_navega_descarga():
    """Regresión: los botones del menú eran type=download sin URL y el
    frontend navegaba al endpoint estático muerto."""
    o = _orc()
    _, sid = o.handle(None, "", None)
    r, _ = o.handle(sid, "", "realizar_tramite")
    assert r.state == ConversationState.TRAMITE_MENU
    for a in r.actions:
        assert a.type != "download", a.id


def test_boton_descargar_lleva_url_generar(tmp_path, monkeypatch):
    import services.tramite_service as T
    (tmp_path / "constancia_no_adeudo.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(T, "TRAMITES_DIR", tmp_path)
    o = _orc()
    _, sid = o.handle(None, "", None)
    _ctx_predios(o, sid, [{"nombre del predio": "A", "codigo de riego": "C1"}])
    r, _ = o.handle(sid, "", "confirmar_tramite")
    assert r.state == ConversationState.CONSTANCIA_USUARIO
    dl = [a for a in r.actions if a.type == "download"]
    assert len(dl) == 1
    assert r.data["download_url"].startswith("/api/tramites/generar/")


def test_flujo_si_no_tras_predio():
    o = _orc()
    _, sid = o.handle(None, "", None)
    # simular estado ASK_TRAMITE directamente
    s = o.sessions.get_or_create(sid)
    from core.models import ConversationState as S
    o.sessions.set_state(sid, S.ASK_TRAMITE)
    r_si, _ = o.handle(sid, "sí", None)
    assert r_si.state == S.TRAMITE_MENU
    o.sessions.set_state(sid, S.ASK_TRAMITE)
    r_no, _ = o.handle(sid, "no", None)
    assert r_no.state == S.END
