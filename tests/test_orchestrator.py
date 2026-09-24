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
