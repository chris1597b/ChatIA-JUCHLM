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


def test_boton_predio_pide_dni():
    o = _orc()
    resp, sid = o.handle(None, "", None)
    resp2, _ = o.handle(sid, "", "informacion_predio")
    assert resp2.state == ConversationState.PREDIO_REQUEST_NAME
    assert "dni" in resp2.message.lower()
    assert any(a.id == "volver_menu" for a in resp2.actions)


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
