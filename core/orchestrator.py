"""ConversationOrchestrator (§8, §20). Única capa que decide qué módulo se ejecuta."""
from __future__ import annotations
import logging
import re

from core.models import ActionButton, AssistantResponse, ConversationState
from core.session_manager import InMemorySessionManager
from core.exceptions import DomainValidationError, DatabaseUnavailableError, TramiteNotFoundError
from security import validation as V
from security import authorization as AuthZ
from security import audit as Audit
import services.menu_service as Menu

log = logging.getLogger("application")

AFIRMATIVO = re.compile(r"^\s*(s[ií]|si|sí|yes|y|claro|por favor|ok)\s*[.]*\s*$", re.I)
NEGATIVO = re.compile(r"^\s*(no|n|nel|para nada)\s*[.]*\s*$", re.I)
PREDIO_HINT = re.compile(r"predio|parcela|c[oó]digo.*predio|sector|titular|propietar", re.I)
TRAMITE_HINT = re.compile(r"tr[aá]mite|constancia|posesi[oó]n|solicitud|descargar|formato", re.I)


def _btn(id: str, label: str, type: str = "action") -> ActionButton:
    return ActionButton(id=id, label=label, type=type)


def main_menu_actions() -> list[ActionButton]:
    return [_btn("informacion_junta", "📚 Información de la Junta", "menu"),
            _btn("informacion_predio", "🏠 Información del predio", "workflow")]


def ask_tramite_actions() -> list[ActionButton]:
    return [_btn("realizar_tramite", "✅ Sí, realizar trámite", "action"),
            _btn("finalizar", "❌ No, por ahora", "action")]


def tramite_menu_actions() -> list[ActionButton]:
    items = Menu.get_tramite_menu()
    out = [_btn(c["id"], c["label"], "download") for c in items]
    out.append(_btn("volver_menu", "⬅️ Volver al menú", "action"))
    return out


def volver_action() -> list[ActionButton]:
    """Escape hatch: ninguna respuesta puede dejar al usuario sin botones."""
    return [_btn("volver_menu", "⬅️ Volver al menú", "action")]


class ConversationOrchestrator:
    def __init__(self, sessions: InMemorySessionManager, role: str = "USUARIO"):
        self.sessions = sessions
        self.role = role

    # ---------- entry ----------
    def handle(self, session_id: str | None, message: str = "", action_id: str | None = None) -> tuple[AssistantResponse, str]:
        session = self.sessions.get_or_create(session_id)
        msg = (message or "").strip()[:2000]
        act = (action_id or "").strip()[:64] or None

        if session.state == ConversationState.START:
            session = self.sessions.set_state(session.session_id, ConversationState.MAIN_MENU)
            resp = AssistantResponse(message="ChatJUCHLM\n\n¿Qué deseas hacer?",
                                     state=ConversationState.MAIN_MENU,
                                     actions=main_menu_actions(), source="menu")
            self.sessions.push_history(session.session_id, "assistant", resp.message)
            return resp, session.session_id

        # 1) BOTÓN EXPLÍCITO (§20) — siempre prevalece
        if act:
            try:
                V.validate_capability_id(act)
            except DomainValidationError:
                return self._safe(session, "Acción no válida."), session.session_id
            if not Menu.is_valid_action(act) and act not in ("realizar_tramite", "finalizar", "volver_menu",
                                                                 "predio_por_dni", "predio_por_codigo"):
                Audit.audit(session.session_id, "unknown_action", act, {"action": act}, "rejected")
                return self._safe(session, "Esa función aún no está disponible."), session.session_id
            if not AuthZ.can_access(act, self.role):
                return self._safe(session, "No tienes permiso para esa función."), session.session_id
            return self._handle_action(session, act), session.session_id

        # 2) RESPUESTA A SOLICITUD DE DATOS (depende del estado, no del LLM §9)
        if session.state == ConversationState.PREDIO_ELEGIR_METODO:
            return self._handle_elegir_metodo(session, msg), session.session_id
        if session.state == ConversationState.PREDIO_REQUEST_NAME:
            return self._handle_predio_nombre(session, msg), session.session_id
        if session.state == ConversationState.ASK_TRAMITE:
            return self._handle_ask_tramite(session, msg), session.session_id

        # 3) PREGUNTA LIBRE (§19): clasificar intención -> RAG | workflow | fallback
        if session.state in (ConversationState.MAIN_MENU, ConversationState.JUNTA_MENU,
                             ConversationState.RAG_QUERY, ConversationState.RAG_RESPONSE,
                             ConversationState.PREDIO_RESULT, ConversationState.END,
                             ConversationState.TRAMITE_MENU):
            if not msg:
                return self._main(session, "Elige una opción o escribe tu pregunta."), session.session_id
            if PREDIO_HINT.search(msg) and len(msg) < 60:
                # "información del predio" escrito a mano equivale al botón
                return self._handle_action(session, "informacion_predio"), session.session_id
            if TRAMITE_HINT.search(msg) and session.state in (ConversationState.PREDIO_RESULT, ConversationState.ASK_TRAMITE):
                return self._handle_action(session, "realizar_tramite"), session.session_id
            return self._handle_rag(session, msg), session.session_id

        return self._main(session, "Elige una opción del menú."), session.session_id

    # ---------- actions ----------
    def _handle_action(self, session, act: str) -> AssistantResponse:
        Audit.audit(session.session_id, "button", act, {"action": act}, "ok")
        if act in ("informacion_junta", "consulta_documental", "rag_query"):
            self.sessions.set_state(session.session_id, ConversationState.RAG_QUERY)
            self.sessions.update_context(session.session_id, capability="informacion_junta")
            return AssistantResponse(
                message="📚 Información de la Junta\n\nEscribe tu pregunta sobre documentos institucionales y la buscaré en el archivo.",
                state=ConversationState.RAG_QUERY,
                actions=[_btn("volver_menu", "⬅️ Volver al menú", "action")], source="menu")
        if act in ("informacion_predio", "consultar_predio"):
            self.sessions.set_state(session.session_id, ConversationState.PREDIO_ELEGIR_METODO)
            self.sessions.update_context(session.session_id, capability="informacion_predio")
            return AssistantResponse(
                message="🏠 Información del predio\n\n¿Cómo deseas buscar?",
                state=ConversationState.PREDIO_ELEGIR_METODO,
                actions=[_btn("predio_por_dni", "🪪 Buscar por DNI", "action"),
                         _btn("predio_por_codigo", "🏷️ Buscar por código de riego", "action"),
                         _btn("volver_menu", "⬅️ Volver al menú", "action")],
                source="menu")
        if act in ("predio_por_dni",):
            self.sessions.set_state(session.session_id, ConversationState.PREDIO_REQUEST_NAME)
            self.sessions.update_context(session.session_id, metodo="dni")
            return AssistantResponse(
                message="Claro. Para consultar la información registrada del predio, indícame tu número de DNI (8 dígitos).",
                state=ConversationState.PREDIO_REQUEST_NAME, actions=volver_action(), source="sql")
        if act in ("predio_por_codigo",):
            self.sessions.set_state(session.session_id, ConversationState.PREDIO_REQUEST_NAME)
            self.sessions.update_context(session.session_id, metodo="codigo")
            return AssistantResponse(
                message="Claro. Indícame el código de riego del predio (p. ej. MOXXXX4).",
                state=ConversationState.PREDIO_REQUEST_NAME, actions=volver_action(), source="sql")
        if act in ("realizar_tramite",):
            self.sessions.set_state(session.session_id, ConversationState.TRAMITE_MENU)
            return AssistantResponse(message="📄 Trámites disponibles:", state=ConversationState.TRAMITE_MENU,
                                     actions=tramite_menu_actions(), source="tramite")
        if act in ("tramite_constancia_usuario", "descargar_constancia_usuario", "constancia_usuario"):
            self.sessions.set_state(session.session_id, ConversationState.CONSTANCIA_USUARIO)
            return AssistantResponse(
                message="📄 Constancia de red de riego\n\nPuedes descargar el formato institucional aquí:",
                state=ConversationState.CONSTANCIA_USUARIO,
                actions=[_btn("descargar_constancia_usuario", "⬇️ Descargar constancia", "download"),
                         _btn("volver_menu", "⬅️ Volver al menú", "action")],
                data={"download_url": "/api/tramites/constancia-usuario",
                      "filename": "constancia_usuario.pdf"}, source="tramite")
        if act in ("finalizar",):
            self.sessions.set_state(session.session_id, ConversationState.END)
            return AssistantResponse(
                message="Gracias por tu atención. Estoy disponible para ayudarte con información de la Junta y de tus predios.",
                state=ConversationState.END, actions=main_menu_actions(), source="orchestrator")
        if act in ("volver_menu",):
            return self._main(session, "ChatJUCHLM\n\n¿Qué deseas hacer?")
        return self._safe(session, "Esa función aún no está disponible.")

    # ---------- predio ----------
    def _handle_elegir_metodo(self, session, msg: str) -> AssistantResponse:
        """Si escribe en vez de pulsar: 8 dígitos -> DNI, código -> riego.
        El dato escrito se procesa directo, sin pedirlo de nuevo."""
        for metodo, validador in (("dni", V.validate_dni), ("codigo", V.validate_codigo_riego)):
            try:
                validador(msg)
                self.sessions.update_context(session.session_id, metodo=metodo)
                self.sessions.set_state(session.session_id, ConversationState.PREDIO_REQUEST_NAME)
                return self._handle_predio_nombre(session, msg)
            except DomainValidationError:
                continue
        # Re-preguntar conservando los 2 botones (el mensaje llega vacío al
        # reanudar sesiones, así que no se exige dato aquí).
        self.sessions.set_state(session.session_id, ConversationState.PREDIO_ELEGIR_METODO)
        return AssistantResponse(
            message="Elige cómo buscar: por DNI (8 dígitos) o por código de riego.",
            state=ConversationState.PREDIO_ELEGIR_METODO,
            actions=[_btn("predio_por_dni", "🪪 Buscar por DNI", "action"),
                     _btn("predio_por_codigo", "🏷️ Buscar por código de riego", "action"),
                     _btn("volver_menu", "⬅️ Volver al menú", "action")],
            source="menu")

    def _handle_predio_nombre(self, session, msg: str) -> AssistantResponse:
        from services import predio_service as Predio
        metodo = self.sessions.get_or_create(session.session_id).context.get("metodo", "dni")
        try:
            if metodo == "codigo":
                result = Predio.consultar_por_codigo(msg)
            else:
                result = Predio.consultar_por_dni(msg)
        except DomainValidationError as e:
            return AssistantResponse(message=str(e), state=ConversationState.PREDIO_REQUEST_NAME, actions=volver_action(), source="sql")
        except DatabaseUnavailableError:
            log.exception("predio sql no disponible")
            Audit.audit(session.session_id, "predio_query", "informacion_predio", {}, "db_unavailable")
            return AssistantResponse(message="No fue posible completar la consulta en este momento.",
                                     state=ConversationState.PREDIO_REQUEST_NAME, actions=volver_action(), source="sql")
        Audit.audit(session.session_id, "predio_query", "informacion_predio", {"documento": msg}, "ok")
        if not result["found"]:
            self.sessions.set_state(session.session_id, ConversationState.PREDIO_REQUEST_NAME)
            return AssistantResponse(message=result["message"], state=ConversationState.PREDIO_REQUEST_NAME,
                                     actions=volver_action(),
                                     data=result, source="sql")
        # Padrón (DNI o código): N filas = N predios del titular → se listan
        # todos y se ofrece trámite (no es ambigüedad como en flujo por nombre).
        if isinstance(result.get("data"), dict) and "predios" in result["data"]:
            self.sessions.set_state(session.session_id, ConversationState.ASK_TRAMITE)
            msg_final = result["message"] + "\n\n¿Deseas realizar un trámite?"
            self.sessions.update_context(session.session_id, predio=result["data"], predio_message=msg_final)
            return AssistantResponse(message=msg_final,
                                     state=ConversationState.ASK_TRAMITE,
                                     actions=ask_tramite_actions(), data=result["data"], source="sql")
        if result["count"] > 1:
            self.sessions.set_state(session.session_id, ConversationState.PREDIO_DISAMBIGUATE)
            self.sessions.update_context(session.session_id, candidatos=result["data"])
            return AssistantResponse(message=result["message"], state=ConversationState.PREDIO_DISAMBIGUATE,
                                     actions=volver_action(), data=result, source="sql")
        # 1 resultado (§15 + §16)
        self.sessions.set_state(session.session_id, ConversationState.ASK_TRAMITE)
        self.sessions.update_context(session.session_id, predio=result["data"])
        return AssistantResponse(message=result["message"] + "\n\n¿Deseas realizar un trámite?",
                                 state=ConversationState.ASK_TRAMITE,
                                 actions=ask_tramite_actions(), data=result, source="sql")

    def _handle_ask_tramite(self, session, msg: str) -> AssistantResponse:
        if AFIRMATIVO.match(msg or ""):
            return self._handle_action(session, "realizar_tramite")
        if NEGATIVO.match(msg or ""):
            return self._handle_action(session, "finalizar")
        return AssistantResponse(message="¿Deseas realizar un trámite?\n\nResponde Sí o No.",
                                 state=ConversationState.ASK_TRAMITE,
                                 actions=ask_tramite_actions(), source="orchestrator")

    # ---------- rag ----------
    def _handle_rag(self, session, msg: str) -> AssistantResponse:
        from services import rag_service as RAG
        self.sessions.set_state(session.session_id, ConversationState.RAG_QUERY)
        self.sessions.push_history(session.session_id, "user", msg)
        out = RAG.ask(msg)
        self.sessions.set_state(session.session_id, ConversationState.RAG_RESPONSE)
        self.sessions.push_history(session.session_id, "assistant", out["answer"][:1000])
        Audit.audit(session.session_id, "rag_query", "informacion_junta", {"q": msg[:80]}, "ok")
        return AssistantResponse(message=out["answer"], state=ConversationState.RAG_RESPONSE,
                                 actions=[_btn("volver_menu", "⬅️ Volver al menú", "action")],
                                 data={"sources": out.get("sources", []), "area": out.get("area", "general")},
                                 source="rag")

    # ---------- helpers ----------
    def _main(self, session, text: str) -> AssistantResponse:
        self.sessions.set_state(session.session_id, ConversationState.MAIN_MENU)
        return AssistantResponse(message=text, state=ConversationState.MAIN_MENU,
                                 actions=main_menu_actions(), source="menu")

    def _safe(self, session, text: str) -> AssistantResponse:
        return AssistantResponse(message=text, state=session.state, actions=volver_action(), source="orchestrator")
