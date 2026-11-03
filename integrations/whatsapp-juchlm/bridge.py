"""
Puente WhatsApp <-> ChatJUCHLM (probe aislado)
=============================================
Arquitectura (Fase "probar sin romper"):

    WhatsApp ---> OpenWA(:2785) --webhook message.received--> bridge(:2786)
        bridge --POST /api/conversation/message--> ChatJUCHLM FastAPI(:8000, INTACTO)
        bridge --POST /api/sessions/{s}/messages/send-text--> OpenWA ---> WhatsApp

Garantias del diseño:
- El sistema web NO se modifica: el puente habla SOLO por su contrato HTTP ya
  validado (el mismo que consume chat.html).
- Este proceso vive y muere fuera del repo principal (integrations/, gitignored).
- Solo biblioteca estandar de Python: cero paquetes nuevos en el .venv real.

Uso:  .venv\\Scripts\\python.exe integrations\\whatsapp-juchlm\\bridge.py
(Copy .env.example -> .env y completa OPENWA_API_KEY / WEBHOOK_SECRET.)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlreq

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "bridge.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("bridge")


# ------------------------------------------------------------------ config
def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        log.error("Falta %s (copia .env.example a .env y rellenalo). Terminando.", path)
        raise SystemExit(1)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env


ENV = load_env(BASE / ".env")

WEBHOOK_SECRET: str = ENV.get("WEBHOOK_SECRET", "")
OPENWA_API_KEY: str = ENV.get("OPENWA_API_KEY", "")
OPENWA_URL: str = ENV.get("OPENWA_URL", "http://127.0.0.1:2785").rstrip("/")
OPENWA_SESSION: str = ENV.get("OPENWA_SESSION", "juchlm-probe")
JC_URL: str = ENV.get("JC_URL", "http://127.0.0.1:8000").rstrip("/")
BRIDGE_PORT: int = int(ENV.get("BRIDGE_PORT", "2786"))
ALLOW_GROUPS: bool = ENV.get("ALLOW_GROUPS", "false").lower() == "true"
SEND_ACK: bool = ENV.get("SEND_ACK", "true").lower() == "true"
SEND_TYPING: bool = ENV.get("SEND_TYPING", "true").lower() == "true"
TYPING_PRIMER_REFRESCO = 3  # 1er re-afirmado tras el ACK (un envio puede limpiar 'composing')
TYPING_INTERVAL = 20        # WhatsApp vence el indicador 'escribiendo...' en ~25 s
MESSAGE_PREFIX: str = ENV.get("MESSAGE_PREFIX", "")

# Autocuracion de la sesion de OpenWA: si el navegador de WhatsApp muere (o el motor
# queda trabado), el puente lo detecta y reconecta solo. Sin esto, una caida deja al
# bot "mudo" (los mensajes entran al gateway pero ningun webhook llega al puente).
WATCHDOG: bool = ENV.get("WATCHDOG_SESSION", "true").lower() == "true"
WATCHDOG_INTERVAL = int(ENV.get("WATCHDOG_INTERVAL", "60"))   # revision de estado (s)
WATCHDOG_START_TRAS = int(ENV.get("WATCHDOG_START_TRAS", "2"))  # 2 fallos -> POST /start
WATCHDOG_RESTART_TRAS = int(ENV.get("WATCHDOG_RESTART_TRAS", "5"))  # 5 fallos -> stop+start

if not WEBHOOK_SECRET or len(WEBHOOK_SECRET) < 16:
    raise SystemExit("WEBHOOK_SECRET debe existir y tener >= 16 caracteres")
if not OPENWA_API_KEY:
    raise SystemExit("Falta OPENWA_API_KEY (ver integrations/OpenWA/data/.api-key)")

WA_TEXT_MAX = 3800          # send-text de OpenWA admite hasta 4096; margen de seguridad
SEEN = OrderedDict()        # deduplicacion de redelivery del webhook
SEEN_LOCK = threading.Lock()
KNOWN_CHATS = set()         # chats ya inicializados con /api/conversation/reset
CHATS_LOCK = threading.Lock()
JC_TIMEOUT = 150            # carga de modelo local puede tardar la primera vez
WA_PER_CHAT_LOCKS: dict[str, threading.Lock] = {}
WA_PER_CHAT_LOCKS_GUARD = threading.Lock()

# Paridad de los "botones" del portal: por chat, las acciones de la ULTIMA respuesta
# del FSM como [(etiquetas_normalizadas, action_id, etiqueta_original)].
# Respondiendo la etiqueta tal cual en WhatsApp equivale a hacer click en el portal.
CHIPS: dict[str, list[tuple[frozenset, str, str]]] = {}
CHIPS_MAX = 64

# ------------------------------------------------------------------ utils
def http_json(method: str, url: str, payload: dict, headers: dict, timeout: int) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urlreq.Request(url, data=body, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlreq.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, {"_raw": raw[:500].decode("utf-8", "replace")}
    except urlerror.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            return e.code, json.loads(raw.decode("utf-8"))
        except Exception:
            return e.code, {"_raw": raw[:500].decode("utf-8", "replace")}


def chat_lock(chat_id: str) -> threading.Lock:
    with WA_PER_CHAT_LOCKS_GUARD:
        return WA_PER_CHAT_LOCKS.setdefault(chat_id, threading.Lock())


def sid_for_chat(chat_id: str) -> str:
    """Una sesion FSM estable por chat de WhatsApp (sin importancia de mayusculas)."""
    digits = chat_id.split("@")[0] or chat_id
    return f"wa{digits}"


def fold(s: str) -> str:
    """Normaliza etiqnas para casar un 'chip' de la web con lo que el usuario teclea en WA.

    Quita tildes/emojis/puntuacion y deja el texto en minusculas, separado por espacios:
    '⬅️ Volver al menú' -> 'volver al menu' ; 'volver_menu' -> 'volver nenu'.
    """
    t = unicodedata.normalize("NFKD", str(s or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^0-9a-z]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def markdown_a_whatsapp(texto: str) -> str:
    t = texto or ""
    t = t.replace("**", "*")                       # negrita markdown -> WA
    t = t.replace("```", "")                       # sin bloques de codigo
    t = t.replace("`", "")
    lineas = []
    for linea in t.splitlines():
        s = linea.lstrip()
        if s.startswith("#"):                       # titulos markdown fuera
            s = s.lstrip("#").strip()
        if s.startswith("- "):                      # bullets estilo WA
            s = "• " + s[2:]
        lineas.append(linea.replace(s, s, 1))
    t = "\n".join(lineas)
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    t = t.strip()
    if len(t) > WA_TEXT_MAX:
        t = t[: WA_TEXT_MAX - 40] + "\n\n[...] mensaje recortado por longitud de WhatsApp"
    return t


def texto_respuesta(response: dict, entrega_doc: bool = False) -> str:
    """Extrae y traduce la respuesta orquestada a texto plano de WhatsApp."""
    partes = [str(response.get("message") or "").strip()]
    datos = response.get("data") if isinstance(response.get("data"), dict) else {}
    enlace = str((datos or {}).get("download_url") or "")
    if enlace:
        if entrega_doc:
            partes.append("📄 *El PDF te lo mando como documento WhatsApp* (tambien queda disponible en el portal).")
        else:
            partes.append("🔗 Descarga disponible en el portal: " + (JC_URL + enlace if enlace.startswith("/") else enlace))
    return "\n\n".join(p for p in partes if p)


# ------------------------------------------------------------------ ChatJUCHLM
__primera_conversacion = time.time()

def jc_reset(sid: str) -> None:
    http_json("POST", f"{JC_URL}/api/conversation/reset", {"session_id": sid},
              {}, JC_TIMEOUT)


def jc_message(sid: str, texto: str) -> dict:
    _, resp = http_json("POST", f"{JC_URL}/api/conversation/message",
                        {"session_id": sid, "message": texto}, {}, JC_TIMEOUT)
    return resp.get("response") or {}


def jc_click(sid: str, action_id: str) -> dict:
    """El 'click' de un boton del portal, por el mismo contrato que usa chat.html."""
    _, resp = http_json("POST", f"{JC_URL}/api/conversation/message",
                        {"session_id": sid, "message": "", "action_id": action_id}, {}, JC_TIMEOUT)
    return resp.get("response") or {}


def chips_de_acciones(acciones: list) -> list:
    """Las acciones del FSM sin la de descarga (el PDF se entrega como documento)."""
    lista = []
    for a in acciones or []:
        if not isinstance(a, dict) or str(a.get("type") or "") == "download":
            continue
        aid = str(a.get("id") or "").strip()
        lab = str(a.get("label") or "").strip()
        if not aid or not lab:
            continue
        claves = {fold(lab), fold(aid)}
        claves.discard("")
        lista.append((frozenset(claves), aid, lab))
    return lista


def chips_registrar(chat_id: str, chips: list) -> None:
    with CHATS_LOCK:
        if chips:
            CHIPS[chat_id] = chips
        else:
            CHIPS.pop(chat_id, None)
        while len(CHIPS) > CHIPS_MAX:
            CHIPS.pop(next(iter(CHIPS)))


def chip_de(chat_id: str, texto_usuario: str) -> str | None:
    """Si el usuario teclea exactamente un chip, devuelve su action_id (paridad click)."""
    f = fold(texto_usuario)
    if not f:
        return None
    with CHATS_LOCK:
        lista = CHIPS.get(chat_id) or []
    for claves, aid, _ in lista:
        if f in claves:
            return aid
    return None


# El propio menú numerado del FSM (p. ej. "1. Predio ..." en la lista del padrón):
# si el texto ya lo trae, los DIGITOS quedan reservados para el FSM y no los interceptamos.
MENU_NUMERADO = re.compile(r"(?m)^\s*1\s*[\)\.\:\u00ba·-]")

def texto_con_opciones(texto: str, acciones: list) -> tuple[str, list]:
    """Paridad de los botones del portal como OPCIONES NUMERADAS de WhatsApp.

    Devuelve (texto_final, chips_a_registrar). Los chips que se registran llevan
    alias numericos SOLO cuando el puente renderizo la lista numerada (>= 2 opciones
    y el FSM no trajo su propio menu numerado en el texto).
    """
    chips = chips_de_acciones(acciones)
    if not chips:
        return texto, []
    if MENU_NUMERADO.search(texto or ""):
        # el FSM ya numera: mostrar los botones como texto libre, sin robar digitos
        hint = " · ".join(c[2] for c in chips)
        return (texto + "\n\n📲 Escribe tal cual: *" + hint + "*").strip(), chips
    if len(chips) == 1:
        # escape unico (p. ej. Volver al menu) SIN alias numerico: un "1" suelto en
        # otros estados del FSM significa otra cosa (elegir predio, etc.)
        return (texto + "\n\n📲 Escribe tal cual: *" + chips[0][2] + "*").strip(), chips
    lineas = []
    numerados = []
    for i, (claves, aid, lab) in enumerate(chips, start=1):
        lineas.append(f"{i}. {lab}")
        numerados.append((frozenset(set(claves) | {str(i)}), aid, lab))
    return (texto + "\n\n👇 *Responde con el número:*\n" + "\n".join(lineas)).strip(), numerados


# ------------------------------------------------------------------ OpenWA
def wa_typing(chat_id: str, estado: str) -> None:
    """Presencia 'escribiendo...' (typing) o 'paused' para limpiarla. Fallo no fatal."""
    url = f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}/chats/typing"
    try:
        code, _ = http_json("POST", url, {"chatId": chat_id, "state": estado},
                            {"X-API-Key": OPENWA_API_KEY}, 15)
        if code in (200, 201):
            log.info("typing '%s' %s", estado, chat_id)
        else:
            log.warning("typing '%s' %s -> HTTP %d", estado, chat_id, code)
    except Exception as e:
        log.warning("typing '%s' %s excepcion: %r", estado, chat_id, e)


def _typing_loop(chat_id: str, stop: threading.Event) -> None:
    """Keep-alive del indicador: WhatsApp lo vence a ~25 s. El 1er re-afirmado es a los
    3 s (el envio del ACK puede limpiar el 'composing' del lado WhatsApp Web); luego
    cada TYPING_INTERVAL."""
    if stop.wait(TYPING_PRIMER_REFRESCO):
        return
    wa_typing(chat_id, "typing")
    while not stop.wait(TYPING_INTERVAL):
        wa_typing(chat_id, "typing")


def wa_send(chat_id: str, texto: str) -> None:
    url = f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}/messages/send-text"
    payload = {"chatId": chat_id, "text": texto}
    nueva = ""
    try:
        code, resp = http_json("POST", url, payload, {"X-API-Key": OPENWA_API_KEY}, 90)
        if code not in (200, 201):
            nueva = f"HTTP {code}: {resp.get('_raw') or resp.get('message') or resp}"
            log.warning("wa_send fallo %s -> %s", chat_id, nueva[:300])
        else:
            log.info("wa_send ok %s (%d chars)", chat_id, len(texto))
    except Exception as e:  # timeout/conexion
        nueva = repr(e)
        log.error("wa_send excepcion %s: %r", chat_id, e)
    return None


def wa_send_document(chat_id: str, archivo: dict, caption: str = "") -> None:
    """Paridad del boton Descargar del portal: entrega el PDF como documento WhatsApp."""
    url = f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}/messages/send-document"
    payload = {
        "chatId": chat_id,
        "base64": archivo["b64"],
        "mimetype": archivo["mimetype"],
        "filename": archivo["filename"],
    }
    if caption:
        payload["caption"] = caption[:1024]
    try:
        code, resp = http_json("POST", url, payload, {"X-API-Key": OPENWA_API_KEY}, 90)
        if code in (200, 201):
            log.info("wa_doc ok %s (%s, %d bytes)", chat_id, archivo["filename"], archivo["bytes"])
        else:
            log.warning("wa_doc fallo %s -> HTTP %d: %s", chat_id, code, str(resp)[:200])
    except Exception as e:
        log.error("wa_doc excepcion %s: %r", chat_id, e)


def jc_descargar_archivo(ruta: str) -> dict | None:
    """Baja el archivo que la web emite en data.download_url y lo prepara para WA."""
    try:
        url = ruta if ruta.startswith("http") else JC_URL + (ruta if ruta.startswith("/") else "/" + ruta)
        req = urlreq.Request(url)
        # (Punto 3) si el portal exige auth para /api/tramites/generar, anadir aca la cabecera de servicio.
        with urlreq.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if not data:
            return None
        if len(data) > 25_000_000:
            log.warning("jc archivo grande para WA (%d bytes): %s", len(data), str(ruta)[:120])
            return None
        ctype = (resp.headers.get("Content-Type") or "application/pdf").split(";")[0].strip() or "application/octet-stream"
        cd = resp.headers.get("Content-Disposition") or ""
        m = re.search(r'filename\*?="?([^";]+)"?', cd)
        nombre = (m.group(1).strip() if m else "")
        if not nombre:
            stem = Path(ruta.split("?")[0]).stem or "documento"
            nombre = stem if Path(stem).suffix else f"{stem}.pdf"
        return {"b64": base64.b64encode(data).decode("ascii"), "mimetype": ctype,
                "filename": nombre, "bytes": len(data)}
    except Exception as e:
        log.warning("jc archivo no descargable %s: %r", str(ruta)[:120], e)
        return None


# ------------------------------------------------------------------ pipeline
def pipeline(chat_id: str, texto_usuario: str) -> None:
    lock = chat_lock(chat_id)
    with lock:
        sid = sid_for_chat(chat_id)
        nuevo_chat = False
        with CHATS_LOCK:
            if chat_id not in KNOWN_CHATS:
                nuevo_chat = True
                KNOWN_CHATS.add(chat_id)
        try:
            if nuevo_chat:
                log.info("chat nuevo %s -> inicializar sesion FSM %s", chat_id, sid)
                jc_reset(sid)
            # paridad de los botones web: si el usuario teclea el texto de un boton
            # (o su id), el puente hace el 'click' exacto, no un mensaje de texto.
            click = chip_de(chat_id, texto_usuario)
            if click:
                log.info("chip '%s' -> action_id=%s (paridad botones web)", texto_usuario[:40], click)
                response = jc_click(sid, click)
            else:
                hilo_typing = None
                stop_typing = None
                try:
                    if SEND_ACK:
                        wa_send(chat_id, "🤔 Consultando al asistente JUCHLM...")
                    if SEND_TYPING:
                        # animacion de escritura mientras FSM/RAG procesa (puede tardar min en
                        # frio). Se enciende DESPUES del ACK: el propio envio del mensaje
                        # limpia el 'composing' del lado WhatsApp Web, y asi los puntos
                        # aparecen justo cuando el usuario termina de leer el ACK.
                        wa_typing(chat_id, "typing")
                        stop_typing = threading.Event()
                        hilo_typing = threading.Thread(
                            target=_typing_loop, args=(chat_id, stop_typing),
                            daemon=True, name="typing")
                        hilo_typing.start()
                    log.info("jc <- %r (%s) sid=%s", texto_usuario[:80], chat_id, sid)
                    response = jc_message(sid, texto_usuario)
                finally:
                    if stop_typing:
                        stop_typing.set()
                        hilo_typing.join(timeout=0.5)
                        wa_typing(chat_id, "paused")
            response = response or {}
            acciones = response.get("actions") if isinstance(response.get("actions"), list) else []
            datos = response.get("data") if isinstance(response.get("data"), dict) else {}
            ruta_archivo = str((datos or {}).get("download_url") or "")
            archivo = jc_descargar_archivo(ruta_archivo) if ruta_archivo else None
            texto = texto_respuesta(response, entrega_doc=archivo is not None)
            texto = markdown_a_whatsapp(texto)
            texto, chips = texto_con_opciones(texto, acciones or [])
            estado = response.get("state", "?")
            detalle = ""
            if archivo:
                detalle = f" archivo={archivo['filename']}"
            elif ruta_archivo:
                detalle = f" archivo-fallo ruta={ruta_archivo[:60]}"
            numerado = " (fsm-numera)" if MENU_NUMERADO.search(texto or "") else ""
            log.info("jc -> state=%s %d chars%s chips=%d%s", estado, len(texto), numerado, len(chips), detalle)
            chips_registrar(chat_id, chips)
            if texto:
                wa_send(chat_id, texto)
            if archivo:
                caption = "📄 " + (str(response.get("message") or "").strip().splitlines()[0][:200] or "documento institucional")
                wa_send_document(chat_id, archivo, caption=caption)
        except Exception as e:
            log.exception("pipeline fallo para %s", chat_id)
            try:
                wa_send(chat_id, "Lo siento, ocurrió un problema procesando tu solicitud.")
            except Exception:
                pass


# ------------------------------------------------------------------ webhook server
class Handler(BaseHTTPRequestHandler):
    server_version = "JUCHLM-WA-Bridge/0.1"

    def _json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # health del puente
        if self.path in ("/health", "/healthz"):
            self._json(200, {"ok": True, "session": OPENWA_SESSION,
                             "portal": JC_URL, "chats_activos": len(KNOWN_CHATS)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path not in ("/webhook", "/webhook/"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 2 * 1024 * 1024:
            self.send_response(413)
            self.end_headers()
            return
        raw = self.rfile.read(length)

        firma = self.headers.get("X-OpenWA-Signature", "")
        esperada = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
        if not firma or not hmac.compare_digest(esperada, firma):
            log.warning("webhook rechazado: firma invalida")
            self.send_response(403)
            self.end_headers()
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        event = payload.get("event")
        data = payload.get("data") or {}
        key = payload.get("idempotencyKey") or payload.get("deliveryId") or ""
        log.info("webhook event=%s key=%s", event, key[:24])

        with SEEN_LOCK:
            if key:
                if key in SEEN:
                    log.info("duplicado ignorado %s", key[:24])
                    self.send_response(200)
                    self.end_headers()
                    return
                SEEN[key] = time.time()
                while len(SEEN) > 128:
                    SEEN.popitem(last=True)

        if event != "message.received":
            self.send_response(200)
            self.end_headers()
            return

        is_group = bool(data.get("isGroup"))
        from_me = bool(data.get("fromMe"))
        mtype = str(data.get("type") or "")
        body_text = str(data.get("body") or "").strip()
        chat_id = str(data.get("chatId") or "")

        if not chat_id:
            self.send_response(200)
            self.end_headers()
            return
        if from_me:
            self.send_response(200)
            self.end_headers()
            return
        if is_group and not ALLOW_GROUPS:
            log.info("grupo ignorado %s", chat_id)
            self.send_response(200)
            self.end_headers()
            return
        # OpenWA normaliza wwebjs: 'chat' -> 'text' (message-mapper.ts); las respuestas de
        # botones/listas tambien llegan como 'text' (traen la eleccion en body). 'chat' queda por
        # compatibilidad con la convencion original del probe.
        if mtype and mtype not in ("chat", "text", "buttons_response", "list_response", "template_button_reply"):
            if not body_text or mtype in ("revoked", "unknown"):
                # tipo de sistema (revocado/editado/sin body): silencio, sin responder
                log.info("tipo '%s' (sistema/sin body) ignorado %s", mtype, chat_id)
            else:
                log.info("mensaje tipo '%s' no soportado en probe (solo texto)", mtype)
                try:
                    wa_send(chat_id, "Este asistente de prueba solo procesa mensajes de *texto*.")
                except Exception:
                    pass
            self.send_response(200)
            self.end_headers()
            return
        if not body_text:
            self.send_response(200)
            self.end_headers()
            return
        if MESSAGE_PREFIX and not body_text.startswith(MESSAGE_PREFIX):
            log.info("sin prefijo %r ignorado", body_text[:30])
            self.send_response(200)
            self.end_headers()
            return

        # ACK inmediato, procesado en hilo aparte (el webhook no debe bloquear)
        executor.submit(pipeline, chat_id, body_text)
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):  # silenciar el log por-defecto verboso
        return


executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="wa")

def _session_start() -> bool:
    """Pide a OpenWA que (re)conecte la sesion. 409 = ya esta arrancando: no es error."""
    code, resp = http_json("POST", f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}/start",
                           {}, {"X-API-Key": OPENWA_API_KEY}, 120)
    return code in (200, 201, 409)


def _session_status() -> str | None:
    try:
        code, resp = http_json("GET", f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}",
                               None, {"X-API-Key": OPENWA_API_KEY}, 20)
        if code not in (200, 201):
            return None
        st = resp.get("status") or resp.get("session") or {}
        return st.get("status") if isinstance(st, dict) else str(st)
    except Exception:
        return None


def _watchdog_loop(stop: threading.Event) -> None:
    """Vigila la sesion de WhatsApp y la recupera sola si se cae.

    Escalada: 2 revisiones fallidas -> POST /start (reconectar el navegador);
    5 revisiones fallidas -> stop + start (motor nuevo, p. ej. proceso colgado).
    """
    caidos = 0
    while not stop.wait(WATCHDOG_INTERVAL):
        estado = _session_status()
        if estado == "ready":
            if caidos:
                log.info("watchdog: sesion recuperada (status=ready)")
            caidos = 0
            continue
        caidos += 1
        log.warning("watchdog: sesion '%s' no lista (status=%s, revision %d)",
                    OPENWA_SESSION, estado, caidos)
        if caidos >= WATCHDOG_RESTART_TRAS:
            log.warning("watchdog: motor trabado -> stop + start de la sesion")
            try:
                http_json("POST", f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}/stop",
                          {}, {"X-API-Key": OPENWA_API_KEY}, 120)
            except Exception as e:
                log.warning("watchdog: stop fallo: %r", e)
            time.sleep(8)
            _session_start()
            caidos = 0
        elif caidos >= WATCHDOG_START_TRAS:
            log.warning("watchdog: reintentando conexion de la sesion (start)")
            _session_start()


def main() -> None:
    # ping de arranque: confirmar config de OpenWA antes de esperar webhooks
    try:
        code, resp = http_json("GET", f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION}",
                               None, {"X-API-Key": OPENWA_API_KEY}, 20)
        if code in (200, 201):
            status = (resp.get("status") or resp.get("session") or {})
            log.info("OpenWA sesion '%s' accesible (HTTP %d) status=%s",
                     OPENWA_SESSION, code, status if isinstance(status, str) else status.get("status"))
        elif code in (401, 403):
            log.error("OPENWA_API_KEY rechazada (HTTP %d). Copiala de integrations/OpenWA/data/.api-key", code)
        else:
            log.warning("Sesion '%s' aun no existe (HTTP %d): creala al arrancar OpenWA (ver README).",
                        OPENWA_SESSION, code)
    except Exception as e:
        log.error("OpenWA inalcanzable en %s: %r (arrancala primero)", OPENWA_URL, e)
    try:
        jc_code, _ = http_json("GET", f"{JC_URL}/health", None, {}, 20)
        log.info("ChatJUCHLM health HTTP %d", jc_code)
    except Exception as e:
        log.error("ChatJUCHLM inalcanzable en %s: %r (arrancalo primero)", JC_URL, e)

    # Autocuracion: si la sesion de WhatsApp quedo caida (navegador muerto o motor
    # trabado), el puente la levanta al arrancar y la vigila en segundo plano.
    if WATCHDOG:
        estado = _session_status()
        if estado != "ready":
            log.warning("Sesion '%s' no lista al arrancar (status=%s): intentando conectar",
                        OPENWA_SESSION, estado)
            _session_start()
        threading.Thread(target=_watchdog_loop, args=(threading.Event(),),
                         daemon=True, name="watchdog").start()
        log.info("Watchdog de sesion activo (revision cada %ds)", WATCHDOG_INTERVAL)

    server = ThreadingHTTPServer(("127.0.0.1", BRIDGE_PORT), Handler)
    log.info("Bridge WA->JUCHLM escuchando en http://127.0.0.1:%d/webhook "
             "(sesion OpenWA: %s | portales: JC=%s, WA=%s)",
             BRIDGE_PORT, OPENWA_SESSION, JC_URL, OPENWA_URL)
    server.serve_forever()


if __name__ == "__main__":
    main()
