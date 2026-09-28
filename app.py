"""ChatJUCHLM — plataforma conversacional híbrida (RAG V11 + SQL + Trámites).

- rag_juchlm.py V11 NO se modifica: se usa vía services/rag_service (lazy).
- Legacy /api/chat SSE se conserva para compatibilidad.
- Nuevo contrato orquestado: POST /api/conversation/message (§25).
"""
from __future__ import annotations
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import get_settings
from config.logging_setup import setup_logging
from core.session_manager import session_manager
from core.orchestrator import ConversationOrchestrator
from core.models import ConversationState
from schemas.chat import ConversationRequest
from security.auth import get_current_role
from security.validation import validate_safe_filename
import services.menu_service as Menu

settings = get_settings()
setup_logging(settings.LOG_DIR)
log = logging.getLogger("application")

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = (BASE_DIR / settings.PDF_FOLDER).resolve()
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---- Lazy legacy RAG (no bloquear startup; V11 intacto) ----
_legacy = {"chain": None, "retriever": None, "lock": threading.Lock()}


def _legacy_chain():
    if _legacy["chain"] is not None:
        return _legacy["chain"], _legacy["retriever"]
    with _legacy["lock"]:
        if _legacy["chain"] is not None:
            return _legacy["chain"], _legacy["retriever"]
        import rag_juchlm as legacy
        try:
            legacy.sincronizar_carpeta()
        except SystemExit:
            log.warning("pdfs/ ausente; se omite sincronización inicial")
        except Exception as e:
            log.warning("sincronización inicial omitida: %s", e)
        chain, retriever = legacy.construir_chain()
        _legacy["chain"], _legacy["retriever"] = chain, retriever
        return chain, retriever


def _safe_pdf_path(filename: str) -> Path:
    name = validate_safe_filename(filename)
    target = (PDF_DIR / name).resolve()
    if PDF_DIR not in target.parents and target != PDF_DIR:
        raise ValueError("Ruta no autorizada")
    return target


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ChatJUCHLM start env=%s sql_configured=%s", settings.APP_ENV, settings.sql_configured())
    yield
    log.info("ChatJUCHLM stop")


app = FastAPI(title="ChatJUCHLM API", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------- vistas ----------
@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html",
                                       context={"app_version": settings.APP_VERSION},
                                       headers={"Cache-Control": "no-store"})


@app.get("/documentos", response_class=HTMLResponse)
async def documents_interface(request: Request):
    return templates.TemplateResponse(request=request, name="documents.html",
                                       headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


# ---------- capabilities (§6) ----------
@app.get("/api/capabilities")
async def capabilities():
    return {"version": "1.0-mvp", "capabilities": Menu.get_capabilities()}


# ---------- conversación orquestada (§8, §25) ----------
@app.post("/api/conversation/message")
async def conversation_message(body: ConversationRequest, role: str = Depends(get_current_role)):
    orch = ConversationOrchestrator(session_manager, role=role)
    try:
        resp, sid = orch.handle(body.session_id, body.message, body.action_id)
        return {"session_id": sid, "response": resp.model_dump()}
    except Exception:
        log.exception("orchestrator falló")
        return JSONResponse(status_code=200, content={
            "session_id": body.session_id or "",
            "response": {"message": "No fue posible completar la consulta en este momento.",
                         "message_type": "assistant", "state": ConversationState.MAIN_MENU,
                         "actions": [], "data": None, "source": "orchestrator"}})


@app.post("/api/conversation/reset")
async def conversation_reset(body: ConversationRequest):
    s = session_manager.reset(body.session_id or "")
    orch = ConversationOrchestrator(session_manager)
    resp, sid = orch.handle(s.session_id, "", None)
    return {"session_id": sid, "response": resp.model_dump()}


# ---------- trámites (§17, seguro) ----------
@app.get("/api/tramites/constancia-usuario")
async def descargar_constancia(role: str = Depends(get_current_role)):
    from services import tramite_service as T
    from security import authorization as AuthZ
    if not AuthZ.can_access("tramite_constancia_usuario", role):
        return JSONResponse(status_code=403, content={"error": "Sin permiso"})
    try:
        path = T.resolve_tramite("constancia_usuario")
        return FileResponse(path=str(path), filename="constancia_usuario.pdf", media_type="application/pdf")
    except Exception as e:
        from core.exceptions import TramiteNotFoundError
        if isinstance(e, TramiteNotFoundError):
            return JSONResponse(status_code=404, content={"error": str(e.user_message)})
        log.exception("tramite descarga falló")
        return JSONResponse(status_code=400, content={"error": "Trámite no disponible."})


# ---------- trámites genérico (crece con capabilities.json) ----------
@app.get("/api/tramites/{tramite_slug}")
async def descargar_tramite(tramite_slug: str, role: str = Depends(get_current_role)):
    from services import tramite_service as T
    from security import authorization as AuthZ
    from core.exceptions import TramiteNotFoundError
    try:
        path = T.resolve_slug(tramite_slug)
    except TramiteNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e.user_message)})
    except Exception:
        log.exception("tramite descarga falló")
        return JSONResponse(status_code=400, content={"error": "Trámite no disponible."})
    cap = T.capability_for_slug(tramite_slug)
    if cap and not AuthZ.can_access(cap.get("id", ""), role):
        return JSONResponse(status_code=403, content={"error": "Sin permiso"})
    return FileResponse(path=str(path), filename=path.name, media_type="application/pdf")


# ---------- constancia generada (datos de sesión + predio elegido + fecha) ----------
@app.get("/api/tramites/generar/{tramite_slug}")
async def generar_constancia(tramite_slug: str, sid: str = "", role: str = Depends(get_current_role)):
    import io
    from services import constancia_pdf as CP
    from security import authorization as AuthZ
    from security import audit as Audit
    from core.session_manager import session_manager
    from core.models import ConversationState
    slug = (tramite_slug or "").strip().lower()
    tid = CP.TRAMITE_POR_SLUG.get(slug)
    if not tid:
        return JSONResponse(status_code=400, content={"error": "Trámite no disponible."})
    if not AuthZ.can_access(tid, role):
        return JSONResponse(status_code=403, content={"error": "Sin permiso"})
    sesion = session_manager.get(sid)
    if sesion is None:
        return JSONResponse(status_code=400, content={"error": "Sesión no válida o expirada."})
    ctx = sesion.context or {}
    datos = ctx.get("tramite_datos") or {}
    elegido = ctx.get("predio_elegido") or {}
    if (sesion.state != ConversationState.CONSTANCIA_USUARIO or ctx.get("tramite_id") != tid
            or not all(datos.get(k) for k in ("nombre", "dni", "celular")) or not elegido):
        return JSONResponse(status_code=400, content={"error": "Completa el flujo del trámite primero."})
    try:
        pdf = CP.generar_constancia_pdf(datos, elegido, tid)
    except Exception:
        log.exception("generar constancia falló")
        return JSONResponse(status_code=500, content={"error": "No fue posible generar el documento."})
    Audit.audit(sid, "tramite_pdf", tid, {**datos, "predio": elegido.get("codigo de riego", "")}, "ok")
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{CP.nombre_archivo(tid, datos.get("dni", ""))}"'})


# ---------- salud módulos ----------
@app.get("/api/health/rag")
async def health_rag():
    from services import rag_service as RAG
    return RAG.health()


@app.get("/api/health/sql")
async def health_sql():
    from database import sqlserver
    return sqlserver.health()


# ---------- legacy RAG SSE (compatibilidad, V11 intacto) ----------
@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = (data.get("question", "") or "").strip()
    if not question:
        return {"error": "Question is required"}

    def event_generator():
        import rag_juchlm as legacy
        chain, retriever = _legacy_chain()
        for chunk in legacy.preguntar_stream(chain, question, retriever):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------- gestión documental (endurecida §21, misma funcionalidad) ----------
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed"})
    try:
        dest = _safe_pdf_path(file.filename)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Nombre de archivo no permitido"})
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        import rag_juchlm as legacy
        legacy.sincronizar_carpeta()
        _legacy["chain"] = None  # invalidar chain lazy tras ingesta
    except Exception as e:
        log.warning("reindex tras upload falló: %s", e)
    return {"message": "File uploaded and processed successfully"}


@app.get("/api/documents")
async def list_documents():
    import rag_juchlm as legacy
    manifest = legacy.cargar_manifest()
    return {"documents": [{"name": n, "chunks": i.get("chunks", 0), "ocr": i.get("ocr", False)}
                          for n, i in manifest.items()]}


@app.delete("/api/delete/{filename}")
async def delete_document(filename: str):
    try:
        target = _safe_pdf_path(filename)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Nombre de archivo no permitido"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "Archivo no encontrado"})
    try:
        target.unlink()
        import rag_juchlm as legacy
        legacy.sincronizar_carpeta()
        _legacy["chain"] = None
        return {"message": f"'{filename}' eliminado correctamente de disco y ChromaDB"}
    except Exception as e:
        log.exception("delete falló")
        return JSONResponse(status_code=500, content={"error": "No fue posible completar la operación."})


@app.get("/api/download/{filename}")
async def download_document(filename: str):
    try:
        target = _safe_pdf_path(filename)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Nombre de archivo no permitido"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "Archivo no encontrado"})
    return FileResponse(path=str(target), filename=target.name, media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
