"""RAG Service (§10, §28). Encapsula V11 SIN modificar rag_juchlm.py.

El Orchestrator solo conoce ask(question). Detalles Chroma/Ollama quedan aquí.
"""
from __future__ import annotations
import logging
import threading

log = logging.getLogger("rag")

_chain = None
_chain_lock = threading.Lock()


def _get_chain():
    """Lazy init: no bloquear startup si Ollama/Chroma no están listos."""
    global _chain
    if _chain is not None:
        return _chain
    with _chain_lock:
        if _chain is not None:
            return _chain
        import rag_juchlm as legacy  # import diferido a propósito
        try:
            legacy.sincronizar_carpeta()
        except SystemExit:
            log.warning("pdfs/ no existe, RAG operará sin documentos")
        except Exception as e:
            log.warning("sincronizar_carpeta omitida: %s", e)
        _chain, _retriever = legacy.construir_chain()
        return _chain


def ask(question: str, timeout_note: str = "") -> dict:
    """Interfaz única. Mantiene pipeline V11 completo + validators. No altera cifras."""
    import rag_juchlm as legacy
    question = (question or "").strip()
    if not question:
        return {"answer": "Por favor escribe tu pregunta.", "sources": [], "area": "general"}
    try:
        chain = _get_chain()
        # ejecutar_pipeline ya aplica answerability gate; chain.stream aplica validators+repair
        resultado = legacy.ejecutar_pipeline(question)
        if not resultado.answerable or (not resultado.modo_resumen and not resultado.docs_evidencia):
            return {"answer": "No encontré información suficiente en los documentos indexados para responder esa pregunta.",
                    "sources": [], "area": resultado.ctx.area if resultado and resultado.ctx else "general"}
        if resultado.modo_resumen and resultado.fuente_resumen:
            chunks = legacy.obtener_todos_los_chunks_de_fuente(resultado.fuente_resumen)
            # construir llm dedicado para resumen (mismo modelo V11)
            llm = legacy.construir_llm()
            answer = legacy.generar_resumen_documento(llm, resultado.ctx, chunks)
            return {"answer": answer, "sources": [{"source": resultado.fuente_resumen, "area": resultado.ctx.area, "page": None}],
                    "area": resultado.ctx.area}
        # Camino normal: recolectar stream validado (ya incluye claim validator + repair)
        parts = list(chain.stream(question, docs=resultado.docs_evidencia))
        answer = "".join(parts).strip()
        sources = legacy.formatear_fuentes(resultado.docs_evidencia)
        return {"answer": answer, "sources": sources, "area": resultado.ctx.area}
    except Exception as e:
        log.exception("RAG ask falló: %s", e)
        return {"answer": "No fue posible completar la consulta en este momento.",
                "sources": [], "area": "general", "error": True}


def health() -> dict:
    try:
        import rag_juchlm as legacy
        m = legacy.cargar_manifest()
        return {"ok": True, "documentos": len(m)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
