import json
import os
import re
import hashlib
import sys
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# ============================================================
# RAG-JUCHLM v12 — DOCUMENTACIÓN GENERAL MULTI-ÁREA
# ============================================================
#
# Objetivo:
#   RAG local generalista para TODA la documentación institucional,
#   organizada por áreas (legal, administrativa, académica, técnica,
#   financiera, RR.HH., operativa, etc.), no solo expedientes legales.
#
# Estructura de carpetas esperada:
#
#   pdfs/
#     legal/            -> área "legal"
#     administrativo/   -> área "administrativo"
#     academico/        -> área "academico"
#     tecnico/          -> área "tecnico"
#     financiero/        -> área "financiero"
#     rrhh/             -> área "rrhh"
#     *.pdf (en la raíz) -> área "general"
#
#   Cualquier nombre de subcarpeta es válido: si no está en
#   AREA_REGISTRY se registra automáticamente como área nueva
#   con perfiles genéricos (fecha, monto, lugar, tabla, página,
#   persona) y sin vocabulario especializado adicional.
#
# Mantiene compatibilidad con app.py:
#   - PDF_FOLDER
#   - cargar_manifest()
#   - sincronizar_carpeta()
#   - construir_chain()
#   - preguntar_stream()
#   - ChainCompatible.stream(question, docs=None)
#   - ChainCompatible.invoke(question)
#
# Arquitectura (10 pasos, igual en todas las áreas):
#   1. Query Analysis        -> intención, entidades, restricciones,
#                                página, área, subpreguntas
#   2. Query Expansion       -> variantes dirigidas por área/perfil
#   3. Retrieval híbrido     -> 3A semántico (embeddings + Chroma)
#                                3B lexical (entidades + tokens)
#   4. Hybrid Ranking        -> semantic + lexical + entity + page + source
#   5. Document Router       -> documento dominante / múltiples / comparación
#   6. Evidence Selection    -> chunks vecinos, cobertura de página, diversidad
#   7. Answerability Gate    -> ¿hay evidencia?, ¿es suficiente?, ¿responde?
#   8. LLM Response          -> generación restringida al contexto
#   9. Claim Validator       -> entidades, fechas, números, relaciones, fuentes
#  10. Repair / Revalidación -> respuesta grounded o rechazo explícito
#
# ============================================================


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PDF_FOLDER = Path("pdfs")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "local-rag"
MANIFEST_PATH = Path(CHROMA_PATH) / "manifest.json"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen3.5:4b"

# Item 20 (auditoría hardening): versión unificada. RAG_VERSION es la
# versión "humana" del sistema completo (banner, logs, diagnóstico).
# INDEX_SCHEMA_VERSION es SOLO el versionado del esquema de metadata
# indexado en Chroma -- cambia cuando cambia la estructura de un
# chunk/documento indexado (obliga a reprocesar todo el corpus).
# No son el mismo número a propósito: RAG_VERSION puede subir sin
# tocar el índice (ej. un cambio de prompt), pero cuando el índice sí
# cambia de forma, ambos se actualizan juntos para que quede trazado.
RAG_VERSION = "12.1"
INDEX_SCHEMA_VERSION = 12
CURRENT_INDEX_SCHEMA_VERSION = INDEX_SCHEMA_VERSION  # alias retrocompatible

DEFAULT_AREA = "general"

# ============================================================
# PILOT MODE (item 29, auditoría hardening)
# ============================================================
#
# Con PILOT_MODE=True se desactivan los validadores LLM y la
# reparación automática, para medir primero accuracy/groundedness/
# coverage/latencia con validación puramente heurística (que sigue
# activa vía VALIDATOR_ENABLED). Los validadores LLM se activan
# selectivamente después, ya con datos reales de qué tan seguido
# hacen falta. PILOT_MODE se aplica una sola vez, al cargar el
# módulo, sobre los flags de la sección VALIDACIÓN de abajo.
# ============================================================
PILOT_MODE = False


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 120


# ============================================================
# RETRIEVAL
# ============================================================

RETRIEVER_K = 40
SEMANTIC_RESULTS_PER_QUERY = 12
LEXICAL_RESULTS_PER_QUERY = 12

FINAL_CONTEXT_CHUNKS = 8
MAX_EVIDENCE_PAGES = 4
MAX_NEIGHBOR_CHUNKS_PER_HIT = 3
NEIGHBOR_RADIUS = 1

QUERY_EXPANSION_MAX = 8
MAX_DOCUMENTS_COMPARISON = 2

# Penalización/bonificación de ranking
SEMANTIC_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.30
RANK_WEIGHT = 0.15


# ============================================================
# LLM
# ============================================================

MAX_TOKENS_RESPUESTA = 4096
CONTEXTO_LLM = 8192
TEMPERATURE = 0


# ============================================================
# OCR
# ============================================================

OCR_ENABLED = True
SCANNED_THRESHOLD_CHARS_POR_PAGINA = 30
OCR_DPI = 200


# ============================================================
# DETECCIÓN DOCUMENTAL
# ============================================================

MIN_LEN_STEM_MATCH = 8
MIN_TOKEN_MATCHES = 2
MIN_TOKEN_PROPORTION = 0.35
LONGITUD_TOKEN_BONO = 7

BONO_LEXICAL_CHUNK = 0.15
BONO_LEXICAL_DOCUMENTO = 0.20


# ============================================================
# ANSWERABILITY
# ============================================================

ANSWERABILITY_ENABLED = True
ANSWERABILITY_MIN_GENERAL = 0.43
ANSWERABILITY_MIN_FACTUAL = 0.50
ANSWERABILITY_MIN_SENSITIVE = 0.52
ANSWERABILITY_MIN_LEXICAL_WITHOUT_RESTRICTION = 0.05
ANSWERABILITY_SEMANTIC_ONLY_MIN = 0.58


# ============================================================
# VALIDACIÓN
# ============================================================

VALIDATOR_ENABLED = True
CLAIM_VALIDATOR_LLM_ENABLED = True
COMPLETENESS_VALIDATOR_LLM_ENABLED = True
REPAIR_UNSUPPORTED_CLAIMS = True
REPAIR_INCOMPLETE_ANSWER = True

MAX_CLAIMS_TO_VERIFY = 10          # claims evaluados HEURÍSTICAMENTE (barato, siempre)
# Item 11 (auditoría hardening): "claim budget" -- de los claims que
# la heurística deja ambiguos/sensibles, solo los N más riesgosos se
# mandan al LLM validator. El resto queda con el veredicto heurístico.
MAX_LLM_CLAIMS_TO_VERIFY = 3
# Item 14: circuit breaker -- tope duro de llamadas LLM de VALIDACIÓN
# (claim validator + completeness validator + repair) por respuesta
# completa. No cuenta la llamada de generación inicial. Al agotarse,
# se detienen más validaciones LLM, se conserva la heurística, y la
# respuesta se marca con validation_budget_exhausted=True en vez de
# fallar o colgarse en latencia.
MAX_VALIDATION_LLM_CALLS = 4
MIN_CLAIM_TOKEN_OVERLAP = 0.22
MIN_CLAIM_TOKEN_OVERLAP_STRICT = 0.40
# Item 13: la reparación ya solo se intenta una vez en el flujo actual
# (generar -> validar -> reparar -> revalidar -> fallback grounded),
# pero lo hacemos explícito y configurable en vez de estar implícito
# en la estructura del código.
MAX_REPAIR_ATTEMPTS = 1


# Item 29/36 (auditoría hardening): PILOT_MODE se aplica UNA vez, acá,
# sobrescribiendo los flags de arriba -- así el resto del código sigue
# leyendo las mismas variables de siempre (CLAIM_VALIDATOR_LLM_ENABLED,
# etc.) sin tener que preguntar "¿estamos en piloto?" en cada función.
if PILOT_MODE:
    CLAIM_VALIDATOR_LLM_ENABLED = False
    COMPLETENESS_VALIDATOR_LLM_ENABLED = False
    REPAIR_UNSUPPORTED_CLAIMS = False
    REPAIR_INCOMPLETE_ANSWER = False
    # print() y no log(): log() se define más abajo en el archivo y
    # este bloque corre a nivel de módulo, antes de esa definición.
    print(
        "PILOT_MODE activo: validadores LLM y reparación automática "
        "desactivados. Validación heurística (VALIDATOR_ENABLED) sigue activa.",
        flush=True,
    )


# ============================================================
# CACHE LEXICAL
# ============================================================
#
# Item 18 (auditoría): tanto LEXICAL_CACHE como _OCR_ENGINE (más abajo)
# son estado global mutable, compartido si app.py corre bajo uvicorn
# con varios workers en modo threaded o gunicorn con threads (no con
# procesos separados, donde cada worker tiene su propio intérprete
# y por tanto su propia copia de estos globals -- ahí no hace falta
# lock, pero protegerlo no tiene costo real y evita un bug sutil si
# el modelo de despliegue cambia).
# - vector_db (cliente de Chroma) es seguro para uso concurrente
#   según la librería, no se protege aquí.
# - LEXICAL_CACHE se construye una vez y se lee muchas veces: el lock
#   solo protege la construcción/invalidación, no cada lectura.
# ============================================================

LEXICAL_CACHE = None
LEXICAL_CACHE_LOCK = threading.Lock()
LEXICAL_INDEX_MAX_POSTING_DOCS = 50000
# Item 5 (auditoría): por encima de este número de chunks, el cache
# lexical en memoria (todo el corpus + postings) empieza a ser un
# riesgo real de RAM. No migramos a un índice persistente todavía
# (no está justificado por el tamaño actual), pero sí avisamos fuerte
# para que se decida ANTES de que el proceso se quede sin memoria.
LEXICAL_CACHE_WARN_CHUNKS = 20000


# ============================================================
# UTILIDADES
# ============================================================


def log(msg: str):
    print(msg, flush=True)


def normalizar_texto(texto: str) -> str:
    texto = str(texto or "").lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(
        c for c in texto if not unicodedata.combining(c)
    )
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "por", "para", "con", "sin",
    "que", "cual", "cuales", "como", "donde", "cuando", "quien", "quienes",
    "es", "son", "fue", "fueron", "se", "su", "sus", "sobre", "segun",
    "este", "esta", "estos", "estas", "mi", "me", "te", "y", "o", "u",
    "lo", "hay", "aparece", "dice", "dime", "indica", "informacion",
    "documento", "documentos", "exacta", "exacto", "exactamente", "puedes",
    "podria", "podrias", "respuesta", "responde", "siguiente", "siguientes",
    "favor", "pagina", "pag", "pags", "area", "areas",
}


def tokens_significativos(texto: str) -> set[str]:
    return {
        token
        for token in normalizar_texto(texto).split()
        if len(token) >= 3 and token not in _STOPWORDS
    }


def extraer_numeros(texto: str) -> set[str]:
    return set(
        re.findall(r"\b\d[\d\.\-/]*\d\b|\b\d+\b", str(texto or ""))
    )


def dividir_en_frases(texto: str) -> list[str]:
    texto = str(texto or "").strip()
    if not texto:
        return []
    partes = re.split(r"(?<=[\.!?])\s+|\n+", texto)
    return [p.strip(" -•\t") for p in partes if p.strip(" -•\t")]


def limpiar_fuentes_generadas(texto: str) -> str:
    return re.split(
        r"\n\s*(?:\*\*)?Fuentes(?:\*\*)?\s*:",
        str(texto or ""),
        flags=re.IGNORECASE,
    )[0].strip()


def fingerprint_doc(doc: Document) -> tuple:
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    chunk = doc.metadata.get("chunk_index")
    content_hash = hashlib.md5(
        (doc.page_content or "").encode("utf-8", errors="ignore")
    ).hexdigest()
    return source, page, chunk, content_hash


# ============================================================
# REGISTRO DE ÁREAS INSTITUCIONALES
# ============================================================
#
# Cada área define:
#   - aliases: formas en que el usuario puede nombrarla en la pregunta
#     (p. ej. "en el área legal", "documentos de rrhh")
#   - profile_terms: vocabulario adicional por perfil de consulta,
#     SOLO relevante dentro de esa área. Se suma al vocabulario
#     genérico (GENERIC_PROFILE_TERMS), que aplica a todas las áreas.
#   - sensitive_profiles: perfiles de esa área que exigen
#     correspondencia exacta de identificadores/entidades (igual
#     de estricto que antes se hacía solo para lo judicial).
#
# Una subcarpeta nueva bajo pdfs/ que no esté aquí se registra
# igualmente como área -> queda con vocabulario genérico y sin
# perfiles sensibles adicionales, sin necesidad de tocar código.
# ============================================================

GENERIC_PROFILE_TERMS = {
    "fecha": [
        "fecha", "dia", "mes", "ano", "emision", "expedicion", "registro",
    ],
    "monto": [
        "monto", "importe", "suma", "cantidad", "soles", "usd", "dolares",
        "presupuesto", "costo", "valor", "precio",
    ],
    "lugar": [
        "lugar", "ubicacion", "direccion", "domicilio", "sede", "distrito",
        "provincia", "ciudad", "interseccion", "avenida", "calle", "local",
    ],
    "tabla": [
        "tabla", "cuadro", "filas", "columnas", "datos", "total",
    ],
    "pagina": [
        "pagina", "pag", "contenido pagina", "aparece pagina", "dice pagina",
    ],
    "persona": [
        "nombre", "autor", "responsable", "encargado", "titular",
        "representante", "firmante",
    ],
    "identificador": [
        "codigo", "numero", "identificador", "referencia", "n",
    ],
    "resultado": [
        "resultado", "conclusion", "decision", "acuerdo", "determino",
        "acordo", "aprobado", "rechazado",
    ],
}

AREA_REGISTRY: dict[str, dict] = {
    "legal": {
        "aliases": ["legal", "juridico", "judicial", "expedientes legales"],
        "profile_terms": {
            "dni": [
                "dni", "documento nacional", "identidad",
                "numero de identidad", "documento de identidad",
                "identificacion",
            ],
            "acusado": [
                "acusado", "imputado", "procesado", "encausado", "investigado",
            ],
            "agraviado": [
                "agraviado", "afectado", "perjudicado", "victima",
                "institucion", "entidad afectada",
            ],
            "juez": [
                "juez", "jueza", "magistrado", "magistrada", "juzgado",
                "tribunal",
            ],
            "expediente": [
                "expediente", "numero expediente", "n expediente",
                "codigo expediente", "proceso", "caso",
            ],
            "resolucion": [
                "resolucion", "numero resolucion", "resolucion judicial",
                "acto administrativo",
            ],
            "delito": [
                "delito", "infraccion", "falta", "tipo penal",
                "calificacion juridica", "cargo", "imputacion",
            ],
            "fecha_hechos": [
                "fecha hechos", "ocurrio", "ocurrieron", "sucedio",
                "acontecio", "hechos", "evento",
            ],
            "fecha_sentencia": [
                "fecha sentencia", "emision sentencia", "fecha resolucion",
                "resolucion emitida", "sentencia emitida",
            ],
        },
        "sensitive_profiles": {
            "dni", "acusado", "agraviado", "juez", "expediente",
            "resolucion", "delito", "fecha_hechos", "fecha_sentencia",
        },
    },
    "administrativo": {
        "aliases": [
            "administrativo", "administracion", "gestion documental",
            "tramite documentario",
        ],
        "profile_terms": {
            "expediente": [
                "expediente administrativo", "numero de expediente",
                "codigo de expediente", "tramite",
            ],
            "resolucion": [
                "resolucion administrativa", "resolucion de gerencia",
                "resolucion de directorio", "acuerdo de directorio",
            ],
            "oficio": [
                "oficio", "numero de oficio", "oficio circular",
            ],
            "memorando": [
                "memorando", "memorandum", "numero de memorando",
            ],
            "responsable": [
                "gerente", "funcionario", "encargado de area",
                "jefe de area", "responsable del tramite",
            ],
        },
        "sensitive_profiles": {"expediente", "resolucion", "oficio", "memorando"},
    },
    "academico": {
        "aliases": ["academico", "universidad", "unprg", "cursos", "docencia"],
        "profile_terms": {
            "docente": [
                "docente", "profesor", "profesora", "catedratico",
                "asesor", "asesora",
            ],
            "curso": [
                "curso", "asignatura", "silabo", "syllabus", "malla curricular",
                "unidad de aprendizaje",
            ],
            "calificacion": [
                "nota", "calificacion", "promedio", "puntaje", "evaluacion",
            ],
            "alumno": [
                "alumno", "estudiante", "matriculado", "egresado",
            ],
        },
        "sensitive_profiles": {"calificacion"},
    },
    "tecnico": {
        "aliases": [
            "tecnico", "ingenieria", "especificaciones", "manual tecnico",
        ],
        "profile_terms": {
            "especificacion": [
                "especificacion", "requisito tecnico", "norma tecnica",
                "estandar", "parametro",
            ],
            "procedimiento": [
                "procedimiento", "protocolo", "instructivo", "manual",
                "guia tecnica",
            ],
            "version": [
                "version", "revision", "release", "actualizacion",
            ],
        },
        "sensitive_profiles": {"especificacion"},
    },
    "financiero": {
        "aliases": ["financiero", "finanzas", "presupuesto", "contabilidad"],
        "profile_terms": {
            "factura": [
                "factura", "numero de factura", "comprobante de pago",
                "boleta",
            ],
            "presupuesto": [
                "presupuesto", "partida presupuestal", "asignacion",
                "ejecucion presupuestal",
            ],
            "pago": [
                "pago", "desembolso", "transferencia", "abono",
            ],
        },
        "sensitive_profiles": {"factura", "presupuesto"},
    },
    "rrhh": {
        "aliases": [
            "rrhh", "recursos humanos", "personal", "planillas",
        ],
        "profile_terms": {
            "contrato": [
                "contrato", "numero de contrato", "tipo de contrato",
                "renovacion",
            ],
            "trabajador": [
                "trabajador", "empleado", "colaborador", "personal",
                "servidor",
            ],
            "cargo": [
                "cargo", "puesto", "funcion", "categoria",
            ],
            "planilla": [
                "planilla", "remuneracion", "sueldo", "bonificacion",
            ],
        },
        "sensitive_profiles": {"contrato", "planilla"},
    },
    DEFAULT_AREA: {
        "aliases": ["general", "institucional"],
        "profile_terms": {},
        "sensitive_profiles": set(),
    },
}


def _registrar_area_si_no_existe(area: str):
    if area not in AREA_REGISTRY:
        AREA_REGISTRY[area] = {
            "aliases": [area.replace("_", " ")],
            "profile_terms": {},
            "sensitive_profiles": set(),
        }


def _perfiles_de_area(area: str) -> dict:
    """Vocabulario disponible para un área: genérico + específico del área."""
    _registrar_area_si_no_existe(area)
    combinado = dict(GENERIC_PROFILE_TERMS)
    combinado.update(AREA_REGISTRY[area].get("profile_terms", {}))
    return combinado


def _perfiles_sensibles_de_area(area: str) -> set[str]:
    _registrar_area_si_no_existe(area)
    return set(AREA_REGISTRY[area].get("sensitive_profiles", set()))


def _detectar_area_en_pregunta(q_norm: str) -> Optional[str]:
    for area_id, cfg in AREA_REGISTRY.items():
        for alias in cfg.get("aliases", []):
            alias_norm = normalizar_texto(alias)
            if alias_norm and re.search(rf"\b{re.escape(alias_norm)}\b", q_norm):
                return area_id
    return None


def listar_areas_disponibles() -> list[str]:
    manifest = cargar_manifest()
    areas = {info.get("area", DEFAULT_AREA) for info in manifest.values()}
    areas.update(AREA_REGISTRY.keys())
    return sorted(areas)


# ============================================================
# MANIFEST
# ============================================================


def cargar_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        contenido = MANIFEST_PATH.read_text(encoding="utf-8")
        if not contenido.strip():
            return {}
        return json.loads(contenido)
    except Exception as e:
        # P0-2 (auditoría): un manifest corrupto NO se trata como
        # "sin documentos indexados" para efectos de escritura -- eso
        # llevaría a sincronizar_carpeta() a reindexar todo y luego
        # SOBRESCRIBIR el manifest bueno-pero-ilegible con uno nuevo,
        # perdiendo metadata (content_hash, ocr, etc.) que ya era
        # correcta en Chroma. Se detiene el proceso explícitamente en
        # vez de continuar con un estado ambiguo.
        log(
            f"ERROR CRÍTICO: manifest.json existe pero está corrupto: {e}. "
            f"No se continúa para evitar sobrescribir el estado real de Chroma "
            f"con un manifest vacío. Revisa/restaura manualmente "
            f"'{MANIFEST_PATH}' (o su copia .bak si existe) antes de reintentar."
        )
        respaldo = MANIFEST_PATH.with_suffix(".json.bak")
        if respaldo.exists():
            try:
                return json.loads(respaldo.read_text(encoding="utf-8"))
            except Exception:
                pass
        raise RuntimeError(f"manifest.json corrupto e irrecuperable: {e}") from e


def guardar_manifest(manifest: dict):
    # P0-2 (auditoría): escritura atómica. Un corte de proceso a
    # mitad de escritura NUNCA debe dejar manifest.json truncado --
    # se escribe primero a un archivo temporal, se fuerza a disco
    # (fsync) y solo entonces se reemplaza el archivo final de forma
    # atómica (os.replace es atómico en POSIX y Windows modernos).
    # Se conserva además un .bak del manifest previo por si el nuevo
    # resultara -- por cualquier motivo -- inválido.
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists():
        try:
            respaldo = MANIFEST_PATH.with_suffix(".json.bak")
            respaldo.write_bytes(MANIFEST_PATH.read_bytes())
        except Exception as e:
            log(f"ADVERTENCIA: no se pudo actualizar manifest.json.bak: {e}")

    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    contenido = json.dumps(manifest, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(contenido)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, MANIFEST_PATH)


def hash_archivo(path: Path) -> str:
    """P1 (auditoría): hash de CONTENIDO real (SHA-256), no de
    tamaño+mtime. El esquema anterior no detectaba: (a) un PDF
    reemplazado por otro de igual tamaño con mtime manipulado, ni
    (b) evitaba reprocesar cuando solo cambiaba el mtime sin cambiar
    contenido. Se lee en bloques para no cargar el PDF completo en
    memoria; el costo (unos ms por MB) es despreciable frente al
    costo real de OCR/embeddings que ya se paga al reprocesar."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(bloque)
    return sha256.hexdigest()


def _area_de_path(path: Path) -> str:
    """El área es el primer subdirectorio bajo PDF_FOLDER. Si el PDF
    está directamente en la raíz, el área es DEFAULT_AREA."""
    try:
        relativo = path.relative_to(PDF_FOLDER)
    except ValueError:
        return DEFAULT_AREA
    partes = relativo.parts
    if len(partes) <= 1:
        return DEFAULT_AREA
    return normalizar_texto(partes[0]).replace(" ", "_") or DEFAULT_AREA


def _source_id_de_path(path: Path) -> str:
    """Identificador estable del documento: ruta relativa a PDF_FOLDER
    en formato posix, para evitar colisiones entre áreas distintas
    que tengan archivos con el mismo nombre."""
    try:
        return path.relative_to(PDF_FOLDER).as_posix()
    except ValueError:
        return path.name


# ============================================================
# PDF / OCR
# ============================================================


def cargar_pdf_texto(path: Path) -> list[Document]:
    try:
        return PyPDFLoader(str(path), mode="page").load()
    except TypeError:
        return PyPDFLoader(str(path)).load()


def es_probable_escaneado(documents: list[Document]) -> bool:
    if not documents:
        return False
    total = sum(len(d.page_content.strip()) for d in documents)
    promedio = total / len(documents)
    return promedio < SCANNED_THRESHOLD_CHARS_POR_PAGINA


_OCR_ENGINE = None
_OCR_ENGINE_LOCK = threading.Lock()


def _get_ocr_engine():
    global _OCR_ENGINE
    # Item 18 (auditoría): doble-check locking -- evita inicializar el
    # engine dos veces si dos requests concurrentes llegan a la vez
    # con _OCR_ENGINE aún en None (RapidOCR carga un modelo ONNX;
    # inicializarlo dos veces desperdicia memoria y tiempo, aunque no
    # corrompe nada).
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        _OCR_ENGINE = _inicializar_ocr_engine()
    return _OCR_ENGINE


def _inicializar_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise RuntimeError(
            "Falta rapidocr-onnxruntime. "
            "Instala: pip install rapidocr-onnxruntime"
        ) from e
    _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr_pagina_con_rapidocr(pdf_path: Path, page_index: int) -> tuple[str, Optional[float]]:
    try:
        import fitz
        import numpy as np
        from PIL import Image
        from io import BytesIO
    except ImportError as e:
        raise RuntimeError(
            "Faltan dependencias OCR. "
            "Instala: pip install pymupdf pillow numpy"
        ) from e

    engine = _get_ocr_engine()
    pdf = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= pdf.page_count:
            return ""
        page = pdf.load_page(page_index)
        escala = OCR_DPI / 72.0
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(escala, escala),
            alpha=False,
        )
        imagen = Image.open(
            BytesIO(pixmap.tobytes("png"))
        ).convert("RGB")
        imagen_array = np.array(imagen)
        resultado = engine(imagen_array)
        resultados = resultado[0] if isinstance(resultado, tuple) else resultado
        if not resultados:
            return "", None

        textos = []
        confidencias = []
        for item in resultados:
            if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            texto = str(item[1] or "").strip()
            if texto:
                textos.append(texto)
            # item[2] = confianza de RapidOCR cuando está disponible.
            if len(item) >= 3:
                try:
                    confidencias.append(float(item[2]))
                except (TypeError, ValueError):
                    pass

        confianza_promedio = sum(confidencias) / len(confidencias) if confidencias else None
        return "\n".join(textos).strip(), confianza_promedio
    finally:
        pdf.close()


def cargar_pdf(path: Path) -> tuple[list[Document], bool]:
    docs = cargar_pdf_texto(path)
    if not docs:
        return [], False

    # source_type por página (item 4/20): "text" salvo que se reemplace por OCR.
    for doc in docs:
        doc.metadata["source_type"] = "text"

    paginas_ocr = [
        i for i, doc in enumerate(docs)
        if OCR_ENABLED and len((doc.page_content or "").strip()) < SCANNED_THRESHOLD_CHARS_POR_PAGINA
    ]

    if not paginas_ocr:
        return docs, False

    log(f"    -> {len(paginas_ocr)} página(s) requieren OCR.")
    fue_ocr = False

    try:
        _get_ocr_engine()
        log("    -> RapidOCR disponible.")
    except Exception as e:
        log(f"    -> OCR no disponible: {e}")
        return docs, False

    for page_index in paginas_ocr:
        try:
            texto_ocr, confianza = _ocr_pagina_con_rapidocr(path, page_index)
            original = docs[page_index].page_content or ""
            if len(texto_ocr) > len(original.strip()):
                docs[page_index].page_content = texto_ocr
                docs[page_index].metadata["source_type"] = "ocr"
                if confianza is not None:
                    docs[page_index].metadata["ocr_confidence"] = round(confianza, 4)
                fue_ocr = True
                nota_confianza = f", confianza {confianza:.2f}" if confianza is not None else ""
                log(
                    f"    -> OCR página {page_index + 1}: "
                    f"{len(texto_ocr)} caracteres{nota_confianza}."
                )
        except Exception as e:
            log(f"    -> ERROR OCR página {page_index + 1}: {e}")

    return docs, fue_ocr


# ============================================================
# EMBEDDINGS / CHROMA / SPLITTER
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL,
    num_gpu=999,
)

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # v12: separadores jerárquicos -> intenta cortar ANTES de un
    # encabezado legal/estructural en vez de partirlo a la mitad
    # (ej. "Artículo 366°..." no debe quedar cortado entre dos chunks).
    separators=[
        "\n\nArtículo ", "\nArtículo ", "\n\nART. ", "\nART. ",
        "\n\nCAPÍTULO", "\n\nCAPITULO", "\n\nSECCIÓN", "\n\nSECCION",
        "\n\nTÍTULO", "\n\nTITULO",
        "\n\n", "\n", ". ", " ", "",
    ],
)


def invalidar_cache_lexical():
    global LEXICAL_CACHE
    with LEXICAL_CACHE_LOCK:
        LEXICAL_CACHE = None


# ============================================================
# INGESTA (recorre TODAS las áreas / subcarpetas)
# ============================================================


def sincronizar_carpeta():
    if not PDF_FOLDER.exists():
        log(f"\nERROR: no existe la carpeta '{PDF_FOLDER.absolute()}'")
        raise SystemExit(1)

    pdfs_en_disco = sorted(PDF_FOLDER.rglob("*.pdf"))
    manifest = cargar_manifest()

    mapa_paths = {_source_id_de_path(p): p for p in pdfs_en_disco}
    nombres_en_disco = set(mapa_paths.keys())
    cambios = False

    # --------------------------------------------------------
    # Eliminados
    # --------------------------------------------------------
    eliminados = [n for n in manifest if n not in nombres_en_disco]
    for nombre in eliminados:
        log(f"\nEliminando de Chroma: '{nombre}'")
        try:
            vector_db.delete(where={"source": nombre})
        except Exception as e:
            log(f"    -> Advertencia: {e}")
        del manifest[nombre]
        cambios = True

    if not pdfs_en_disco:
        guardar_manifest(manifest)
        invalidar_cache_lexical()
        log(f"\nNo hay PDFs en '{PDF_FOLDER.absolute()}' (ni en sus subcarpetas de área).")
        return

    # --------------------------------------------------------
    # Nuevos / modificados / cambio de schema
    # --------------------------------------------------------
    a_procesar = []
    for source_id, path in mapa_paths.items():
        info = manifest.get(source_id, {})
        firma_actual = hash_archivo(path)
        firma_previa = info.get("firma")
        version = info.get("index_schema_version", 0)
        if firma_actual != firma_previa or version != CURRENT_INDEX_SCHEMA_VERSION:
            a_procesar.append((source_id, path))

    if not a_procesar and not cambios:
        areas_presentes = sorted({_area_de_path(p) for p in pdfs_en_disco})
        log(
            f"\nBase vectorial ya actualizada. "
            f"{len(pdfs_en_disco)} PDFs indexados en {len(areas_presentes)} área(s) "
            f"({', '.join(areas_presentes)}), sin cambios."
        )
        return

    log(f"\n{len(a_procesar)} PDF(s) serán procesados.")

    for i, (source_id, path) in enumerate(a_procesar, 1):
        area = _area_de_path(path)
        log(f"\n[{i}/{len(a_procesar)}] [{area}] {source_id}")
        inicio = time.time()

        # --------------------------------------------------------
        # P0-1 (auditoría): NO borrar antes de tener la versión nueva
        # insertada y confirmada. Capturamos los IDs de la versión
        # ANTERIOR (si existe) para poder borrarlos explícitamente
        # DESPUÉS -- nunca antes -- de que add_documents() confirme
        # éxito. Si cualquier paso intermedio falla (OCR, chunking,
        # embeddings, Chroma), el documento anterior permanece intacto
        # y consultable.
        # --------------------------------------------------------
        ids_version_anterior: list[str] = []
        try:
            existentes = vector_db.get(where={"source": source_id}, include=[])
            ids_version_anterior = existentes.get("ids") or []
        except Exception as e:
            log(f"    -> Advertencia consultando versión anterior en Chroma: {e}")

        try:
            docs, fue_ocr = cargar_pdf(path)
        except Exception as e:
            log(f"    -> ERROR procesando '{source_id}': {e}")
            continue

        if not docs:
            log(f"    -> '{source_id}' no contiene páginas.")
            continue

        for page_index, doc in enumerate(docs):
            doc.metadata["source"] = source_id
            doc.metadata["document_id"] = path.stem
            doc.metadata["area"] = area
            doc.metadata["page"] = page_index
            doc.metadata["page_number"] = page_index + 1
            doc.metadata["ocr"] = fue_ocr
            doc.metadata["index_schema_version"] = CURRENT_INDEX_SCHEMA_VERSION

        chunks = splitter.split_documents(docs)
        if not chunks:
            log(f"    -> '{source_id}' no generó chunks. Se conserva la versión anterior (si existía).")
            continue

        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["index_schema_version"] = CURRENT_INDEX_SCHEMA_VERSION

        # Item 23: hash de CONTENIDO (no del archivo) para avisar de
        # posibles duplicados. Solo informativo: nunca se borra nada
        # automáticamente, porque dos documentos distintos pueden
        # legítimamente compartir texto (ej. una plantilla oficial).
        contenido_hash = hashlib.md5(
            "".join(d.page_content or "" for d in docs).encode("utf-8", errors="ignore")
        ).hexdigest()
        posibles_duplicados = [
            otro for otro, info in manifest.items()
            if otro != source_id and info.get("content_hash") == contenido_hash
        ]
        if posibles_duplicados:
            log(
                f"    -> ADVERTENCIA: '{source_id}' tiene el mismo contenido "
                f"que: {', '.join(posibles_duplicados)}. Revisar si es un duplicado real."
            )

        # Token único por corrida de ingesta: garantiza que los IDs
        # nuevos NUNCA colisionan con los IDs antiguos (aunque el
        # contenido sea idéntico), así add_documents() nunca falla
        # por "ID ya existe" y no necesitamos borrar antes de insertar.
        ingest_token = uuid.uuid4().hex[:8]
        ids = [
            f"{source_id}::page-{c.metadata.get('page', 0)}::chunk-{i}::{ingest_token}"
            for i, c in enumerate(chunks)
        ]

        try:
            vector_db.add_documents(documents=chunks, ids=ids)
        except Exception as e:
            log(
                f"    -> ERROR indexando '{source_id}': {e}. "
                f"La versión anterior (si existía) NO fue tocada."
            )
            continue

        # Solo AHORA, con la versión nueva confirmada en Chroma,
        # eliminamos explícitamente los IDs de la versión anterior
        # (nunca por filtro "where source" amplio, que también
        # borraría los chunks recién insertados si compartieran
        # fuente).
        if ids_version_anterior:
            try:
                vector_db.delete(ids=ids_version_anterior)
            except Exception as e:
                log(
                    f"    -> ADVERTENCIA: no se pudo limpiar la versión "
                    f"anterior de '{source_id}' ({e}). Pueden quedar "
                    f"chunks duplicados temporalmente; no hay pérdida de datos."
                )

        manifest[source_id] = {
            "firma": hash_archivo(path),
            "content_hash": contenido_hash,
            "area": area,
            "chunks": len(chunks),
            "pages": len(docs),
            "ocr": fue_ocr,
            "index_schema_version": CURRENT_INDEX_SCHEMA_VERSION,
        }
        cambios = True

        log(
            f"    -> {len(chunks)} chunks | {len(docs)} páginas | "
            f"{'OCR' if fue_ocr else 'texto'} | "
            f"{time.time() - inicio:.2f}s"
        )

    guardar_manifest(manifest)
    if cambios:
        invalidar_cache_lexical()

    areas_finales = sorted({info.get("area", DEFAULT_AREA) for info in manifest.values()})
    log(
        "\nSincronización completa. "
        f"Total PDFs registrados: {len(manifest)} en {len(areas_finales)} área(s): "
        f"{', '.join(areas_finales) if areas_finales else '—'}"
    )


def listar_indexados():
    manifest = cargar_manifest()
    if not manifest:
        print("\nNo hay PDFs indexados todavía.")
        return

    por_area = defaultdict(list)
    for nombre, info in manifest.items():
        por_area[info.get("area", DEFAULT_AREA)].append((nombre, info))

    print(f"\n{len(manifest)} PDF(s) registrados en {len(por_area)} área(s):\n")
    for area in sorted(por_area):
        print(f"[{area}]")
        for nombre, info in sorted(por_area[area]):
            tipo = "OCR" if info.get("ocr") else "texto"
            print(
                f"  - {nombre} | {info.get('chunks', 0)} chunks | "
                f"{info.get('pages', '?')} páginas | {tipo} | "
                f"schema {info.get('index_schema_version', '?')}"
            )


# ============================================================
# QUERY CONTEXT
# ============================================================


@dataclass
class QueryContext:
    question: str
    q_tokens: set[str] = field(default_factory=set)
    area: str = DEFAULT_AREA
    area_explicita: Optional[str] = None
    fuente_explicita: Optional[str] = None
    fuentes_explicitas: list[str] = field(default_factory=list)
    pagina_explicita: Optional[int] = None
    ultima_pagina_solicitada: bool = False
    es_comparacion: bool = False
    tipo_consulta: str = "general"
    subpreguntas: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    perfil: str = "general"
    focus_terms: set[str] = field(default_factory=set)
    sensitive: bool = False
    requires_completeness: bool = True
    es_resumen: bool = False


@dataclass
class PipelineResult:
    ctx: QueryContext
    docs_candidatos: list[Document]
    ranking_documentos: list[tuple[float, str, list[Document]]]
    docs_aislados: list[Document]
    docs_evidencia: list[Document]
    scores_evidencia: list[tuple[float, Document]] = field(default_factory=list)
    answerable: bool = False
    answerability_score: float = 0.0
    answerability_reason: str = ""
    modo_resumen: bool = False
    fuente_resumen: Optional[str] = None
    metricas: dict = field(default_factory=dict)


# ============================================================
# BATCH MULTI-PREGUNTA (items 3-9, 26 · auditoría hardening)
# ============================================================
#
# Antes, cuando la pregunta traía varias subpreguntas ("¿expediente?,
# ¿imputado?, ¿agraviado?, ¿fecha?"), TODAS competían por el mismo
# ranking global y los mismos FINAL_CONTEXT_CHUNKS=8 -- una subpregunta
# muy "afín" semánticamente podía monopolizar la evidencia y dejar a
# las demás sin contexto, aunque el LLM de generación igual intentara
# responderlas todas dentro del mismo prompt.
#
# Ahora, cuando se detectan 2+ subpreguntas, cada una corre su PROPIO
# pipeline completo (paso 1 a 7) de forma independiente -- su propia
# expansión de query, su propio retrieval, su propio answerability --
# y el resultado se sintetiza preservando el orden original. Una
# subpregunta sin evidencia NUNCA bloquea a las demás ni se inventa:
# queda marcada como no cubierta, explícitamente.
# ============================================================

MAX_EVIDENCE_PER_SUBQUESTION = 3
MIN_EVIDENCE_PER_SUBQUESTION = 1


@dataclass
class SubQuestionResult:
    question: str
    queries: list[str] = field(default_factory=list)
    evidence: list[Document] = field(default_factory=list)
    answerability_score: float = 0.0
    answerability_reason: str = ""
    answer: Optional[str] = None
    covered: bool = False
    confidence: float = 0.0


@dataclass
class BatchQuestionResult:
    original_query: str
    subquestions: list[SubQuestionResult] = field(default_factory=list)
    covered_count: int = 0
    uncovered_count: int = 0
    coverage_score: float = 0.0
    metricas: dict = field(default_factory=dict)


# ============================================================
# PASO 1 · QUERY ANALYSIS
# ============================================================


def _detectar_pagina(question: str) -> tuple[Optional[int], bool]:
    q = normalizar_texto(question)
    correcciones = {
        "pagaina": "pagina", "pagnina": "pagina", "pagnia": "pagina",
        "paginaa": "pagina", "pagna": "pagina", "pajina": "pagina",
    }
    q = " ".join(correcciones.get(p, p) for p in q.split())

    if re.search(r"\bultima pagina\b", q):
        return None, True

    ordinales = {
        "primera pagina": 0, "segunda pagina": 1, "tercera pagina": 2,
        "cuarta pagina": 3, "quinta pagina": 4, "sexta pagina": 5,
        "septima pagina": 6, "octava pagina": 7, "novena pagina": 8,
        "decima pagina": 9, "undecima pagina": 10, "duodecima pagina": 11,
    }
    for expresion, pagina in ordinales.items():
        if re.search(rf"\b{re.escape(expresion)}\b", q):
            return pagina, False

    numeros_escritos = {
        "uno": 0, "dos": 1, "tres": 2, "cuatro": 3, "cinco": 4,
        "seis": 5, "siete": 6, "ocho": 7, "nueve": 8, "diez": 9,
        "once": 10, "doce": 11, "trece": 12, "catorce": 13, "quince": 14,
        "dieciseis": 15, "diecisiete": 16, "dieciocho": 17, "diecinueve": 18,
        "veinte": 19,
    }
    for palabra, pagina in numeros_escritos.items():
        if re.search(rf"\bpagina\s+{palabra}\b", q):
            return pagina, False

    patrones = [
        r"\bpagina\s+(?:numero\s+)?(\d+)\b",
        r"\bpag\s+(\d+)\b",
        r"\bpag\.?\s*(\d+)\b",
    ]
    for patron in patrones:
        m = re.search(patron, q)
        if m:
            n = int(m.group(1))
            if n >= 1:
                return n - 1, False
    return None, False


def _detectar_resumen(question: str) -> bool:
    q = normalizar_texto(question)
    patrones = [
        r"\bresume\b", r"\bresumen\b", r"\bresumir\b", r"\bresumeme\b",
        r"\bhaz un resumen\b", r"\bhazme un resumen\b", r"\bsintetiza\b",
        r"\bsintesis del documento\b", r"\bde que trata\b", r"\bde que se trata\b",
        r"\bresumen ejecutivo\b",
    ]
    return any(re.search(p, q) for p in patrones)


def _detectar_comparacion(question: str) -> bool:
    q = normalizar_texto(question)
    patrones = [
        r"\bcompara\b", r"\bcomparar\b", r"\bcomparacion\b",
        r"\bcomparando\b", r"\bdiferencia(?:s)? entre\b",
        r"\bambos documentos\b", r"\blos dos documentos\b",
        r"\bentre los documentos\b", r"\bcontrasta\b", r"\bcontrastar\b",
        r"\brelacion entre\b", r"\brelaciona\b", r"\bsimilitudes entre\b",
        r"\bsimilaridades entre\b", r"\bque tienen en comun\b",
        r"\buno frente al otro\b", r"\bfrente al otro\b",
    ]
    return any(re.search(p, q) for p in patrones)


def _detectar_subpreguntas(question: str) -> list[str]:
    q = str(question or "").strip()
    if not q:
        return []

    partes = re.split(r"\?+\s*", q)
    partes = [p.strip(" ,;\n\t") for p in partes if p.strip(" ,;\n\t")]

    if len(partes) > 1:
        return partes

    patron = r",\s*(?=(?:¿|cu[aá]l|qu[ié]n|cu[aá]ndo|d[oó]nde|qu[eé]|c[oó]mo)\b)"
    partes = [p.strip(" ,;\n\t") for p in re.split(patron, q, flags=re.IGNORECASE) if p.strip(" ,;\n\t")]
    return partes if len(partes) > 1 else [q]


def _detectar_perfil(question: str, area: str, pagina: Optional[int], comparacion: bool) -> str:
    q = normalizar_texto(question)
    if comparacion:
        return "comparacion"
    if pagina is not None or "pagina" in q:
        return "pagina"

    terminos_area = _perfiles_de_area(area)
    # Primero perfiles específicos del área (más discriminantes), luego genéricos.
    orden = [p for p in terminos_area if p not in GENERIC_PROFILE_TERMS] + list(GENERIC_PROFILE_TERMS.keys())
    for perfil in orden:
        terminos = terminos_area.get(perfil, [])
        for termino in terminos:
            termino_norm = normalizar_texto(termino)
            if termino_norm and re.search(rf"\b{re.escape(termino_norm)}\b", q):
                return perfil

    if any(k in q for k in ("quien", "quienes", "nombre", "autor", "responsable")):
        return "persona"
    return "general"


def _detectar_tipo_consulta(question: str, comparacion: bool, fuente: Optional[str], pagina: Optional[int]) -> str:
    q = normalizar_texto(question)
    if comparacion:
        return "comparacion"
    if pagina is not None:
        return "pagina"
    if fuente:
        return "documento"
    patrones = [
        r"\bquien\b", r"\bquienes\b", r"\bcual\b", r"\bcuales\b",
        r"\bcuando\b", r"\bfecha\b", r"\bcuanto\b", r"\bcuantos\b",
        r"\bnumero de\b", r"\bnombre de\b", r"\bdonde\b", r"\bque monto\b",
        r"\bque cargo\b", r"\bque codigo\b", r"\bque resolucion\b",
        r"\bque oficio\b", r"\bque memorando\b", r"\bque contrato\b",
        r"\bque factura\b",
    ]
    return "factual" if any(re.search(p, q) for p in patrones) else "general"


# ============================================================
# FUENTES EXPLÍCITAS (busca en TODAS las áreas indexadas)
# ============================================================


def _listar_pdfs_disco() -> list[Path]:
    if not PDF_FOLDER.exists():
        return []
    return sorted(PDF_FOLDER.rglob("*.pdf"))


def _obtener_fuentes_explicitas(question: str, q_norm: str, q_tokens: set[str]) -> list[str]:
    pdfs = _listar_pdfs_disco()
    if not pdfs:
        return []

    encontrados = []

    for path in pdfs:
        source_id = _source_id_de_path(path)
        nombre_norm = normalizar_texto(path.name)
        stem_norm = normalizar_texto(path.stem)
        if nombre_norm and nombre_norm in q_norm:
            encontrados.append(source_id)
        elif len(stem_norm) >= MIN_LEN_STEM_MATCH and stem_norm in q_norm:
            encontrados.append(source_id)

    candidatos = []
    for path in pdfs:
        source_id = _source_id_de_path(path)
        if source_id in encontrados:
            continue
        nombre_tokens = tokens_significativos(path.stem)
        inter = q_tokens & nombre_tokens
        if not inter:
            continue
        proporcion = len(inter) / max(1, len(nombre_tokens))
        largos = sum(1 for t in inter if len(t) >= LONGITUD_TOKEN_BONO)
        score = len(inter) + proporcion + 0.5 * largos
        if len(inter) >= MIN_TOKEN_MATCHES and proporcion >= MIN_TOKEN_PROPORTION:
            candidatos.append((score, proporcion, len(inter), source_id))

    candidatos.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    for _, _, _, source_id in candidatos:
        if source_id not in encontrados:
            encontrados.append(source_id)

    if not encontrados and "expediente" in q_norm:
        posibles = [
            _source_id_de_path(p) for p in pdfs
            if "expediente" in normalizar_texto(p.stem)
        ]
        if len(posibles) == 1:
            encontrados.append(posibles[0])

    return encontrados


def _area_de_fuentes(fuentes: list[str]) -> Optional[str]:
    """Si todas las fuentes explícitas detectadas comparten área, la devuelve.

    A diferencia de la ruta de ingesta (donde un manifest corrupto debe
    DETENER el proceso para no perder metadata), esta función corre en
    cada pregunta del usuario: un manifest corrupto aquí no debe tumbar
    las consultas -- se degrada a "área no determinada" y se sigue
    dependiendo de la metadata que ya vive en Chroma."""
    if not fuentes:
        return None
    try:
        manifest = cargar_manifest()
    except RuntimeError as e:
        log(f"ADVERTENCIA: manifest ilegible durante una consulta ({e}); se continúa sin filtrar por área heredada.")
        return None
    areas = {manifest.get(f, {}).get("area", DEFAULT_AREA) for f in fuentes}
    if len(areas) == 1:
        return next(iter(areas))
    return None


# ============================================================
# PASO 2 · QUERY EXPANSION
# ============================================================


def construir_query_variants(ctx: QueryContext) -> list[str]:
    variantes = []

    def add(texto: str):
        texto = str(texto or "").strip()
        if texto and texto not in variantes:
            variantes.append(texto)

    add(ctx.question)

    for sub in ctx.subpreguntas:
        add(sub)

    for sub in ctx.subpreguntas[:3]:
        limpio = re.sub(r"[¿?!]", "", sub).strip()
        add(limpio)

    terminos_area = _perfiles_de_area(ctx.area)
    terms = terminos_area.get(ctx.perfil, [])
    if terms:
        add(" ".join(terms[:8]))

    # Expansiones genéricas por perfil (aplican a cualquier área).
    expansiones_genericas = {
        "fecha": ["fecha emision registro creacion documento"],
        "monto": ["monto importe suma cantidad valor costo presupuesto"],
        "lugar": ["lugar ubicacion direccion sede distrito ciudad"],
        "tabla": ["tabla cuadro datos filas columnas total"],
        "persona": ["nombre responsable encargado titular firmante"],
        "identificador": ["codigo numero identificador referencia"],
        "resultado": ["resultado conclusion decision acuerdo determino"],
    }
    for texto in expansiones_genericas.get(ctx.perfil, []):
        add(texto)

    # Expansiones específicas por área + perfil (vocabulario especializado).
    if ctx.perfil in terminos_area and ctx.perfil not in GENERIC_PROFILE_TERMS:
        add(f"{ctx.perfil.replace('_', ' ')} " + " ".join(terms[:6]))

    if ctx.perfil == "pagina" and ctx.pagina_explicita is not None:
        add(f"página {ctx.pagina_explicita + 1} contenido")
        add(f"página {ctx.pagina_explicita + 1} texto información")

    return variantes[:QUERY_EXPANSION_MAX]


# ============================================================
# ÚLTIMA PÁGINA
# ============================================================


def obtener_numero_paginas(fuente: str) -> Optional[int]:
    path = PDF_FOLDER / fuente
    if not path.exists():
        return None
    try:
        import fitz
        pdf = fitz.open(str(path))
        try:
            return pdf.page_count
        finally:
            pdf.close()
    except Exception:
        try:
            return len(cargar_pdf_texto(path))
        except Exception:
            return None


def resolver_ultima_pagina(ctx: QueryContext):
    if not ctx.ultima_pagina_solicitada or not ctx.fuente_explicita:
        return
    paginas = obtener_numero_paginas(ctx.fuente_explicita)
    if paginas:
        ctx.pagina_explicita = paginas - 1
        log(f"  Última página resuelta: {paginas}")


def detectar_intencion(question: str) -> QueryContext:
    question = str(question or "").strip()
    if not question:
        return QueryContext(question="")

    q_norm = normalizar_texto(question)
    q_tokens = tokens_significativos(question)
    fuentes = _obtener_fuentes_explicitas(question, q_norm, q_tokens)
    fuente = fuentes[0] if fuentes else None
    pagina, ultima = _detectar_pagina(question)
    comparacion = _detectar_comparacion(question)
    es_resumen = _detectar_resumen(question) and not comparacion

    # Área: 1) mencionada explícitamente en la pregunta,
    #       2) heredada de la(s) fuente(s) explícita(s),
    #       3) DEFAULT_AREA (la búsqueda igual cubre todas las áreas).
    area_explicita = _detectar_area_en_pregunta(q_norm)
    area = area_explicita or _area_de_fuentes(fuentes) or DEFAULT_AREA

    tipo = "resumen" if es_resumen else _detectar_tipo_consulta(question, comparacion, fuente, pagina)
    subpreguntas = _detectar_subpreguntas(question)
    perfil = "resumen" if es_resumen else _detectar_perfil(question, area, pagina, comparacion)

    terminos_area = _perfiles_de_area(area)
    focus_terms = set(terminos_area.get(perfil, []))
    focus_terms.add(perfil.replace("_", " "))

    sensitive = perfil in _perfiles_sensibles_de_area(area) or perfil in {"fecha", "monto"}

    ctx = QueryContext(
        question=question,
        q_tokens=q_tokens,
        area=area,
        area_explicita=area_explicita,
        fuente_explicita=fuente,
        fuentes_explicitas=fuentes,
        pagina_explicita=pagina,
        ultima_pagina_solicitada=ultima,
        es_comparacion=comparacion,
        tipo_consulta=tipo,
        subpreguntas=subpreguntas,
        perfil=perfil,
        focus_terms=focus_terms,
        sensitive=sensitive,
        # Item 10 (auditoría): política explícita de costo de LLM por
        # tipo de consulta -- NUNCA se recorta para sensibles/factuales:
        #   general no sensible -> generación + heurística (sin
        #     completeness LLM: una pregunta abierta rara vez tiene una
        #     única "respuesta completa" verificable, y es la categoría
        #     de mayor volumen -> aquí es donde más rinde ahorrar).
        #   factual / comparación / sensible / resumen -> completeness
        #     LLM sigue activo (resumen la deshabilita aparte, por su
        #     propio flujo map-reduce que no pasa por este validador).
        requires_completeness=(
            not es_resumen
            and (sensitive or tipo != "general")
        ),
        es_resumen=es_resumen,
    )

    resolver_ultima_pagina(ctx)
    ctx.query_variants = construir_query_variants(ctx)
    return ctx


# ============================================================
# LEXICAL INDEX
# ============================================================


def _construir_lexical_index():
    global LEXICAL_CACHE
    # Item 18 (auditoría): doble-check locking. Sin esto, dos requests
    # concurrentes con cache frío podrían disparar dos reconstrucciones
    # completas simultáneas (cada una trae TODO el corpus de Chroma) --
    # no corrompe datos, pero duplica una operación costosa.
    if LEXICAL_CACHE is not None:
        return LEXICAL_CACHE

    with LEXICAL_CACHE_LOCK:
        if LEXICAL_CACHE is not None:
            return LEXICAL_CACHE

        try:
            data = vector_db.get(include=["documents", "metadatas"])
        except Exception as e:
            log(f"ADVERTENCIA construyendo índice lexical: {e}")
            LEXICAL_CACHE = ([], defaultdict(set))
            return LEXICAL_CACHE

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        docs = []
        postings = defaultdict(set)

        for i, (content, metadata) in enumerate(zip(documents, metadatas)):
            metadata = dict(metadata or {})
            doc = Document(page_content=content or "", metadata=metadata)
            docs.append(doc)
            for token in tokens_significativos(content or ""):
                bucket = postings[token]
                if len(bucket) < LEXICAL_INDEX_MAX_POSTING_DOCS:
                    bucket.add(i)

        # Item 5 (auditoría): métricas de tamaño del cache en memoria,
        # con umbral de advertencia. No migramos automáticamente a un
        # índice persistente (SQLite FTS5 sería la opción natural si
        # esto se dispara con frecuencia) -- solo lo hacemos visible
        # para decidir con datos reales, no por intuición.
        n_chunks = len(docs)
        n_tokens_unicos = len(postings)
        log(
            f"Índice lexical construido: {n_chunks} chunks, "
            f"{n_tokens_unicos} tokens únicos en memoria."
        )
        if n_chunks > LEXICAL_CACHE_WARN_CHUNKS:
            log(
                f"ADVERTENCIA: el corpus supera LEXICAL_CACHE_WARN_CHUNKS="
                f"{LEXICAL_CACHE_WARN_CHUNKS} chunks ({n_chunks}). El índice "
                f"lexical en memoria puede empezar a pesar en RAM -- evaluar "
                f"migrar a un índice persistente (ej. SQLite FTS5) si esto "
                f"se sostiene o crece."
            )

        LEXICAL_CACHE = (docs, postings)
        return LEXICAL_CACHE


def _filtrar_doc_por_ctx(doc: Document, ctx: QueryContext) -> bool:
    if ctx.fuentes_explicitas:
        if doc.metadata.get("source") not in set(ctx.fuentes_explicitas):
            return False
    elif ctx.fuente_explicita:
        if doc.metadata.get("source") != ctx.fuente_explicita:
            return False

    if ctx.area_explicita:
        if doc.metadata.get("area") != ctx.area_explicita:
            return False

    if ctx.pagina_explicita is not None:
        if doc.metadata.get("page") != ctx.pagina_explicita:
            return False

    return True


def _score_lexical(texto: str, query_tokens: set[str], focus_terms: set[str]) -> float:
    text_tokens = tokens_significativos(texto)
    if not text_tokens:
        return 0.0

    inter = query_tokens & text_tokens
    base = len(inter) / max(1, len(query_tokens))

    bonus = sum(
        BONO_LEXICAL_CHUNK
        for token in inter
        if len(token) >= LONGITUD_TOKEN_BONO
    )

    focus_norm = {
        t for t in focus_terms
        if len(t) >= 3
    }
    focus_hits = sum(
        1 for term in focus_norm
        if term in normalizar_texto(texto)
    )
    focus_bonus = min(0.50, 0.08 * focus_hits)

    return base + bonus + focus_bonus


def recuperar_lexical(variant: str, ctx: QueryContext, k: int) -> list[tuple[Document, float]]:
    docs, postings = _construir_lexical_index()
    if not docs:
        return []

    query_tokens = tokens_significativos(variant)
    if not query_tokens:
        return []

    indices = set()
    for token in query_tokens:
        indices.update(postings.get(token, set()))

    for term in ctx.focus_terms:
        norm = normalizar_texto(term)
        if " " in norm:
            for token in tokens_significativos(norm):
                indices.update(postings.get(token, set()))
        else:
            indices.update(postings.get(norm, set()))

    resultados = []
    for idx in indices:
        doc = docs[idx]
        if not _filtrar_doc_por_ctx(doc, ctx):
            continue
        score = _score_lexical(doc.page_content, query_tokens, ctx.focus_terms)
        if score > 0:
            resultados.append((doc, score))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados[:k]


# ============================================================
# PASO 3A · SEMANTIC RETRIEVAL
# ============================================================


def _semantic_score_from_distance(distance: float) -> float:
    try:
        d = max(0.0, float(distance))
    except (ValueError, TypeError):
        return 0.0
    return 1.0 / (1.0 + d)


def _construir_filtro_chroma(ctx: QueryContext) -> Optional[dict]:
    """Combina filtros de source/area/page en un filtro Chroma válido
    (Chroma exige un solo operador de nivel superior con 2+ condiciones)."""
    condiciones = []

    if len(ctx.fuentes_explicitas) == 1:
        condiciones.append({"source": ctx.fuentes_explicitas[0]})
    elif ctx.fuente_explicita:
        condiciones.append({"source": ctx.fuente_explicita})
    elif len(ctx.fuentes_explicitas) > 1:
        condiciones.append({"source": {"$in": ctx.fuentes_explicitas}})

    if ctx.area_explicita and not ctx.fuentes_explicitas and not ctx.fuente_explicita:
        condiciones.append({"area": ctx.area_explicita})

    if ctx.pagina_explicita is not None and not ctx.fuentes_explicitas and not ctx.fuente_explicita:
        condiciones.append({"page": ctx.pagina_explicita})

    if not condiciones:
        return None
    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}


def recuperar_semantico(variant: str, ctx: QueryContext, k: int, rank_offset: int = 0) -> list[Document]:
    try:
        filtros = _construir_filtro_chroma(ctx)
        if filtros:
            resultados = vector_db.similarity_search_with_score(
                variant,
                k=k,
                filter=filtros,
            )
        else:
            resultados = vector_db.similarity_search_with_score(
                variant,
                k=k,
            )
    except Exception as e:
        log(f"ADVERTENCIA semantic retrieval '{variant}': {e}")
        return []

    docs = []
    for rank, (doc, distance) in enumerate(resultados, start=1 + rank_offset):
        doc.metadata["_distance"] = float(distance)
        doc.metadata["_semantic_score"] = _semantic_score_from_distance(distance)
        doc.metadata["_semantic_rank"] = 1.0 / max(1, rank)
        docs.append(doc)
    return docs


# ============================================================
# PASO 4 · HYBRID RANKING (fusión semántica + lexical)
# ============================================================


def _doc_key(doc: Document) -> tuple:
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    chunk = doc.metadata.get("chunk_index")
    if chunk is not None:
        return source, page, chunk
    return source, page, hashlib.md5(
        (doc.page_content or "").encode("utf-8", errors="ignore")
    ).hexdigest()


def recuperar_candidatos(ctx: QueryContext) -> list[Document]:
    variants = ctx.query_variants or [ctx.question]
    aggregate = {}

    for variant in variants:
        semantic_docs = recuperar_semantico(variant, ctx, SEMANTIC_RESULTS_PER_QUERY)
        for doc in semantic_docs:
            key = _doc_key(doc)
            lex = _score_lexical(
                doc.page_content,
                tokens_significativos(variant),
                ctx.focus_terms,
            )
            sem = float(doc.metadata.get("_semantic_score", 0.0) or 0.0)
            rank_score = float(doc.metadata.get("_semantic_rank", 0.0) or 0.0)
            hybrid = SEMANTIC_WEIGHT * sem + LEXICAL_WEIGHT * lex + RANK_WEIGHT * rank_score

            current = aggregate.get(key)
            if current is None or hybrid > current["hybrid"]:
                aggregate[key] = {
                    "doc": doc,
                    "hybrid": hybrid,
                    "semantic": sem,
                    "lexical": lex,
                    "rank": rank_score,
                }

        lexical_docs = recuperar_lexical(variant, ctx, LEXICAL_RESULTS_PER_QUERY)
        for doc, lex in lexical_docs:
            key = _doc_key(doc)
            sem = float(doc.metadata.get("_semantic_score", 0.0) or 0.0)
            rank_score = float(doc.metadata.get("_semantic_rank", 0.0) or 0.0)
            hybrid = SEMANTIC_WEIGHT * sem + LEXICAL_WEIGHT * lex + RANK_WEIGHT * rank_score
            current = aggregate.get(key)
            if current is None or hybrid > current["hybrid"]:
                aggregate[key] = {
                    "doc": doc,
                    "hybrid": hybrid,
                    "semantic": sem,
                    "lexical": lex,
                    "rank": rank_score,
                }

    salida = []
    for item in aggregate.values():
        doc = item["doc"]
        doc.metadata["_semantic_score"] = item["semantic"]
        doc.metadata["_lexical_score"] = item["lexical"]
        doc.metadata["_rank_score"] = item["rank"]
        doc.metadata["_hybrid_score"] = item["hybrid"]
        salida.append(doc)

    salida = [d for d in salida if _filtrar_doc_por_ctx(d, ctx)]
    salida.sort(
        key=lambda d: float(d.metadata.get("_hybrid_score", 0.0) or 0.0),
        reverse=True,
    )
    return salida[:RETRIEVER_K]


# ============================================================
# PASO 4.5 · RERANKING HEURÍSTICO (sin modelo cross-encoder)
# ============================================================
#
# Decisión de diseño: NO se agrega un reranker basado en modelo
# (ej. bge-reranker vía sentence-transformers) porque:
#   - añade una dependencia pesada + un modelo adicional a cargar,
#   - en hardware local ya limitado por Ollama, cuesta latencia real,
#   - el problema concreto reportado (fecha/expediente/monto exacto
#     no siempre en el chunk mejor rankeado por similitud) se
#     resuelve con una señal determinista y barata: ¿el chunk
#     contiene la ENTIDAD EXACTA (fecha, número, nombre propio,
#     identificador) que aparece en la pregunta o en el perfil?
#
# Si en el futuro se quiere un reranker real:
#   pip install sentence-transformers
#   modelo sugerido (100% local, corre en CPU): BAAI/bge-reranker-base
# ============================================================

_MESES = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
    "setiembre|octubre|noviembre|diciembre"
)


def _extraer_entidades_pregunta(ctx: QueryContext) -> dict:
    """Entidades exactas mencionadas EN LA PREGUNTA (no en el contexto).
    Sirven de ancla para el rerank: si la pregunta ya trae un número,
    fecha o nombre propio, el chunk ganador debería contenerlo."""
    q = ctx.question
    q_norm = normalizar_texto(q)
    return {
        "fechas": set(re.findall(
            rf"\b\d{{1,2}}\s+de\s+(?:{_MESES})\s+del?\s+\d{{4}}\b"
            r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            q_norm,
        )),
        "numeros": extraer_numeros(q),
        "nombres": set(_extraer_secuencias_nombre(q)) if any(c.isupper() for c in q) else set(),
    }


def _rerank_bonus(doc: Document, entidades_pregunta: dict, ctx: QueryContext) -> float:
    """Bonifica un chunk si contiene literalmente una entidad que la
    pregunta ya trae (caso: pregunta con número de expediente parcial,
    con una fecha, o con un nombre propio) y, para perfiles temporales
    o numéricos, bonifica cualquier chunk que traiga UNA fecha/monto
    aunque la pregunta no la traiga (caso #33: "fecha de la sentencia")."""
    texto_norm = normalizar_texto(doc.page_content)
    bonus = 0.0

    for fecha in entidades_pregunta["fechas"]:
        if fecha and fecha in texto_norm:
            bonus += 0.25

    for numero in entidades_pregunta["numeros"]:
        if len(numero) >= 4 and numero in doc.page_content:
            bonus += 0.15

    for nombre in entidades_pregunta["nombres"]:
        if normalizar_texto(nombre) in texto_norm:
            bonus += 0.20

    # Perfiles temporales/numéricos: el chunk ganador DEBE traer una
    # fecha/monto propios, aunque la pregunta no la mencione todavía.
    if ctx.perfil in {"fecha", "fecha_hechos", "fecha_sentencia"} and _extraer_fechas(doc.page_content):
        bonus += 0.20
    if ctx.perfil == "monto" and _extraer_montos(doc.page_content):
        bonus += 0.20
    if ctx.perfil in {"expediente", "resolucion", "oficio", "memorando", "factura", "contrato"} and _extraer_ids(doc.page_content):
        bonus += 0.20

    return min(0.60, bonus)


def rerank_candidatos(ctx: QueryContext, candidatos: list[Document]) -> list[Document]:
    if not candidatos:
        return candidatos

    entidades_pregunta = _extraer_entidades_pregunta(ctx)
    for doc in candidatos:
        bonus = _rerank_bonus(doc, entidades_pregunta, ctx)
        hybrid = float(doc.metadata.get("_hybrid_score", 0.0) or 0.0)
        doc.metadata["_rerank_bonus"] = bonus
        doc.metadata["_hybrid_score"] = hybrid + bonus  # se propaga a ranking/evidencia/gate

    candidatos.sort(
        key=lambda d: float(d.metadata.get("_hybrid_score", 0.0) or 0.0),
        reverse=True,
    )
    return candidatos


# ============================================================
# PASO 7 · ANSWERABILITY / GROUNDING GATE
# ============================================================


def evaluar_answerability(ctx: QueryContext, candidatos: list[Document]) -> tuple[bool, float, str]:
    if not candidatos:
        return False, 0.0, "sin candidatos"

    top = candidatos[0]
    best_hybrid = float(top.metadata.get("_hybrid_score", 0.0) or 0.0)
    best_sem = float(top.metadata.get("_semantic_score", 0.0) or 0.0)
    best_lex = float(top.metadata.get("_lexical_score", 0.0) or 0.0)

    restricted = bool(
        ctx.fuente_explicita or ctx.fuentes_explicitas
        or ctx.pagina_explicita is not None or ctx.area_explicita
    )

    if restricted and ctx.pagina_explicita is not None:
        exact_page = any(
            d.metadata.get("page") == ctx.pagina_explicita
            for d in candidatos
        )
        if exact_page and best_hybrid >= 0.32:
            return True, min(0.99, 0.65 + 0.35 * best_hybrid), "restricción explícita con evidencia"

    focus_hit = 1.0 if any(
        term and term in normalizar_texto(top.page_content)
        for term in ctx.focus_terms
        if len(normalizar_texto(term)) >= 4
    ) else 0.0

    if ctx.tipo_consulta == "factual":
        threshold = ANSWERABILITY_MIN_SENSITIVE if ctx.sensitive else ANSWERABILITY_MIN_FACTUAL
    else:
        threshold = ANSWERABILITY_MIN_GENERAL

    if best_hybrid < threshold:
        return False, best_hybrid, "score híbrido insuficiente"

    if (
        not restricted
        and ctx.tipo_consulta == "factual"
        and best_lex < ANSWERABILITY_MIN_LEXICAL_WITHOUT_RESTRICTION
        and best_sem < ANSWERABILITY_SEMANTIC_ONLY_MIN
    ):
        return False, best_hybrid, "señal factual insuficiente sin ancla lexical"

    if ctx.sensitive and best_lex == 0 and best_sem < 0.62 and not restricted:
        return False, best_hybrid, "consulta sensible sin ancla verificable"

    razon = "señal híbrida suficiente"
    if focus_hit:
        razon = "señal híbrida + ancla de intención"
    elif best_lex > 0:
        razon = "señal semántica + lexical"

    return True, min(0.99, best_hybrid), razon


# ============================================================
# PASO 5 · DOCUMENT ROUTER (ranking + aislamiento)
# ============================================================


def _puntuar_documento(ctx: QueryContext, docs: list[Document]) -> float:
    if not docs:
        return 0.0

    scores = []
    for doc in docs:
        hybrid = float(doc.metadata.get("_hybrid_score", 0.0) or 0.0)
        lexical = float(doc.metadata.get("_lexical_score", 0.0) or 0.0)
        semantic = float(doc.metadata.get("_semantic_score", 0.0) or 0.0)
        focus = _score_lexical(doc.page_content, ctx.q_tokens, ctx.focus_terms)
        score = 0.55 * hybrid + 0.20 * lexical + 0.15 * semantic + 0.10 * min(1.0, focus)
        scores.append(score)

    scores.sort(reverse=True)
    result = scores[0]
    if len(scores) >= 2:
        result += 0.20 * scores[1]
    if len(scores) >= 3:
        result += 0.10 * scores[2]
    return result


def identificar_documento_relevante(ctx: QueryContext, candidatos: list[Document]):
    if not candidatos:
        return []

    por_fuente = defaultdict(list)
    for d in candidatos:
        source = d.metadata.get("source")
        if source:
            por_fuente[source].append(d)

    ranking = [
        (_puntuar_documento(ctx, docs), source, docs)
        for source, docs in por_fuente.items()
    ]
    ranking.sort(key=lambda x: x[0], reverse=True)

    log("\n--- Paso 3: ranking de documentos ---")
    for score, source, docs in ranking:
        area_doc = docs[0].metadata.get("area", DEFAULT_AREA) if docs else DEFAULT_AREA
        log(f"  {score:.4f} -> [{area_doc}] {source} ({len(docs)} chunks)")
    return ranking


def aislar_documento(ctx: QueryContext, ranking_documentos) -> list[Document]:
    if not ranking_documentos:
        return []

    if ctx.es_comparacion:
        if ctx.fuentes_explicitas:
            permitidos = set(ctx.fuentes_explicitas)
            salida = []
            for _, source, docs in ranking_documentos:
                if source in permitidos:
                    salida.extend(docs)
            return salida

        salida = []
        for _, _, docs in ranking_documentos[:MAX_DOCUMENTS_COMPARISON]:
            salida.extend(docs)
        return salida

    if ctx.fuente_explicita:
        for _, source, docs in ranking_documentos:
            if source == ctx.fuente_explicita:
                return docs
        return []

    return ranking_documentos[0][2]


# ============================================================
# PASO 6 · EVIDENCE SELECTION (chunks vecinos, diversidad de página)
# ============================================================


def obtener_chunks_de_pagina(source: str, page: int) -> list[Document]:
    try:
        data = vector_db.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    docs = []
    for content, metadata in zip(
        data.get("documents") or [],
        data.get("metadatas") or [],
    ):
        metadata = dict(metadata or {})
        if metadata.get("page") == page:
            docs.append(Document(page_content=content or "", metadata=metadata))

    docs.sort(key=lambda d: int(d.metadata.get("chunk_index", 0) or 0))
    return docs


def expandir_evidencia_vecina(ctx: QueryContext, docs: list[Document]) -> list[Document]:
    if not docs:
        return []

    agrupados = {}
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        chunk_index = doc.metadata.get("chunk_index")
        if source is None or page is None:
            continue
        key = (source, page)
        agrupados.setdefault(key, set()).add(chunk_index)

    salida = []
    vistos = set()

    for (source, page), hit_chunks in agrupados.items():
        page_docs = obtener_chunks_de_pagina(source, page)
        if not page_docs:
            continue

        indices_hit = {
            int(i)
            for i in hit_chunks
            if i is not None and str(i).isdigit()
        }

        seleccionados = []
        for page_doc in page_docs:
            idx = page_doc.metadata.get("chunk_index")
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                idx_int = None

            incluir = False
            if idx_int is not None and indices_hit:
                incluir = any(
                    abs(idx_int - hit) <= NEIGHBOR_RADIUS
                    for hit in indices_hit
                )
            elif not indices_hit:
                incluir = True

            if incluir:
                seleccionados.append(page_doc)

        if len(seleccionados) > MAX_NEIGHBOR_CHUNKS_PER_HIT:
            seleccionados.sort(
                key=lambda d: (
                    0 if d.metadata.get("chunk_index") in hit_chunks else 1,
                    d.metadata.get("chunk_index", 0),
                )
            )
            seleccionados = seleccionados[:MAX_NEIGHBOR_CHUNKS_PER_HIT]

        for doc in seleccionados:
            key = _doc_key(doc)
            if key in vistos:
                continue
            vistos.add(key)
            base = max(
                float(d.metadata.get("_hybrid_score", 0.0) or 0.0)
                for d in docs
                if d.metadata.get("source") == source
                and d.metadata.get("page") == page
            ) if any(
                d.metadata.get("source") == source and d.metadata.get("page") == page
                for d in docs
            ) else 0.0
            doc.metadata["_expanded"] = True
            doc.metadata["_page_base_score"] = base
            salida.append(doc)

    return salida


def _bonus_tipo_consulta(ctx: QueryContext, doc: Document) -> float:
    texto = normalizar_texto(doc.page_content)
    terminos_area = _perfiles_de_area(ctx.area)
    terms = terminos_area.get(ctx.perfil, [])
    if not terms:
        return 0.0
    hits = sum(1 for t in terms[:8] if normalizar_texto(t) in texto)
    return min(0.20, hits * 0.025)


def _puntuacion_lexica_chunk(ctx: QueryContext, doc: Document) -> float:
    return _score_lexical(
        doc.page_content,
        ctx.q_tokens,
        ctx.focus_terms,
    )


def _puntuacion_semantica_chunk(doc: Document) -> float:
    return float(doc.metadata.get("_semantic_score", 0.0) or 0.0)


def seleccionar_evidencia(ctx: QueryContext, docs_aislados: list[Document]):
    if not docs_aislados:
        return [], []

    docs = [d for d in docs_aislados if _filtrar_doc_por_ctx(d, ctx)]
    if not docs:
        return [], []

    expandidos = expandir_evidencia_vecina(ctx, docs)
    combinados = docs + [d for d in expandidos if _doc_key(d) not in {_doc_key(x) for x in docs}]

    candidatos = []
    for doc in combinados:
        lexical = _puntuacion_lexica_chunk(ctx, doc)
        semantic = _puntuacion_semantica_chunk(doc)
        hybrid = float(doc.metadata.get("_hybrid_score", 0.0) or 0.0)
        bonus = _bonus_tipo_consulta(ctx, doc)
        expanded_bonus = 0.04 if doc.metadata.get("_expanded") else 0.0
        score = 0.45 * hybrid + 0.25 * semantic + 0.20 * lexical + bonus + expanded_bonus
        doc.metadata["_evidence_score"] = score
        candidatos.append((score, doc))

    candidatos.sort(key=lambda x: x[0], reverse=True)

    grupos = defaultdict(list)
    for score, doc in candidatos:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page"))
        grupos[key].append((score, doc))

    page_rank = []
    for key, items in grupos.items():
        items.sort(key=lambda x: x[0], reverse=True)
        page_score = items[0][0]
        if len(items) > 1:
            page_score += 0.15 * items[1][0]
        if len(items) > 2:
            page_score += 0.05 * items[2][0]
        page_rank.append((page_score, key, items))

    page_rank.sort(key=lambda x: x[0], reverse=True)

    selected_pages = page_rank[:MAX_EVIDENCE_PAGES]
    evidencia = []
    scores_finales = []
    vistos = set()

    for _, _, items in selected_pages:
        items.sort(key=lambda x: x[1].metadata.get("chunk_index", 0))
        for score, doc in items:
            key = _doc_key(doc)
            if key in vistos:
                continue
            vistos.add(key)
            evidencia.append(doc)
            scores_finales.append((score, doc))
            if len(evidencia) >= FINAL_CONTEXT_CHUNKS:
                return evidencia, scores_finales

    return evidencia, scores_finales


# ============================================================
# PIPELINE COMPLETO (pasos 1 a 7)
# ============================================================


def _un_pase_pipeline(ctx: QueryContext) -> tuple[list[Document], list, list[Document], list[Document], list, bool, float, str]:
    candidatos = recuperar_candidatos(ctx)
    candidatos = rerank_candidatos(ctx, candidatos)  # PASO 4.5
    ranking = identificar_documento_relevante(ctx, candidatos)
    aislados = aislar_documento(ctx, ranking)
    evidencia, scores = seleccionar_evidencia(ctx, aislados)
    answerable, answer_score, answer_reason = evaluar_answerability(ctx, candidatos)
    return candidatos, ranking, aislados, evidencia, scores, answerable, answer_score, answer_reason


def puede_relajar_answerability(ctx: QueryContext, candidatos: list[Document]) -> tuple[bool, str]:
    """P1 (auditoría): distingue explícitamente cuándo es seguro
    relajar el umbral de answerability de cuándo NO lo es.

    Relajable (A): evidencia débil pero potencialmente relevante --
    el usuario no restringió nada, o restringió algo que SÍ tiene
    contenido recuperado, solo que el score híbrido general es bajo.

    NO relajable:
      B) documento incorrecto -- el usuario pidió un archivo/fuente
         explícita y no hay NINGÚN candidato de esa fuente.
      C) página incorrecta -- pidió una página explícita y no hay
         NINGÚN chunk de esa página en los candidatos.
      D) área incompatible -- pidió un área explícita y no hay
         candidatos de esa área.
    En estos casos, relajar el umbral fabricaría una respuesta sobre
    una restricción que el sistema sabe que no puede satisfacer --
    exactamente el comportamiento que NO queremos en un RAG
    institucional."""
    if not candidatos:
        return False, "sin candidatos: nada que relajar"

    if ctx.pagina_explicita is not None:
        if not any(d.metadata.get("page") == ctx.pagina_explicita for d in candidatos):
            return False, "página explícita sin ningún chunk recuperado (C)"

    fuentes_pedidas = set(ctx.fuentes_explicitas) if ctx.fuentes_explicitas else (
        {ctx.fuente_explicita} if ctx.fuente_explicita else set()
    )
    if fuentes_pedidas:
        if not any(d.metadata.get("source") in fuentes_pedidas for d in candidatos):
            return False, "documento explícito sin ningún chunk recuperado (B)"

    if ctx.area_explicita:
        if not any(d.metadata.get("area") == ctx.area_explicita for d in candidatos):
            return False, "área explícita sin ningún chunk recuperado (D)"

    return True, "evidencia débil pero dentro del alcance solicitado (A)"


def ejecutar_pipeline_batch(question: str) -> Optional[BatchQuestionResult]:
    """Items 3-9, 26 (auditoría hardening): si la pregunta trae 2+
    subpreguntas reales, cada una corre su PROPIO pipeline de
    retrieval (pasos 1-7) de forma independiente -- nunca comparten
    ranking ni el mismo cupo de FINAL_CONTEXT_CHUNKS. Devuelve None
    si no hay batch real (pregunta simple), señal para que el
    llamador use el camino de pregunta única sin cambios."""
    ctx_global = detectar_intencion(question)
    if len(ctx_global.subpreguntas) < 2:
        return None

    log(f"\n--- Batch detectado: {len(ctx_global.subpreguntas)} subpreguntas ---")

    resultados: list[SubQuestionResult] = []
    for sub in ctx_global.subpreguntas:
        ctx_sub = detectar_intencion(sub)

        # Item 5: una subpregunta sin restricción propia hereda la
        # restricción del enunciado completo (ej. "según el expediente
        # X: ¿quién es el acusado? ¿cuál es la fecha?" -- ambas deben
        # quedar acotadas al expediente X, aunque solo se mencione una vez).
        if not ctx_sub.fuente_explicita and not ctx_sub.fuentes_explicitas:
            ctx_sub.fuente_explicita = ctx_global.fuente_explicita
            ctx_sub.fuentes_explicitas = ctx_global.fuentes_explicitas
        if ctx_sub.area_explicita is None and ctx_global.area_explicita:
            ctx_sub.area_explicita = ctx_global.area_explicita
            ctx_sub.area = ctx_global.area
        if ctx_sub.pagina_explicita is None and ctx_global.pagina_explicita is not None:
            ctx_sub.pagina_explicita = ctx_global.pagina_explicita

        candidatos = recuperar_candidatos(ctx_sub)
        candidatos = rerank_candidatos(ctx_sub, candidatos)
        ranking = identificar_documento_relevante(ctx_sub, candidatos)
        aislados = aislar_documento(ctx_sub, ranking)
        evidencia, _scores = seleccionar_evidencia(ctx_sub, aislados)
        # Item 5: tope por subpregunta, para que una subpregunta no
        # consuma toda la capacidad de evidencia y deje a las demás sin contexto.
        evidencia = evidencia[:MAX_EVIDENCE_PER_SUBQUESTION]

        answerable, score, reason = evaluar_answerability(ctx_sub, candidatos)
        cubierta = answerable and len(evidencia) >= MIN_EVIDENCE_PER_SUBQUESTION

        resultados.append(SubQuestionResult(
            question=sub,
            queries=ctx_sub.query_variants,
            evidence=evidencia if cubierta else [],
            answerability_score=score,
            answerability_reason=reason,
            answer=None,
            covered=cubierta,
            confidence=score if cubierta else 0.0,
        ))
        log(f"  [{'cubierta' if cubierta else 'SIN evidencia'}] {sub!r} -> score {score:.4f} ({reason})")

    covered_count = sum(1 for r in resultados if r.covered)
    total = len(resultados)
    coverage_score = round(covered_count / total, 4) if total else 0.0

    return BatchQuestionResult(
        original_query=question,
        subquestions=resultados,
        covered_count=covered_count,
        uncovered_count=total - covered_count,
        coverage_score=coverage_score,
        metricas={
            "query_type": "batch",
            "subquestions": total,
            "covered_subquestions": covered_count,
            "uncovered_subquestions": total - covered_count,
            "coverage_score": coverage_score,
        },
    )


SUBANSWER_PROMPT_TEMPLATE = """
Eres el asistente documental local de la institución. Responde
ÚNICAMENTE esta subpregunta, usando EXCLUSIVAMENTE el contexto dado.

No uses conocimiento externo. No inventes datos. Si el contexto no
alcanza para responder con certeza, dilo explícitamente en vez de
adivinar. Sé directo: da el dato pedido primero, sin preámbulo.

CONTEXTO:
{context}

SUBPREGUNTA:
{question}

RESPUESTA (solo a esta subpregunta):
"""

NO_EVIDENCIA_SUBPREGUNTA = (
    "No encontré evidencia suficiente en los documentos recuperados "
    "para responder esta parte."
)


def generar_respuesta_batch(llm, batch: BatchQuestionResult) -> tuple[str, dict]:
    """Items 7-9, 27 (auditoría hardening): genera una subrespuesta por
    subpregunta CUBIERTA (grounded, prompt acotado a su propia
    evidencia), y NADA para las no cubiertas -- ahí se usa el mensaje
    fijo de abstención, sin tocar el LLM. Costo deliberadamente bajo:
    sin claim/completeness validator LLM por subpregunta (eso
    multiplicaría el costo por N subpreguntas); en su lugar, un check
    heurístico barato (_validar_claim_basico) descarta subrespuestas
    obviamente no respaldadas por su propia evidencia."""
    prompt = ChatPromptTemplate.from_template(SUBANSWER_PROMPT_TEMPLATE)
    llm_generation_calls = 0
    lineas = []
    fuentes_todas: list[Document] = []

    for i, sub in enumerate(batch.subquestions, 1):
        pregunta_limpia = sub.question.strip().rstrip("?¿").strip()

        if not sub.covered or not sub.evidence:
            lineas.append(f"{i}. {pregunta_limpia}: {NO_EVIDENCIA_SUBPREGUNTA}")
            continue

        contexto_sub = formatear_contexto(sub.evidence)
        try:
            llm_generation_calls += 1
            respuesta = llm.invoke(
                prompt.format_messages(context=contexto_sub, question=sub.question)
            )
            texto = str(getattr(respuesta, "content", "") or "").strip()
        except Exception as e:
            log(f"    -> ERROR LLM subrespuesta '{sub.question}': {e}")
            texto = ""

        if not texto:
            lineas.append(f"{i}. {pregunta_limpia}: {NO_EVIDENCIA_SUBPREGUNTA}")
            sub.covered = False
            continue

        # Check heurístico barato (sin LLM): ¿la subrespuesta tiene
        # AL MENOS un claim con respaldo básico en su propia evidencia?
        frases = dividir_en_frases(texto)
        hay_respaldo = False
        for frase in frases[:5]:
            ok, _overlap, _reason = _validar_claim_basico(frase, contexto_sub, QueryContext(question=sub.question, sensitive=False))
            if ok:
                hay_respaldo = True
                break

        if not hay_respaldo and frases:
            lineas.append(f"{i}. {pregunta_limpia}: {NO_EVIDENCIA_SUBPREGUNTA}")
            sub.covered = False
            continue

        sub.answer = texto
        fuentes_todas.extend(sub.evidence)
        lineas.append(f"{i}. {pregunta_limpia}: {texto}")

    respuesta_final = "\n".join(lineas)

    # Recalcular cobertura real (post heurística, puede bajar si algún
    # LLM call falló o el check heurístico descartó la subrespuesta).
    covered_final = sum(1 for s in batch.subquestions if s.covered)
    total = len(batch.subquestions)
    coverage_final = round(covered_final / total, 4) if total else 0.0

    return respuesta_final, {
        "validated": True,
        "repaired": False,
        "batch": True,
        "subquestions": total,
        "covered_subquestions": covered_final,
        "uncovered_subquestions": total - covered_final,
        "coverage_score": coverage_final,
        "llm_generation_calls": llm_generation_calls,
        "llm_validation_calls": 0,
        "fuentes_batch": fuentes_todas,
    }


def ejecutar_pipeline(question: str) -> PipelineResult:
    metricas_tiempo = {}
    t0 = time.time()
    ctx = detectar_intencion(question)
    metricas_tiempo["query_analysis"] = time.time() - t0

    t1 = time.time()
    candidatos, ranking, aislados, evidencia, scores, answerable, answer_score, answer_reason = _un_pase_pipeline(ctx)
    metricas_tiempo["retrieval_pase_1"] = time.time() - t1

    # --------------------------------------------------------
    # Item 15: reintento con señal relajada ANTES de abstenerse.
    # P1 (auditoría): la relajación ahora pasa primero por
    # puede_relajar_answerability(), que bloquea explícitamente los
    # casos de documento/página/área incorrectos -- ahí NO se relaja,
    # se abstiene con el motivo real.
    # --------------------------------------------------------
    intento_relajado = False
    if not answerable and candidatos:
        se_puede_relajar, motivo_relajacion = puede_relajar_answerability(ctx, candidatos)
        if se_puede_relajar:
            intento_relajado = True
            t2 = time.time()
            top = candidatos[0]
            best_hybrid = float(top.metadata.get("_hybrid_score", 0.0) or 0.0)
            best_sem = float(top.metadata.get("_semantic_score", 0.0) or 0.0)
            umbral_relajado = max(0.30, ANSWERABILITY_MIN_GENERAL - 0.10)
            if best_hybrid >= umbral_relajado or best_sem >= ANSWERABILITY_SEMANTIC_ONLY_MIN:
                evidencia, scores = seleccionar_evidencia(ctx, aislados)
                if evidencia:
                    answerable = True
                    answer_score = best_hybrid
                    answer_reason = f"reintento con umbral relajado ({motivo_relajacion})"
            metricas_tiempo["reintento_relajado"] = time.time() - t2
        else:
            answer_reason = f"sin reintento: {motivo_relajacion}"

    if not answerable:
        evidencia = []
        scores = []

    # --------------------------------------------------------
    # Modo resumen (item 18): una vez identificado el documento
    # (vía ranking/aislamiento normal), el resumen no debe limitarse
    # a los FINAL_CONTEXT_CHUNKS más similares -> se recupera TODO
    # el documento aparte, para el map-reduce.
    # --------------------------------------------------------
    modo_resumen = False
    fuente_resumen = None
    if ctx.es_resumen and ranking:
        fuente_resumen = ctx.fuente_explicita or ranking[0][1]
        modo_resumen = True
        answerable = True
        answer_reason = "modo resumen: documento identificado"
        answer_score = ranking[0][0] if ranking else 0.0

    scores_hybrid = [float(d.metadata.get("_hybrid_score", 0.0) or 0.0) for d in candidatos]
    documentos_considerados = len({d.metadata.get("source") for d in candidatos})
    paginas_consideradas = len({(d.metadata.get("source"), d.metadata.get("page")) for d in candidatos})
    ocr_documentos = len({d.metadata.get("source") for d in candidatos if d.metadata.get("ocr")})

    # Item 11 (auditoría): estructura consistente para observabilidad,
    # además del log en texto libre -- esto es lo que un endpoint
    # /metrics o un dashboard consumiría sin tener que parsear logs.
    metricas = {
        "query_analysis_ms": round(metricas_tiempo.get("query_analysis", 0.0) * 1000, 1),
        "retrieval_ms": round(metricas_tiempo.get("retrieval_pase_1", 0.0) * 1000, 1),
        "reintento_relajado_ms": round(metricas_tiempo.get("reintento_relajado", 0.0) * 1000, 1),
        "documents_considered": documentos_considerados,
        "pages_considered": paginas_consideradas,
        "chunks_retrieved": len(candidatos),
        "chunks_final": len(evidencia),
        "max_hybrid_score": round(max(scores_hybrid), 4) if scores_hybrid else 0.0,
        "avg_hybrid_score": round(sum(scores_hybrid) / len(scores_hybrid), 4) if scores_hybrid else 0.0,
        "answerability_score": round(answer_score, 4),
        "answerability_reason": answer_reason,
        "retry_relaxed_used": intento_relajado,
        "area_detected": ctx.area,
        "profile_detected": ctx.perfil,
        "sensitive_query": ctx.sensitive,
        "modo_resumen": modo_resumen,
        "ocr_documents_in_candidates": ocr_documentos,
    }

    log(
        "\n--- Métricas del pipeline ---\n"
        f"  Documentos candidatos (fuentes distintas): {documentos_considerados}\n"
        f"  Chunks candidatos: {len(candidatos)}\n"
        f"  Chunks finales de evidencia: {len(evidencia)}\n"
        f"  Score máximo: {max(scores_hybrid):.4f}" if scores_hybrid else "  Score máximo: 0.0000"
    )
    if scores_hybrid:
        log(f"  Score promedio: {sum(scores_hybrid) / len(scores_hybrid):.4f}")
    log(f"  Reintento relajado usado: {'sí' if intento_relajado else 'no'}")
    for fase, dur in metricas_tiempo.items():
        log(f"  Tiempo {fase}: {dur:.2f}s")

    return PipelineResult(
        ctx=ctx,
        docs_candidatos=candidatos,
        ranking_documentos=ranking,
        docs_aislados=aislados,
        docs_evidencia=evidencia,
        scores_evidencia=scores,
        answerable=answerable,
        answerability_score=answer_score,
        answerability_reason=answer_reason,
        modo_resumen=modo_resumen,
        fuente_resumen=fuente_resumen,
        metricas=metricas,
    )


# ============================================================
# PASO 8 · PROMPT GENERALISTA (multi-área)
# ============================================================

PROMPT_TEMPLATE = """
Eres el asistente documental local de la institución.

Tu fuente de verdad es EXCLUSIVAMENTE el CONTEXTO RECUPERADO.
El sistema consulta documentación de distintas áreas institucionales:
legal, administrativa, académica, técnica, financiera, de recursos
humanos, operativa, procedimental u otras.

REGLAS OBLIGATORIAS:

1. Responde únicamente con información explícita del contexto.
2. No uses conocimiento externo.
3. No inventes, rellenes ni completes datos ausentes.
4. No confundas personas, cargos, entidades, fechas, números ni identificadores.
5. Una fecha de un hecho no equivale a una fecha de emisión, firma o publicación.
6. Un rol o cargo no equivale a otro distinto (autor ≠ responsable ≠ firmante ≠ juez, etc.), según lo que diga el documento.
7. Un identificador de un tipo no equivale a otro tipo (expediente ≠ resolución ≠ oficio ≠ memorando ≠ contrato ≠ factura ≠ código técnico).
8. Un monto, porcentaje o cantidad debe mantenerse exactamente como aparece.
9. Si una parte del texto parece dañada por OCR, no la corrijas por intuición.
10. Para nombres, números, códigos, expedientes, contratos, facturas, fechas, montos y otros identificadores exige correspondencia explícita.
11. Si el usuario solicita una página concreta, utiliza únicamente esa página.
12. Si el usuario solicita un documento o un área concreta, utiliza únicamente ese alcance, salvo que pida comparación o análisis conjunto.
13. Si el usuario formula varias subpreguntas, responde cada una de manera independiente.
14. No introduzcas información de otra subpregunta dentro de una respuesta distinta.
15. Si falta evidencia para una subpregunta, dilo para esa subpregunta.
16. Si existen contradicciones en el contexto, señálalas y no elijas arbitrariamente.
17. Si una afirmación requiere inferencia, no la presentes como hecho documental.
18. No inventes citas ni números de página.
19. No incluyas una sección "Fuentes". Las fuentes serán añadidas por Python.
20. Da prioridad a exactitud, trazabilidad y completitud sobre extensión.
21. Si la pregunta pide un dato concreto, da primero ese dato y luego una aclaración breve.

CONTEXTO RECUPERADO:

{context}

PREGUNTA:

{question}

RESPUESTA:
"""


# ============================================================
# MODO RESUMEN (item 18) · map-reduce sobre el documento completo
# ============================================================
#
# No reutiliza FINAL_CONTEXT_CHUNKS=8: un resumen de "todo el
# documento" necesita ver todo el documento (hasta un tope), no
# solo los 8 chunks más similares a la pregunta "resume esto".
# ============================================================

MAX_RESUMEN_TOTAL_CHUNKS = 160          # tope duro por documento
RESUMEN_BLOQUE_MAX_CHARS = 12000        # ~ mitad de CONTEXTO_LLM, deja margen para prompt + salida

RESUMEN_BLOQUE_PROMPT = """
Eres un asistente documental. Resume EXCLUSIVAMENTE la información
contenida en el siguiente fragmento del documento.

No inventes ni agregues información externa.
Sé conciso pero conserva datos concretos: fechas, números,
identificadores, nombres, montos y conclusiones textuales.

FRAGMENTO:
{context}

RESUMEN DEL FRAGMENTO:
"""

RESUMEN_SINTESIS_PROMPT = """
Eres un asistente documental. Tienes varios resúmenes parciales
del MISMO documento, en el orden en que aparecen en el documento.

Combínalos en un resumen final coherente, sin repeticiones,
atendiendo la intención de la pregunta original si aplica.
No inventes información que no esté en los resúmenes parciales.
No agregues una sección "Fuentes".

PREGUNTA ORIGINAL (puede ser genérica, ej. "resume el documento"):
{question}

RESÚMENES PARCIALES EN ORDEN:
{resumenes}

RESUMEN FINAL:
"""


def obtener_todos_los_chunks_de_fuente(source: str) -> list[Document]:
    """A diferencia de recuperar_candidatos (limitado a RETRIEVER_K por
    similitud), esto trae TODOS los chunks indexados de un documento,
    necesarios para poder resumirlo completo."""
    try:
        data = vector_db.get(where={"source": source}, include=["documents", "metadatas"])
    except Exception as e:
        log(f"ADVERTENCIA obteniendo chunks completos de '{source}': {e}")
        return []

    docs = []
    for content, metadata in zip(data.get("documents") or [], data.get("metadatas") or []):
        metadata = dict(metadata or {})
        docs.append(Document(page_content=content or "", metadata=metadata))

    docs.sort(key=lambda d: (int(d.metadata.get("page", 0) or 0), int(d.metadata.get("chunk_index", 0) or 0)))
    return docs


def generar_resumen_documento(llm, ctx: QueryContext, chunks: list[Document]) -> str:
    if not chunks:
        return "No encontré información suficiente en los documentos indexados para responder esa pregunta."

    total_original = len(chunks)
    truncado = total_original > MAX_RESUMEN_TOTAL_CHUNKS
    chunks = chunks[:MAX_RESUMEN_TOTAL_CHUNKS]

    bloques: list[list[Document]] = []
    actual: list[Document] = []
    chars_actual = 0
    for chunk in chunks:
        texto = chunk.page_content or ""
        if actual and chars_actual + len(texto) > RESUMEN_BLOQUE_MAX_CHARS:
            bloques.append(actual)
            actual = []
            chars_actual = 0
        actual.append(chunk)
        chars_actual += len(texto)
    if actual:
        bloques.append(actual)

    log(f"    -> Modo resumen: {len(chunks)} chunks en {len(bloques)} bloque(s).")
    if truncado:
        log(
            f"    -> ADVERTENCIA: el documento tiene {total_original} chunks, "
            f"por encima del tope MAX_RESUMEN_TOTAL_CHUNKS={MAX_RESUMEN_TOTAL_CHUNKS}. "
            f"El resumen cubrirá solo los primeros {MAX_RESUMEN_TOTAL_CHUNKS}."
        )

    aviso_truncado = (
        f"\n\n_Nota: este documento es extenso ({total_original} fragmentos indexados). "
        f"El resumen cubre las primeras secciones (hasta {MAX_RESUMEN_TOTAL_CHUNKS} fragmentos); "
        f"puede no incluir el contenido final del documento. Pregunta por una página o "
        f"sección específica si necesitas esa parte._"
        if truncado else ""
    )

    prompt_bloque = ChatPromptTemplate.from_template(RESUMEN_BLOQUE_PROMPT)
    resumenes_parciales = []
    for i, bloque in enumerate(bloques, 1):
        contexto_bloque = formatear_contexto(bloque)
        try:
            respuesta = llm.invoke(prompt_bloque.format_messages(context=contexto_bloque))
            texto = str(getattr(respuesta, "content", "") or "").strip()
            if texto:
                resumenes_parciales.append(texto)
        except Exception as e:
            log(f"    -> Error resumiendo bloque {i}/{len(bloques)}: {e}")

    if not resumenes_parciales:
        return "No encontré información suficiente en los documentos indexados para responder esa pregunta."

    if len(resumenes_parciales) == 1:
        return resumenes_parciales[0] + aviso_truncado

    prompt_sintesis = ChatPromptTemplate.from_template(RESUMEN_SINTESIS_PROMPT)
    try:
        respuesta = llm.invoke(
            prompt_sintesis.format_messages(
                question=ctx.question,
                resumenes="\n\n---\n\n".join(resumenes_parciales),
            )
        )
        final = str(getattr(respuesta, "content", "") or "").strip()
        return (final if final else "\n\n".join(resumenes_parciales)) + aviso_truncado
    except Exception as e:
        log(f"    -> Error en síntesis final del resumen: {e}")
        return "\n\n".join(resumenes_parciales) + aviso_truncado


# ============================================================
# PASO 9 · CLAIM / COMPLETENESS VALIDATION
# ============================================================

CLAIM_VALIDATION_PROMPT = """
Eres un verificador estricto de evidencia documental.

Determina si la afirmación aparece respaldada explícitamente por el contexto.

Responde exclusivamente:
RESPALDADA
NO_RESPALDADA

CONTEXTO:
{context}

AFIRMACIÓN:
{claim}
"""


COMPLETENESS_VALIDATION_PROMPT = """
Eres un evaluador de calidad de respuestas documentales.

Determina si la respuesta:
1) contesta la pregunta planteada,
2) incluye la información esencial solicitada,
3) no omite una parte evidente de una pregunta compuesta,
4) no introduce datos ajenos al contexto.

Responde exclusivamente:
COMPLETA
INCOMPLETA

PREGUNTA:
{question}

CONTEXTO:
{context}

RESPUESTA:
{answer}
"""


REPAIR_PROMPT_TEMPLATE = """
Eres un revisor de respuestas documentales.

Debes corregir la respuesta utilizando exclusivamente el contexto autorizado.

PREGUNTA:
{question}

CONTEXTO AUTORIZADO:
{context}

RESPUESTA ORIGINAL:
{answer}

REGLAS:
1. Conserva únicamente información explícitamente respaldada.
2. Completa la respuesta únicamente con información que sí aparezca en el contexto.
3. No inventes ni hagas inferencias.
4. No corrijas OCR por intuición.
5. No cambies nombres, números, fechas o montos por aproximación.
6. Si hay varias subpreguntas, responde cada una.
7. Si una subpregunta no tiene evidencia suficiente, indícalo claramente.
8. No agregues una sección "Fuentes".
9. Entrega solo la respuesta final corregida.
"""


# ============================================================
# CONTEXTO / FUENTES
# ============================================================


def formatear_contexto(docs: list[Document]) -> str:
    if not docs:
        return ""

    partes = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        area = doc.metadata.get("area", DEFAULT_AREA)
        page = doc.metadata.get("page")
        try:
            page_label = str(int(page) + 1) if page is not None else "?"
        except (ValueError, TypeError):
            page_label = str(page)

        partes.append(
            f"=== EVIDENCIA {i} ===\n"
            f"Área: {area}\n"
            f"Archivo: {source}\n"
            f"Página: {page_label}\n"
            f"Contenido:\n{doc.page_content}"
        )

    return "\n\n".join(partes)


def medir_contexto(texto: str) -> dict:
    caracteres = len(texto)
    palabras = len(texto.split())
    tokens_estimados = max(1, caracteres // 4)
    return {
        "caracteres": caracteres,
        "palabras": palabras,
        "tokens_estimados": tokens_estimados,
    }


def formatear_fuentes(docs: list[Document]) -> list[dict]:
    fuentes = []
    vistos = set()
    for doc in docs:
        source = doc.metadata.get("source", "?")
        area = doc.metadata.get("area", DEFAULT_AREA)
        page = doc.metadata.get("page")
        try:
            page_number = int(page) + 1 if page is not None else None
        except (ValueError, TypeError):
            page_number = None
        clave = (source, page_number)
        if clave in vistos:
            continue
        vistos.add(clave)
        fuentes.append({"source": source, "area": area, "page": page_number})
    return fuentes


# ============================================================
# VALIDACIÓN HEURÍSTICA (entidades genéricas de cualquier área)
# ============================================================


def _detecta_inferencia(claim: str) -> bool:
    q = normalizar_texto(claim)
    patrones = [
        r"\bse puede inferir\b", r"\bpuede inferirse\b", r"\bpor lo tanto\b",
        r"\bpor ende\b", r"\bpor consiguiente\b", r"\besto demuestra\b",
        r"\besto significa\b", r"\blo que indica\b", r"\blo cual indica\b",
        r"\bde ello se desprende\b", r"\ben consecuencia\b", r"\bprobablemente\b",
        r"\bse deduce\b", r"\bse puede deducir\b", r"\bdebe haber sido\b",
        r"\bdebio haber sido\b", r"\bdespues de\b", r"\bantes de\b",
    ]
    return any(re.search(p, q) for p in patrones)


def _extraer_fechas(texto: str) -> set[str]:
    t = normalizar_texto(texto)
    resultados = set()
    patrones = [
        r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+del?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    ]
    for pattern in patrones:
        resultados.update(re.findall(pattern, t))
    return resultados


def _extraer_ids(texto: str) -> set[str]:
    """Identificadores institucionales genéricos: expedientes, resoluciones,
    oficios, memorandos, contratos, facturas, códigos técnicos, DNI/RUC, etc.
    No se limita al formato judicial."""
    t = normalizar_texto(texto)
    ids = set()
    ids.update(re.findall(r"\b\d{8}\b", t))               # DNI
    ids.update(re.findall(r"\b\d{11}\b", t))               # RUC
    ids.update(re.findall(r"\b\d{2,6}-\d{1,6}-\d{4}-\d-[a-z0-9-]+\b", t))  # expediente judicial
    ids.update(re.findall(
        r"\b(?:expediente|resolucion|proceso|oficio|memorando|"
        r"memorandum|informe|contrato|factura|orden de compra|"
        r"orden de servicio|convenio|acta|proveido|dictamen|"
        r"comprobante|codigo)\s+(?:n[°º.]?\s*)?[a-z0-9./-]+\b",
        t,
    ))
    return {x.strip() for x in ids}


def _extraer_montos(texto: str) -> set[str]:
    t = normalizar_texto(texto)
    valores = re.findall(r"\b(?:s/|s)\s*\d[\d\.,]*\b|\b\d[\d\.,]*\s*(?:soles|usd|dolares)\b", t)
    return set(valores)


def _extraer_secuencias_nombre(claim: str) -> list[str]:
    clean = re.sub(r"[*_`\"()]", " ", claim)
    patrones = re.findall(
        r"\b(?:[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]{2,}\s+){1,}[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]{2,}\b",
        clean,
    )
    stop_phrases = {"No encontré Información", "Fuentes", "La Página"}
    return [p.strip() for p in patrones if p.strip() not in stop_phrases]


def _validar_consistencia_entidades(claim: str, context: str, ctx: QueryContext) -> tuple[bool, str]:
    claim_norm = normalizar_texto(claim)
    context_norm = normalizar_texto(context)

    claim_ids = _extraer_ids(claim)
    if claim_ids:
        context_ids = _extraer_ids(context)
        faltantes = claim_ids - context_ids
        if faltantes:
            return False, "identificador no presente exactamente"

    claim_dates = _extraer_fechas(claim)
    if claim_dates:
        context_dates = _extraer_fechas(context)
        if not claim_dates.issubset(context_dates):
            return False, "fecha no presente exactamente"

    claim_amounts = _extraer_montos(claim)
    if claim_amounts:
        context_amounts = _extraer_montos(context)
        if not claim_amounts.issubset(context_amounts):
            return False, "monto no presente exactamente"

    nombre_phrases = _extraer_secuencias_nombre(claim)
    if nombre_phrases and ctx.sensitive:
        for phrase in nombre_phrases:
            norm_phrase = normalizar_texto(phrase)
            if len(norm_phrase.split()) >= 3:
                if norm_phrase not in context_norm:
                    return False, f"entidad/persona no coincide exactamente: {phrase}"

    return True, "consistencia exacta"


def _validar_claim_basico(claim: str, context: str, ctx: QueryContext) -> tuple[bool, float, str]:
    claim = claim.strip()
    if len(claim) < 8:
        return True, 1.0, "fragmento no factual"

    if _detecta_inferencia(claim):
        return False, 0.0, "posible inferencia"

    claim_tokens = tokens_significativos(claim)
    context_tokens = tokens_significativos(context)
    if not claim_tokens:
        return True, 1.0, "sin tokens críticos"

    inter = claim_tokens & context_tokens
    overlap = len(inter) / max(1, len(claim_tokens))

    numeros_claim = extraer_numeros(claim)
    numeros_context = extraer_numeros(context)
    if numeros_claim - numeros_context:
        return False, overlap, "número no presente"

    perfiles_sensibles_area = _perfiles_sensibles_de_area(ctx.area)
    sensible = ctx.sensitive or any(
        normalizar_texto(perfil.replace("_", " ")) in normalizar_texto(claim)
        for perfil in perfiles_sensibles_area
    )

    if sensible:
        ok_entity, entity_reason = _validar_consistencia_entidades(claim, context, ctx)
        if not ok_entity:
            return False, overlap, entity_reason
        if overlap >= MIN_CLAIM_TOKEN_OVERLAP_STRICT:
            return True, overlap, "afirmación sensible respaldada"
        return False, overlap, "afirmación sensible débil"

    if overlap >= 0.45:
        return True, overlap, "alta coincidencia"
    if overlap >= MIN_CLAIM_TOKEN_OVERLAP:
        return True, overlap, "coincidencia suficiente"
    return False, overlap, "baja coincidencia"


def validar_claim_con_llm(llm, context: str, claim: str) -> Optional[bool]:
    """Devuelve True (respaldada), False (no respaldada) o None (el
    validator FALLÓ -- estado VALIDATION_ERROR).

    P0 (auditoría): antes, un error de Ollama/timeout hacía `return True`
    (fail-open), lo que podía dar por respaldada CUALQUIER afirmación
    cuando el validator en realidad nunca se ejecutó. Para un RAG
    institucional eso es peligroso: ahora el error se distingue
    explícitamente (None) y quien llama decide -- pero el default en
    `validar_respuesta()` es tratar None como NO respaldada (fail-closed),
    nunca como validación exitosa."""
    if not CLAIM_VALIDATOR_LLM_ENABLED:
        return True
    prompt = ChatPromptTemplate.from_template(CLAIM_VALIDATION_PROMPT)
    try:
        respuesta = llm.invoke(
            prompt.format_messages(context=context, claim=claim)
        )
        contenido = str(getattr(respuesta, "content", "") or "").strip().upper()
        return contenido.startswith("RESPALDADA")
    except Exception as e:
        log(f"    -> VALIDATION_ERROR (claim validator): {e}")
        return None


def validar_respuesta(llm, answer: str, docs: list[Document], ctx: QueryContext, presupuesto: dict) -> dict:
    """Item 11/14 (auditoría hardening): la heurística (barata) corre
    SIEMPRE sobre todos los claims. El LLM validator es el recurso
    caro -- se reserva para los MAX_LLM_CLAIMS_TO_VERIFY claims más
    riesgosos (sensibles primero, luego menor overlap = más ambiguo),
    y respeta `presupuesto` (el circuit breaker compartido de la
    consulta completa: claim validator + completeness + repair no
    pueden sumar más de MAX_VALIDATION_LLM_CALLS llamadas LLM)."""
    answer_clean = limpiar_fuentes_generadas(answer)
    context = formatear_contexto(docs)
    frases = dividir_en_frases(answer_clean)

    claims = []
    for claim in frases[:MAX_CLAIMS_TO_VERIFY]:
        ok, overlap, reason = _validar_claim_basico(claim, context, ctx)
        necesita_llm = (
            ok
            and VALIDATOR_ENABLED
            and CLAIM_VALIDATOR_LLM_ENABLED
            and len(claim) >= 20
            and (ctx.sensitive or overlap < 0.75)
        )
        claims.append({
            "claim": claim,
            "ok": ok,
            "overlap": overlap,
            "reason": reason,
            "_necesita_llm": necesita_llm,
            # riesgo: sensible primero (0 antes que 1), luego menor
            # overlap primero (más ambiguo = más riesgoso).
            "_riesgo": (0 if ctx.sensitive else 1, overlap),
        })

    candidatos_llm = sorted(
        (c for c in claims if c["_necesita_llm"]),
        key=lambda c: c["_riesgo"],
    )[:MAX_LLM_CLAIMS_TO_VERIFY]

    for item in candidatos_llm:
        if presupuesto.get("llm_calls", 0) >= MAX_VALIDATION_LLM_CALLS:
            presupuesto["budget_exhausted"] = True
            item["reason"] = item["reason"] + " (sin verificar por LLM: presupuesto de validación agotado)"
            break
        presupuesto["llm_calls"] = presupuesto.get("llm_calls", 0) + 1
        resultado_llm = validar_claim_con_llm(llm, context, item["claim"])
        if resultado_llm is None:
            # P0: VALIDATION_ERROR -- el validator no pudo ejecutarse.
            # Fail-closed: nunca se declara "validado" por default.
            item["ok"] = False
            item["reason"] = "VALIDATION_ERROR: el validator LLM falló, claim tratado como no respaldado"
        elif not resultado_llm:
            item["ok"] = False
            item["reason"] = "validator LLM: no respaldada"
        else:
            item["reason"] = "validator LLM: respaldada"

    for item in claims:
        item.pop("_necesita_llm", None)
        item.pop("_riesgo", None)

    unsupported = [c for c in claims if not c["ok"]]

    return {
        "ok": len(unsupported) == 0,
        "claims": claims,
        "unsupported": unsupported,
        "answer_clean": answer_clean,
        "llm_claim_checks": len(candidatos_llm),
    }


def validar_completitud_con_llm(llm, question: str, context: str, answer: str) -> Optional[bool]:
    """Devuelve True (completa), False (incompleta) o None (VALIDATION_ERROR).

    P0 (auditoría): antes, un error del validator devolvía `True`
    (fail-open) -> una respuesta podía declararse "completa" sin haber
    sido evaluada. Ahora el error se distingue y el llamador decide;
    el default aguas abajo es fail-closed (tratar como incompleta)."""
    if not COMPLETENESS_VALIDATOR_LLM_ENABLED:
        return True
    prompt = ChatPromptTemplate.from_template(COMPLETENESS_VALIDATION_PROMPT)
    try:
        respuesta = llm.invoke(
            prompt.format_messages(
                question=question,
                context=context,
                answer=answer,
            )
        )
        contenido = str(getattr(respuesta, "content", "") or "").strip().upper()
        return contenido.startswith("COMPLETA")
    except Exception as e:
        log(f"    -> VALIDATION_ERROR (completeness validator): {e}")
        return None


# ============================================================
# PASO 10 · REPARACIÓN / REVALIDACIÓN
# ============================================================


def reparar_respuesta(llm, question: str, answer: str, docs: list[Document]) -> str:
    context = formatear_contexto(docs)
    prompt = ChatPromptTemplate.from_template(REPAIR_PROMPT_TEMPLATE)
    try:
        respuesta = llm.invoke(
            prompt.format_messages(
                question=question,
                context=context,
                answer=answer,
            )
        )
        contenido = str(getattr(respuesta, "content", "") or "").strip()
        return contenido if contenido else answer
    except Exception as e:
        log(f"    -> Error reparación: {e}")
        return answer


def generar_respuesta_validada(llm, prompt, question: str, docs: list[Document], ctx: QueryContext) -> tuple[str, dict]:
    context = formatear_contexto(docs)
    if not context:
        return (
            "No encontré información suficiente en los documentos indexados para responder esa pregunta.",
            {"validated": True, "repaired": False},
        )

    try:
        respuesta = llm.invoke(prompt.format_messages(context=context, question=question))
        draft = str(getattr(respuesta, "content", "") or "").strip()
    except Exception as e:
        log(f"\nERROR LLM: {e}")
        return (
            "Error al generar la respuesta con Ollama.",
            {"validated": False, "repaired": False, "error": True},
        )

    if not draft:
        return (
            "No encontré información suficiente en los documentos indexados para responder esa pregunta.",
            {"validated": True, "repaired": False},
        )

    # Item 14 (auditoría hardening): presupuesto compartido de llamadas
    # LLM de validación (claim + completeness + repair-revalidación)
    # para TODA esta respuesta. Es un dict mutable a propósito: se pasa
    # por referencia y cada función que consume presupuesto lo actualiza.
    presupuesto = {"llm_calls": 0, "budget_exhausted": False}

    validacion = validar_respuesta(llm, draft, docs, ctx, presupuesto)
    completeness_ok = True
    completeness_status = "not_run"

    if VALIDATOR_ENABLED and ctx.requires_completeness:
        if presupuesto["llm_calls"] >= MAX_VALIDATION_LLM_CALLS:
            presupuesto["budget_exhausted"] = True
            completeness_status = "budget_exhausted"
            completeness_ok = False  # fail-closed: no evaluado, no se asume completa
        else:
            presupuesto["llm_calls"] += 1
            resultado_completitud = validar_completitud_con_llm(
                llm,
                question,
                context,
                validacion["answer_clean"],
            )
            if resultado_completitud is None:
                # P0: VALIDATION_ERROR -- fail-closed, nunca se asume completa.
                completeness_ok = False
                completeness_status = "validation_error"
            else:
                completeness_ok = resultado_completitud
                completeness_status = "ok" if resultado_completitud else "incomplete"

    if validacion["ok"] and completeness_ok:
        return (
            validacion["answer_clean"],
            {
                "validated": True,
                "repaired": False,
                "unsupported": 0,
                "complete": True,
                "completeness_status": completeness_status,
                "llm_validation_calls": presupuesto["llm_calls"],
                "validation_budget_exhausted": presupuesto["budget_exhausted"],
            },
        )

    # Item 13: reparación selectiva -- solo si hay algo reparable
    # (claims no soportados) o incompletitud con evidencia adicional
    # disponible (docs no vacíos), y solo UN intento (MAX_REPAIR_ATTEMPTS).
    hay_algo_que_reparar = len(validacion["unsupported"]) > 0 or not completeness_ok
    puede_reparar = bool(docs) and MAX_REPAIR_ATTEMPTS >= 1
    presupuesto_disponible = presupuesto["llm_calls"] < MAX_VALIDATION_LLM_CALLS

    if (
        hay_algo_que_reparar
        and puede_reparar
        and presupuesto_disponible
        and (REPAIR_UNSUPPORTED_CLAIMS or (REPAIR_INCOMPLETE_ANSWER and not completeness_ok))
    ):
        presupuesto["llm_calls"] += 1
        repaired = reparar_respuesta(
            llm,
            question,
            validacion["answer_clean"],
            docs,
        )

        validacion_reparada = validar_respuesta(llm, repaired, docs, ctx, presupuesto)
        if VALIDATOR_ENABLED and presupuesto["llm_calls"] < MAX_VALIDATION_LLM_CALLS:
            presupuesto["llm_calls"] += 1
            resultado_completitud_reparada = validar_completitud_con_llm(
                llm,
                question,
                context,
                validacion_reparada["answer_clean"],
            )
            # Fail-closed también tras reparar: un error del validator
            # aquí NO debe convertir una respuesta reparada en "completa".
            completeness_reparada = bool(resultado_completitud_reparada)
        else:
            presupuesto["budget_exhausted"] = True
            completeness_reparada = False

        if validacion_reparada["ok"] and completeness_reparada:
            return (
                validacion_reparada["answer_clean"],
                {
                    "validated": True,
                    "repaired": True,
                    "unsupported_before": len(validacion["unsupported"]),
                    "complete": True,
                    "llm_validation_calls": presupuesto["llm_calls"],
                    "validation_budget_exhausted": presupuesto["budget_exhausted"],
                },
            )

        claims_validos = [x["claim"] for x in validacion["claims"] if x["ok"]]
        if claims_validos:
            return (
                "\n".join(claims_validos),
                {
                    "validated": True,
                    "repaired": True,
                    "partial": True,
                    "complete": False,
                    "unsupported_before": len(validacion["unsupported"]),
                    "llm_validation_calls": presupuesto["llm_calls"],
                    "validation_budget_exhausted": presupuesto["budget_exhausted"],
                },
            )

    return (
        "No encontré información suficiente en los documentos indexados para responder esa pregunta.",
        {
            "validated": False,
            "repaired": False,
            "failed_validation": True,
            "complete": False,
            "unsupported": len(validacion["unsupported"]),
            "llm_validation_calls": presupuesto["llm_calls"],
            "validation_budget_exhausted": presupuesto["budget_exhausted"],
        },
    )


# ============================================================
# LLM / CHAIN
# ============================================================


def construir_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS_RESPUESTA,
        num_ctx=CONTEXTO_LLM,
        reasoning=False,
        streaming=False,
    )


def construir_chain():
    llm = construir_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    class ChainCompatible:
        def stream(self, question, docs=None):
            if docs is None:
                # Items 3-9 (auditoría hardening): batch multi-pregunta
                # tiene prioridad sobre el camino de pregunta única.
                batch = ejecutar_pipeline_batch(question)
                if batch is not None:
                    respuesta, validacion = generar_respuesta_batch(llm, batch)
                    log(
                        f"\n--- Batch: {validacion['covered_subquestions']}/"
                        f"{validacion['subquestions']} cubiertas "
                        f"(coverage_score={validacion['coverage_score']}) ---"
                    )
                    for pos in range(0, len(respuesta), 160):
                        yield respuesta[pos:pos + 160]
                    return

                resultado = ejecutar_pipeline(question)
                if not resultado.answerable:
                    yield "No encontré información suficiente en los documentos indexados para responder esa pregunta."
                    return

                if resultado.modo_resumen and resultado.fuente_resumen:
                    chunks_completos = obtener_todos_los_chunks_de_fuente(resultado.fuente_resumen)
                    respuesta = generar_resumen_documento(llm, resultado.ctx, chunks_completos)
                    for pos in range(0, len(respuesta), 160):
                        yield respuesta[pos:pos + 160]
                    return

                docs = resultado.docs_evidencia
                ctx = resultado.ctx
            else:
                ctx = detectar_intencion(question)

            if not docs:
                yield "No encontré información suficiente en los documentos indexados para responder esa pregunta."
                return

            contexto = formatear_contexto(docs)
            metricas = medir_contexto(contexto)
            log(
                "\n--- Métricas del contexto ---\n"
                f"  Área: {ctx.area}\n"
                f"  Chunks: {len(docs)}\n"
                f"  Caracteres: {metricas['caracteres']}\n"
                f"  Palabras: {metricas['palabras']}\n"
                f"  Tokens estimados: {metricas['tokens_estimados']}\n"
            )

            inicio = time.time()
            respuesta, validacion = generar_respuesta_validada(
                llm,
                prompt,
                question,
                docs,
                ctx,
            )

            log("\n--- Validación de respuesta ---")
            log(f"  Validada: {validacion.get('validated')}")
            log(f"  Reparada: {validacion.get('repaired')}")
            if "complete" in validacion:
                log(f"  Completa: {validacion.get('complete')}")
            if "unsupported_before" in validacion:
                log(f"  Claims no respaldados antes: {validacion['unsupported_before']}")
            if "unsupported" in validacion:
                log(f"  Claims no respaldados: {validacion['unsupported']}")
            log(f"  Tiempo LLM + validación: {time.time() - inicio:.2f}s")

            for pos in range(0, len(respuesta), 160):
                yield respuesta[pos:pos + 160]

        def invoke(self, question):
            batch = ejecutar_pipeline_batch(question)
            if batch is not None:
                respuesta, _ = generar_respuesta_batch(llm, batch)
                return Document(page_content=respuesta)

            resultado = ejecutar_pipeline(question)
            if not resultado.answerable:
                return Document(
                    page_content="No encontré información suficiente en los documentos indexados para responder esa pregunta."
                )
            if resultado.modo_resumen and resultado.fuente_resumen:
                chunks_completos = obtener_todos_los_chunks_de_fuente(resultado.fuente_resumen)
                respuesta = generar_resumen_documento(llm, resultado.ctx, chunks_completos)
                return Document(page_content=respuesta)
            if not resultado.docs_evidencia:
                return Document(
                    page_content="No encontré información suficiente en los documentos indexados para responder esa pregunta."
                )
            respuesta, _ = generar_respuesta_validada(
                llm,
                prompt,
                question,
                resultado.docs_evidencia,
                resultado.ctx,
            )
            return Document(page_content=respuesta)

    retriever = vector_db.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
    return ChainCompatible(), retriever


# ============================================================
# DEBUG
# ============================================================


def imprimir_debug(resultado: PipelineResult):
    ctx = resultado.ctx
    print("--- Paso 1: intención detectada ---")
    print(f"  Área: {ctx.area}" + (" (explícita)" if ctx.area_explicita else " (por defecto / global)"))
    print(f"  Fuente principal: {ctx.fuente_explicita or 'ninguna explícita'}")

    if ctx.fuentes_explicitas:
        print("  Fuentes explícitas:")
        for fuente in ctx.fuentes_explicitas:
            print(f"    - {fuente}")

    if ctx.pagina_explicita is not None:
        print(f"  Página detectada: {ctx.pagina_explicita + 1}")
    elif ctx.ultima_pagina_solicitada:
        print("  Página solicitada: última")
    else:
        print("  Página detectada: ninguna explícita")

    print(f"  Comparación: {'sí' if ctx.es_comparacion else 'no'}")
    print(f"  Tipo de consulta: {ctx.tipo_consulta}")
    print(f"  Perfil: {ctx.perfil}")
    print(f"  Sensible: {'sí' if ctx.sensitive else 'no'}")

    if ctx.subpreguntas:
        print("  Subpreguntas:")
        for sub in ctx.subpreguntas:
            print(f"    - {sub}")

    print("  Query variants:")
    for variant in ctx.query_variants:
        print(f"    - {variant}")

    print("\n--- Paso 2: candidatos recuperados ---")
    for doc in resultado.docs_candidatos:
        source = doc.metadata.get("source", "?")
        area = doc.metadata.get("area", DEFAULT_AREA)
        page = doc.metadata.get("page")
        try:
            visible = int(page) + 1
        except (ValueError, TypeError):
            visible = page

        dist = doc.metadata.get("_distance", "?")
        sem = doc.metadata.get("_semantic_score", "?")
        lex = doc.metadata.get("_lexical_score", "?")
        hybrid = doc.metadata.get("_hybrid_score", "?")
        preview = (doc.page_content[:110] or "").replace("\n", " ")

        print(
            f"  [{area} | {source} | pág {visible} | dist {dist} | "
            f"sem {sem} | lex {lex} | hybrid {hybrid}] {preview}..."
        )

    print("\n--- Answerability / Grounding Gate ---")
    print(f"  Answerable: {resultado.answerable}")
    print(f"  Score: {resultado.answerability_score:.4f}")
    print(f"  Razón: {resultado.answerability_reason}")

    print("\n--- Paso 5: evidencia final enviada al LLM ---")
    for score, doc in resultado.scores_evidencia:
        source = doc.metadata.get("source", "?")
        area = doc.metadata.get("area", DEFAULT_AREA)
        page = doc.metadata.get("page")
        try:
            visible = int(page) + 1
        except (ValueError, TypeError):
            visible = page
        preview = (doc.page_content[:120] or "").replace("\n", " ")
        expanded = " | vecino" if doc.metadata.get("_expanded") else ""
        print(
            f"  [{area} | {source} | pág {visible} | score {score:.4f}{expanded}] {preview}..."
        )

    print("-----------------------------------\n")


# ============================================================
# CLI
# ============================================================


def preguntar(chain, question: str, retriever):
    inicio = time.time()

    batch = ejecutar_pipeline_batch(question)
    if batch is not None:
        print(
            f"--- Batch: {len(batch.subquestions)} subpreguntas | "
            f"coverage_score={batch.coverage_score} ---\n"
        )
        for sub in batch.subquestions:
            estado = "cubierta" if sub.covered else "SIN evidencia"
            print(f"  [{estado}] {sub.question!r} (score {sub.answerability_score:.4f})")
        print()
        respuesta_completa = ""
        primer_fragmento = None
        for pedazo in chain.stream(question):
            if primer_fragmento is None:
                primer_fragmento = time.time()
                print(f"[primer fragmento en {primer_fragmento - inicio:.2f}s]\n")
            respuesta_completa += pedazo
            print(pedazo, end="", flush=True)
        print(f"\n\n[Completado en {time.time() - inicio:.2f}s]")
        return

    resultado = ejecutar_pipeline(question)
    imprimir_debug(resultado)
    print(f"[Contexto final: {len(resultado.docs_evidencia)} chunks | modo resumen: {resultado.modo_resumen}]\n")

    if not resultado.answerable:
        print("No encontré información suficiente en los documentos indexados para responder esa pregunta.")
        return
    if not resultado.modo_resumen and not resultado.docs_evidencia:
        print("No encontré información suficiente en los documentos indexados para responder esa pregunta.")
        return

    docs_para_stream = None if resultado.modo_resumen else resultado.docs_evidencia
    if resultado.modo_resumen:
        # El streaming compatible de ChainCompatible.stream() detecta el modo
        # resumen por sí mismo cuando docs=None, así que forzamos ese camino
        # reconstruyendo el pipeline internamente (misma pregunta).
        pass

    respuesta_completa = ""
    primer_fragmento = None

    fuente_iter = chain.stream(question) if resultado.modo_resumen else chain.stream(question, docs=docs_para_stream)
    for pedazo in fuente_iter:
        if primer_fragmento is None:
            primer_fragmento = time.time()
            print(f"[primer fragmento en {primer_fragmento - inicio:.2f}s]\n")
        respuesta_completa += pedazo
        print(pedazo, end="", flush=True)

    if resultado.modo_resumen:
        print(f"\n\nFuente: [{resultado.ctx.area}] {resultado.fuente_resumen} (resumen del documento completo)")
    else:
        fuentes = formatear_fuentes(resultado.docs_evidencia)
        print("\n\nFuentes:")
        for fuente in fuentes:
            print(f"- [{fuente['area']}] {fuente['source']}, página {fuente['page']}")

    print(f"\n[Completado en {time.time() - inicio:.2f}s]")


# ============================================================
# FASTAPI STREAM
# ============================================================


def preguntar_stream(chain, question: str, retriever):
    batch = ejecutar_pipeline_batch(question)
    if batch is not None:
        fuentes_batch = []
        vistos_batch = set()
        for sub in batch.subquestions:
            for doc in sub.evidence:
                clave = (doc.metadata.get("source"), doc.metadata.get("page"))
                if clave in vistos_batch:
                    continue
                vistos_batch.add(clave)
                fuentes_batch.append({
                    "source": doc.metadata.get("source", "?"),
                    "area": doc.metadata.get("area", DEFAULT_AREA),
                    "page": (doc.metadata.get("page") + 1) if doc.metadata.get("page") is not None else None,
                })
        yield (
            "data: " +
            json.dumps({"type": "sources", "data": fuentes_batch}, ensure_ascii=False) +
            "\n\n"
        )
        # Item 6/27: el frontend puede mostrar la cobertura explícitamente
        # sin tener que parsear el texto de la respuesta.
        yield (
            "data: " +
            json.dumps(
                {
                    "type": "coverage",
                    "data": {
                        "subquestions": len(batch.subquestions),
                        "covered": batch.covered_count,
                        "uncovered": batch.uncovered_count,
                        "coverage_score": batch.coverage_score,
                    },
                },
                ensure_ascii=False,
            ) +
            "\n\n"
        )
        for pedazo in chain.stream(question):
            yield (
                "data: " +
                json.dumps({"type": "chunk", "data": pedazo}, ensure_ascii=False) +
                "\n\n"
            )
        yield (
            "data: " +
            json.dumps({"type": "done"}, ensure_ascii=False) +
            "\n\n"
        )
        return

    resultado = ejecutar_pipeline(question)

    if not resultado.answerable or (not resultado.modo_resumen and not resultado.docs_evidencia):
        yield (
            "data: " +
            json.dumps({"type": "sources", "data": []}, ensure_ascii=False) +
            "\n\n"
        )
        yield (
            "data: " +
            json.dumps(
                {
                    "type": "chunk",
                    "data": "No encontré información suficiente en los documentos indexados para responder esa pregunta.",
                },
                ensure_ascii=False,
            ) +
            "\n\n"
        )
        yield (
            "data: " +
            json.dumps({"type": "done"}, ensure_ascii=False) +
            "\n\n"
        )
        return

    if resultado.modo_resumen:
        fuentes = [{"source": resultado.fuente_resumen, "area": resultado.ctx.area, "page": None}]
    else:
        fuentes = formatear_fuentes(resultado.docs_evidencia)

    yield (
        "data: " +
        json.dumps({"type": "sources", "data": fuentes}, ensure_ascii=False) +
        "\n\n"
    )
    yield (
        "data: " +
        json.dumps({"type": "area", "data": resultado.ctx.area}, ensure_ascii=False) +
        "\n\n"
    )

    fuente_iter = chain.stream(question) if resultado.modo_resumen else chain.stream(question, docs=resultado.docs_evidencia)
    for pedazo in fuente_iter:
        yield (
            "data: " +
            json.dumps({"type": "chunk", "data": pedazo}, ensure_ascii=False) +
            "\n\n"
        )

    yield (
        "data: " +
        json.dumps({"type": "done"}, ensure_ascii=False) +
        "\n\n"
    )


# ============================================================
# MAIN
# ============================================================


def main():
    print("\n======================================")
    print("      RAG-JUCHLM v11")
    print(" RAG HÍBRIDO MULTI-ÁREA + GROUNDING + EVIDENCE")
    print("======================================")

    if "--listar" in sys.argv:
        listar_indexados()
        return

    if "--areas" in sys.argv:
        print("\nÁreas disponibles:")
        for area in listar_areas_disponibles():
            print(f"  - {area}")
        return

    sincronizar_carpeta()

    if "--solo-ingesta" in sys.argv:
        return

    chain, retriever = construir_chain()

    print("\n======================================")
    print("Escribí tu pregunta (o 'salir' para terminar)")
    print("Tip: podés acotar por área, ej: 'en el área de rrhh, ¿...?'")
    print("======================================")

    while True:
        pregunta = input("\nPregunta: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break
        print()
        preguntar(chain, pregunta, retriever)


if __name__ == "__main__":
    main()