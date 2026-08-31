import json
import re
import hashlib
import sys
import time
import unicodedata

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
# RAG-JUCHLM v8
# ============================================================
#
# Arquitectura:
#
#                         PREGUNTA
#                            │
#                            ▼
#                  ┌────────────────────┐
#                  │ 1. INTENCIÓN       │
#                  │ documento          │
#                  │ página             │
#                  │ comparación        │
#                  │ tipo factual      │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ QUERY EXPANSION    │
#                  │ búsqueda dirigida  │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ 2. RETRIEVAL       │
#                  │ semántico          │
#                  │ + lexical          │
#                  │ + filtros duros    │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ 3. RANKING         │
#                  │ documental         │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ 4. AISLAMIENTO     │
#                  │ documento(s)       │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ 5. EVIDENCIA       │
#                  │ semántica          │
#                  │ lexical            │
#                  │ diversidad páginas │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ 6. LLM             │
#                  │ Qwen 3.5 4B        │
#                  └──────────┬─────────┘
#                             │
#                             ▼
#                  ┌────────────────────┐
#                  │ 7. CLAIM VALIDATOR  │
#                  │                    │
#                  │ números            │
#                  │ entidades          │
#                  │ fechas             │
#                  │ fuente             │
#                  │ inferencias        │
#                  │ contradicciones    │
#                  └──────────┬─────────┘
#                             │
#                       ¿RESPALDADO?
#                         /      \
#                       NO        SÍ
#                       │          │
#                       ▼          ▼
#                  reparación   entrega
#
# ============================================================


# ============================================================
# CONFIGURACIÓN
# ============================================================

PDF_FOLDER = Path("pdfs")

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "local-rag"

MANIFEST_PATH = Path(CHROMA_PATH) / "manifest.json"

OLLAMA_URL = "http://127.0.0.1:11434"

EMBEDDING_MODEL = "nomic-embed-text:latest"

LLM_MODEL = "qwen3.5:4b"

CURRENT_INDEX_SCHEMA_VERSION = 8


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 1500

CHUNK_OVERLAP = 100


# ============================================================
# RETRIEVAL
# ============================================================

RETRIEVER_K = 40

FINAL_CONTEXT_CHUNKS = 8

QUERY_EXPANSION_MAX = 6

MAX_RESULTS_PER_QUERY = 12

MAX_DOCUMENTS_COMPARISON = 2


# ============================================================
# LLM
# ============================================================

MAX_TOKENS_RESPUESTA = 4096

CONTEXTO_LLM = 8192

TEMPERATURE = 0


# ============================================================
# VALIDACIÓN
# ============================================================

VALIDATOR_ENABLED = True

REPAIR_UNSUPPORTED_CLAIMS = True

CLAIM_VALIDATOR_LLM_ENABLED = True

MAX_CLAIMS_TO_VERIFY = 12

MIN_CLAIM_TOKEN_OVERLAP = 0.22

MIN_CLAIM_TOKEN_OVERLAP_STRICT = 0.40


# ============================================================
# OCR
# ============================================================

OCR_ENABLED = True

SCANNED_THRESHOLD_CHARS_POR_PAGINA = 30

OCR_DPI = 200


# ============================================================
# DOCUMENTOS
# ============================================================

MIN_LEN_STEM_MATCH = 8

MIN_TOKEN_MATCHES = 2

MIN_TOKEN_PROPORTION = 0.35

LONGITUD_TOKEN_BONO = 7

BONO_LEXICAL_CHUNK = 0.15

BONO_LEXICAL_DOCUMENTO = 0.20


# ============================================================
# RANKING
# ============================================================

PESO_SEMANTICO = 0.70

PESO_LEXICO = 0.30


# ============================================================
# UTILIDADES
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def normalizar_texto(texto: str) -> str:
    texto = str(texto or "").lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        c
        for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(
        r"[^a-z0-9]+",
        " ",
        texto
    )

    return re.sub(
        r"\s+",
        " ",
        texto
    ).strip()


_STOPWORDS = {
    "el", "la", "los", "las",
    "un", "una", "unos", "unas",
    "de", "del", "al",
    "a", "en", "por", "para",
    "con", "sin",
    "que", "cual", "cuales",
    "como", "donde", "cuando",
    "quien", "quienes",
    "es", "son", "fue", "fueron",
    "se", "su", "sus",
    "sobre", "segun",
    "este", "esta", "estos", "estas",
    "mi", "me", "te",
    "y", "o", "u",
    "lo", "hay",
    "aparece", "dice", "dime", "indica",
    "informacion", "documento",
    "exacta", "exacto", "exactamente",
    "puedes", "podria", "podrias",
}


def tokens_significativos(texto: str) -> set[str]:
    return {
        token
        for token in normalizar_texto(texto).split()
        if len(token) >= 3
        and token not in _STOPWORDS
    }


def extraer_numeros(texto: str) -> set[str]:
    return set(
        re.findall(
            r"\b\d[\d\.\-/]*\d\b|\b\d+\b",
            str(texto or "")
        )
    )


def dividir_en_frases(texto: str) -> list[str]:
    texto = str(texto or "").strip()

    if not texto:
        return []

    partes = re.split(
        r"(?<=[\.\!\?])\s+|\n+",
        texto
    )

    return [
        p.strip(" -•\t")
        for p in partes
        if p.strip(" -•\t")
    ]


def limpiar_fuentes_generadas(
    texto: str
) -> str:

    return re.split(
        r"\n\s*(?:\*\*)?Fuentes(?:\*\*)?\s*:",
        str(texto or ""),
        flags=re.IGNORECASE
    )[0].strip()


# ============================================================
# MANIFEST
# ============================================================

def cargar_manifest() -> dict:

    if not MANIFEST_PATH.exists():
        return {}

    try:

        contenido = MANIFEST_PATH.read_text(
            encoding="utf-8"
        )

        if not contenido.strip():
            return {}

        return json.loads(
            contenido
        )

    except Exception as e:

        log(
            "ADVERTENCIA: no se pudo leer "
            f"manifest.json: {e}"
        )

        return {}


def guardar_manifest(
    manifest: dict
):

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def hash_archivo(
    path: Path
) -> str:

    stat = path.stat()

    firma = (
        f"{stat.st_size}-"
        f"{stat.st_mtime_ns}"
    )

    return hashlib.md5(
        firma.encode("utf-8")
    ).hexdigest()


# ============================================================
# PDF
# ============================================================

def cargar_pdf_texto(
    path: Path
) -> list[Document]:

    try:

        return PyPDFLoader(
            str(path),
            mode="page"
        ).load()

    except TypeError:

        return PyPDFLoader(
            str(path)
        ).load()


def es_probable_escaneado(
    documents: list[Document]
) -> bool:

    if not documents:
        return False

    total_chars = sum(
        len(
            d.page_content.strip()
        )
        for d in documents
    )

    promedio = (
        total_chars /
        len(documents)
    )

    return (
        promedio <
        SCANNED_THRESHOLD_CHARS_POR_PAGINA
    )


# ============================================================
# OCR DIRECTO
# ============================================================

def _ocr_pagina_con_rapidocr(
    pdf_path: Path,
    page_index: int
) -> str:

    try:

        import fitz
        import numpy as np
        from PIL import Image
        from io import BytesIO
        from rapidocr_onnxruntime import RapidOCR

    except ImportError as e:

        raise RuntimeError(
            "Faltan dependencias para OCR. "
            "Instala con:\n"
            "pip install rapidocr-onnxruntime pymupdf"
        ) from e

    pdf = fitz.open(
        str(pdf_path)
    )

    try:

        if (
            page_index < 0
            or
            page_index >= pdf.page_count
        ):
            return ""

        page = pdf.load_page(
            page_index
        )

        escala = OCR_DPI / 72.0

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(
                escala,
                escala
            ),
            alpha=False
        )

        imagen_bytes = pixmap.tobytes(
            "png"
        )

        imagen = Image.open(
            BytesIO(imagen_bytes)
        ).convert("RGB")

        imagen_array = np.array(
            imagen
        )

        engine = RapidOCR()

        resultado = engine(
            imagen_array
        )

        if isinstance(
            resultado,
            tuple
        ):
            resultados = resultado[0]
        else:
            resultados = resultado

        if not resultados:
            return ""

        textos = []

        for item in resultados:

            if not item:
                continue

            if (
                isinstance(
                    item,
                    (list, tuple)
                )
                and
                len(item) >= 2
            ):

                texto = str(
                    item[1] or ""
                ).strip()

                if texto:
                    textos.append(
                        texto
                    )

        return "\n".join(
            textos
        ).strip()

    finally:

        pdf.close()


def cargar_pdf(
    path: Path
) -> tuple[list[Document], bool]:

    docs = cargar_pdf_texto(
        path
    )

    if not docs:
        return [], False

    paginas_ocr = []

    for indice, doc in enumerate(
        docs
    ):

        chars = len(
            doc.page_content.strip()
        )

        if (
            OCR_ENABLED
            and
            chars <
            SCANNED_THRESHOLD_CHARS_POR_PAGINA
        ):

            paginas_ocr.append(
                indice
            )

    if not paginas_ocr:

        return docs, False

    log(
        f"    -> {len(paginas_ocr)} "
        "página(s) requieren OCR."
    )

    fue_ocr = False

    for page_index in paginas_ocr:

        try:

            texto_ocr = (
                _ocr_pagina_con_rapidocr(
                    path,
                    page_index
                )
            )

            texto_original = (
                docs[page_index].page_content
                or ""
            )

            if len(
                texto_ocr
            ) > len(
                texto_original.strip()
            ):

                docs[
                    page_index
                ].page_content = (
                    texto_ocr
                )

                fue_ocr = True

                log(
                    f"    -> OCR página "
                    f"{page_index + 1}: "
                    f"{len(texto_ocr)} caracteres."
                )

        except Exception as e:

            log(
                f"    -> ERROR OCR página "
                f"{page_index + 1}: {e}"
            )

    return docs, fue_ocr


# ============================================================
# EMBEDDINGS / CHROMA
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
)


# ============================================================
# INGESTA
# ============================================================

def sincronizar_carpeta():

    if not PDF_FOLDER.exists():

        log(
            f"\nERROR: no existe la carpeta "
            f"'{PDF_FOLDER.absolute()}'"
        )

        raise SystemExit(1)

    pdfs_en_disco = sorted(
        PDF_FOLDER.glob("*.pdf")
    )

    manifest = cargar_manifest()

    nombres_en_disco = {
        p.name
        for p in pdfs_en_disco
    }

    # ========================================================
    # ELIMINAR AUSENTES
    # ========================================================

    eliminados = [

        nombre
        for nombre in manifest
        if nombre not in nombres_en_disco
    ]

    for nombre in eliminados:

        log(
            f"\nEliminando de Chroma: "
            f"'{nombre}'"
        )

        try:

            vector_db.delete(
                where={
                    "source":
                        nombre
                }
            )

        except Exception as e:

            log(
                f"    -> Advertencia: {e}"
            )

        del manifest[
            nombre
        ]

    if not pdfs_en_disco:

        guardar_manifest(
            manifest
        )

        log(
            f"\nNo hay PDFs en "
            f"'{PDF_FOLDER.absolute()}'."
        )

        return

    # ========================================================
    # DETECTAR NUEVOS / MODIFICADOS /
    # CAMBIO DE ESQUEMA
    # ========================================================

    nuevos_o_cambiados = []

    for path in pdfs_en_disco:

        info = manifest.get(
            path.name,
            {}
        )

        firma_actual = hash_archivo(
            path
        )

        firma_previa = info.get(
            "firma"
        )

        version_previa = info.get(
            "index_schema_version",
            0
        )

        necesita_reindexar = (

            firma_actual != firma_previa

            or

            version_previa
            !=
            CURRENT_INDEX_SCHEMA_VERSION
        )

        if necesita_reindexar:

            nuevos_o_cambiados.append(
                path
            )

    if not nuevos_o_cambiados and not eliminados:

        log(
            f"\nBase vectorial ya actualizada. "
            f"{len(pdfs_en_disco)} PDFs indexados, "
            "sin cambios."
        )

        return

    log(
        f"\n{len(nuevos_o_cambiados)} "
        "PDF(s) serán procesados."
    )

    # ========================================================
    # PROCESAR
    # ========================================================

    for i, path in enumerate(
        nuevos_o_cambiados,
        1
    ):

        log(
            f"\n[{i}/{len(nuevos_o_cambiados)}] "
            f"{path.name}"
        )

        inicio = time.time()

        # ----------------------------------------------------
        # Eliminar versión anterior
        # ----------------------------------------------------

        try:

            vector_db.delete(
                where={
                    "source":
                        path.name
                }
            )

        except Exception as e:

            log(
                f"    -> Advertencia eliminando "
                f"versión anterior: {e}"
            )

        # ----------------------------------------------------
        # Cargar PDF
        # ----------------------------------------------------

        try:

            docs, fue_ocr = cargar_pdf(
                path
            )

        except Exception as e:

            log(
                f"    -> ERROR procesando "
                f"'{path.name}': {e}"
            )

            continue

        if not docs:

            log(
                f"    -> '{path.name}' "
                "no contiene páginas."
            )

            continue

        # ----------------------------------------------------
        # Metadata por página
        # ----------------------------------------------------

        for page_index, doc in enumerate(
            docs
        ):

            doc.metadata[
                "source"
            ] = path.name

            doc.metadata[
                "document_id"
            ] = path.stem

            doc.metadata[
                "page"
            ] = page_index

            doc.metadata[
                "page_number"
            ] = page_index + 1

            doc.metadata[
                "ocr"
            ] = fue_ocr

            doc.metadata[
                "index_schema_version"
            ] = CURRENT_INDEX_SCHEMA_VERSION

        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        chunks = splitter.split_documents(
            docs
        )

        if not chunks:

            log(
                f"    -> '{path.name}' "
                "no generó chunks."
            )

            continue

        # ----------------------------------------------------
        # Metadata por chunk
        # ----------------------------------------------------

        for chunk_index, chunk in enumerate(
            chunks
        ):

            chunk.metadata[
                "chunk_index"
            ] = chunk_index

            chunk.metadata[
                "index_schema_version"
            ] = CURRENT_INDEX_SCHEMA_VERSION

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        ids = []

        for index, chunk in enumerate(
            chunks
        ):

            page = chunk.metadata.get(
                "page",
                0
            )

            ids.append(
                f"{path.name}::"
                f"page-{page}::"
                f"chunk-{index}"
            )

        # ----------------------------------------------------
        # Indexar
        # ----------------------------------------------------

        try:

            vector_db.add_documents(
                documents=chunks,
                ids=ids
            )

        except Exception as e:

            log(
                f"    -> ERROR indexando "
                f"'{path.name}': {e}"
            )

            continue

        # ----------------------------------------------------
        # Manifest
        # ----------------------------------------------------

        manifest[
            path.name
        ] = {

            "firma":
                hash_archivo(path),

            "chunks":
                len(chunks),

            "pages":
                len(docs),

            "ocr":
                fue_ocr,

            "index_schema_version":
                CURRENT_INDEX_SCHEMA_VERSION,
        }

        log(
            f"    -> {len(chunks)} chunks | "
            f"{len(docs)} páginas | "
            f"{'OCR' if fue_ocr else 'texto'} | "
            f"{time.time() - inicio:.2f}s"
        )

    guardar_manifest(
        manifest
    )

    log(
        "\nSincronización completa. "
        f"Total PDFs registrados: "
        f"{len(manifest)}"
    )


# ============================================================
# LISTAR
# ============================================================

def listar_indexados():

    manifest = cargar_manifest()

    if not manifest:

        print(
            "\nNo hay PDFs indexados todavía."
        )

        return

    print(
        f"\n{len(manifest)} PDF(s) registrados:\n"
    )

    for nombre, info in manifest.items():

        tipo = (
            "OCR"
            if info.get("ocr")
            else "texto"
        )

        print(
            f"  - {nombre} | "
            f"{info.get('chunks', 0)} chunks | "
            f"{info.get('pages', '?')} páginas | "
            f"{tipo} | "
            f"schema {info.get('index_schema_version', '?')}"
        )


# ============================================================
# QUERY CONTEXT
# ============================================================

@dataclass
class QueryContext:

    question: str

    q_tokens: set[str] = field(
        default_factory=set
    )

    fuente_explicita: Optional[str] = None

    fuentes_explicitas: list[str] = field(
        default_factory=list
    )

    pagina_explicita: Optional[int] = None

    ultima_pagina_solicitada: bool = False

    es_comparacion: bool = False

    tipo_consulta: str = "general"

    query_variants: list[str] = field(
        default_factory=list
    )


@dataclass
class PipelineResult:

    ctx: QueryContext

    docs_candidatos: list[Document]

    ranking_documentos: list[
        tuple[
            float,
            str,
            list[Document]
        ]
    ]

    docs_aislados: list[Document]

    docs_evidencia: list[Document]

    scores_evidencia: list[
        tuple[
            float,
            Document
        ]
    ] = field(
        default_factory=list
    )


# ============================================================
# DETECCIÓN DE PÁGINA
# ============================================================

def _detectar_pagina(
    question: str
) -> tuple[
    Optional[int],
    bool
]:

    q = normalizar_texto(
        question
    )

    # Correcciones comunes
    correcciones = {

        "pagaina": "pagina",

        "pagnina": "pagina",

        "pagnia": "pagina",

        "paginaa": "pagina",

        "pagna": "pagina",

        "pajina": "pagina",
    }

    palabras = q.split()

    q = " ".join(
        correcciones.get(
            palabra,
            palabra
        )
        for palabra in palabras
    )

    # --------------------------------------------------------
    # Última página
    # --------------------------------------------------------

    if re.search(
        r"\bultima pagina\b",
        q
    ):

        return None, True

    # --------------------------------------------------------
    # Ordinales
    # --------------------------------------------------------

    ordinales = {

        "primera pagina": 0,
        "segunda pagina": 1,
        "tercera pagina": 2,
        "cuarta pagina": 3,
        "quinta pagina": 4,
        "sexta pagina": 5,
        "septima pagina": 6,
        "octava pagina": 7,
        "novena pagina": 8,
        "decima pagina": 9,
        "undecima pagina": 10,
        "duodecima pagina": 11,
    }

    for expresion, pagina in ordinales.items():

        if re.search(
            rf"\b{re.escape(expresion)}\b",
            q
        ):

            return pagina, False

    # --------------------------------------------------------
    # Números escritos
    # --------------------------------------------------------

    numeros_escritos = {

        "uno": 0,
        "dos": 1,
        "tres": 2,
        "cuatro": 3,
        "cinco": 4,
        "seis": 5,
        "siete": 6,
        "ocho": 7,
        "nueve": 8,
        "diez": 9,
        "once": 10,
        "doce": 11,
        "trece": 12,
        "catorce": 13,
        "quince": 14,
        "dieciseis": 15,
        "diecisiete": 16,
        "dieciocho": 17,
        "diecinueve": 18,
        "veinte": 19,
    }

    for palabra, pagina in (
        numeros_escritos.items()
    ):

        if re.search(
            rf"\bpagina\s+{palabra}\b",
            q
        ):

            return pagina, False

    # --------------------------------------------------------
    # Numérica
    # --------------------------------------------------------

    patrones = [

        r"\bpagina\s+(?:numero\s+)?(\d+)\b",

        r"\bpag\s+(\d+)\b",

        r"\bpag\.?\s*(\d+)\b",

    ]

    for patron in patrones:

        match = re.search(
            patron,
            q
        )

        if match:

            numero = int(
                match.group(1)
            )

            if numero >= 1:

                return (
                    numero - 1,
                    False
                )

    return None, False


# ============================================================
# COMPARACIÓN
# ============================================================

def _detectar_comparacion(
    question: str
) -> bool:

    q = normalizar_texto(
        question
    )

    patrones = [

        r"\bcompara\b",
        r"\bcomparar\b",
        r"\bcomparacion\b",
        r"\bcomparando\b",
        r"\bdiferencia entre\b",
        r"\bdiferencias entre\b",
        r"\bambos documentos\b",
        r"\blos dos documentos\b",
        r"\bentre los documentos\b",
        r"\bcontrasta\b",
        r"\bcontrastar\b",
        r"\brelacion entre\b",
        r"\brelaciona\b",
        r"\bsimilitudes entre\b",
        r"\bsimilaridades entre\b",
        r"\bque tienen en comun\b",
        r"\buno frente al otro\b",
        r"\bfrente al otro\b",
    ]

    return any(
        re.search(
            patron,
            q
        )
        for patron in patrones
    )


# ============================================================
# TIPO DE CONSULTA
# ============================================================

def _detectar_tipo_consulta(
    question: str,
    es_comparacion: bool,
    fuente_explicita: Optional[str],
    pagina_explicita: Optional[int],
) -> str:

    q = normalizar_texto(
        question
    )

    if es_comparacion:
        return "comparacion"

    if pagina_explicita is not None:
        return "pagina"

    if fuente_explicita:
        return "documento"

    patrones = [

        r"\bquien\b",
        r"\bquienes\b",
        r"\bcual\b",
        r"\bcuales\b",
        r"\bcuando\b",
        r"\bque fecha\b",
        r"\ben que fecha\b",
        r"\bfecha\b",
        r"\bcuanto\b",
        r"\bcuantos\b",
        r"\bnumero de\b",
        r"\bnombre de\b",
        r"\bdonde\b",
        r"\bque monto\b",
        r"\bque cargo\b",
        r"\bque articulo\b",
        r"\bque resolucion\b",
        r"\bque expediente\b",
        r"\bdni\b",
        r"\bjuez\b",
        r"\bjueza\b",
        r"\bmagistrad",
        r"\bacusado\b",
        r"\bagraviado\b",
        r"\bdelito\b",
        r"\bsentencia\b",
        r"\bhechos\b",
    ]

    if any(
        re.search(
            patron,
            q
        )
        for patron in patrones
    ):

        return "factual"

    return "general"


# ============================================================
# FUENTES
# ============================================================

def _obtener_fuentes_explicitas(
    question: str,
    q_norm: str,
    q_tokens: set[str],
) -> list[str]:

    pdfs = sorted(
        PDF_FOLDER.glob("*.pdf")
    )

    if not pdfs:
        return []

    encontrados = []

    # ========================================================
    # NOMBRE COMPLETO
    # ========================================================

    for path in pdfs:

        nombre_norm = normalizar_texto(
            path.name
        )

        if (
            nombre_norm
            and
            nombre_norm in q_norm
        ):

            encontrados.append(
                path.name
            )

    # ========================================================
    # STEM
    # ========================================================

    for path in pdfs:

        if path.name in encontrados:
            continue

        stem_norm = normalizar_texto(
            path.stem
        )

        if (
            len(stem_norm)
            >= MIN_LEN_STEM_MATCH
            and
            stem_norm in q_norm
        ):

            encontrados.append(
                path.name
            )

    # ========================================================
    # TOKENS
    # ========================================================

    candidatos = []

    for path in pdfs:

        if path.name in encontrados:
            continue

        nombre_tokens = tokens_significativos(
            path.stem
        )

        if not nombre_tokens:
            continue

        coincidencias = (
            q_tokens &
            nombre_tokens
        )

        if not coincidencias:
            continue

        proporcion = (
            len(coincidencias)
            /
            max(
                1,
                len(nombre_tokens)
            )
        )

        tokens_largos = sum(
            1
            for token
            in coincidencias
            if len(token)
            >= LONGITUD_TOKEN_BONO
        )

        score = (

            len(coincidencias)

            +

            proporcion

            +

            tokens_largos * 0.5
        )

        if (
            len(coincidencias)
            >= MIN_TOKEN_MATCHES
            and
            proporcion
            >= MIN_TOKEN_PROPORTION
        ):

            candidatos.append(
                (
                    score,
                    proporcion,
                    len(coincidencias),
                    path.name
                )
            )

    candidatos.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
            x[3]
        ),
        reverse=True
    )

    for _, _, _, nombre in candidatos:

        if nombre not in encontrados:

            encontrados.append(
                nombre
            )

    # ========================================================
    # EXPEDIENTE GENÉRICO
    # ========================================================

    if not encontrados:

        if "expediente" in q_norm:

            candidatos_exp = [

                path.name
                for path in pdfs
                if "expediente"
                in normalizar_texto(
                    path.stem
                )
            ]

            if len(
                candidatos_exp
            ) == 1:

                encontrados.append(
                    candidatos_exp[0]
                )

    return encontrados


# ============================================================
# QUERY EXPANSION
# ============================================================

def construir_query_variants(
    ctx: QueryContext
) -> list[str]:

    q = ctx.question.strip()

    if not q:
        return []

    variantes = []

    def agregar(texto: str):

        texto = texto.strip()

        if (
            texto
            and
            texto not in variantes
        ):

            variantes.append(
                texto
            )

    # Consulta original
    agregar(q)

    q_norm = normalizar_texto(
        q
    )

    tipo = ctx.tipo_consulta

    # ========================================================
    # CONSULTAS FACTUALES
    # ========================================================

    if tipo == "factual":

        if "dni" in q_norm:

            agregar(
                "DNI documento identidad acusado"
            )

            agregar(
                "documento nacional de identidad acusado"
            )

        if "acusado" in q_norm:

            agregar(
                "nombre del acusado imputado procesado"
            )

            agregar(
                "acusado nombre completo"
            )

        if (
            "agraviado" in q_norm
            or
            "agraviiado" in q_norm
        ):

            agregar(
                "agraviado agraviados Estado Policía Nacional"
            )

        if (
            "juez" in q_norm
            or
            "jueza" in q_norm
            or
            "magistrad" in q_norm
        ):

            agregar(
                "juez jueza magistrado magistrada"
            )

            agregar(
                "magistrados integrantes juzgado"
            )

            agregar(
                "sentencia juez magistrado"
            )

        if "fecha" in q_norm:

            agregar(
                "fecha sentencia resolución"
            )

            agregar(
                "fecha de emisión sentencia"
            )

            agregar(
                "fecha sentencia emitida"
            )

        if "hechos" in q_norm:

            agregar(
                "fecha de los hechos ocurridos"
            )

            agregar(
                "hechos imputados fecha lugar"
            )

        if "monto" in q_norm:

            agregar(
                "monto reparación civil suma soles"
            )

        if "delito" in q_norm:

            agregar(
                "delito imputado calificación jurídica"
            )

        if "expediente" in q_norm:

            agregar(
                "número de expediente"
            )

            agregar(
                "expediente número"
            )

    # ========================================================
    # PÁGINAS
    # ========================================================

    if (
        ctx.pagina_explicita
        is not None
    ):

        pagina_visible = (
            ctx.pagina_explicita + 1
        )

        agregar(
            f"pagina {pagina_visible} "
            f"{q}"
        )

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    if ctx.es_comparacion:

        agregar(
            f"comparación {q}"
        )

    # ========================================================
    # LÍMITE
    # ========================================================

    return variantes[
        :QUERY_EXPANSION_MAX
    ]


# ============================================================
# ÚLTIMA PÁGINA
# ============================================================

def obtener_numero_paginas(
    fuente: str
) -> Optional[int]:

    path = PDF_FOLDER / fuente

    if not path.exists():
        return None

    try:

        import fitz

        pdf = fitz.open(
            str(path)
        )

        try:

            return pdf.page_count

        finally:

            pdf.close()

    except Exception:

        try:

            docs = cargar_pdf_texto(
                path
            )

            return len(
                docs
            )

        except Exception:

            return None


def resolver_ultima_pagina(
    ctx: QueryContext
):

    if not ctx.ultima_pagina_solicitada:
        return

    if not ctx.fuente_explicita:

        return

    paginas = obtener_numero_paginas(
        ctx.fuente_explicita
    )

    if not paginas:
        return

    ctx.pagina_explicita = (
        paginas - 1
    )

    log(
        f"  Última página resuelta: "
        f"{paginas}"
    )


# ============================================================
# DETECTAR INTENCIÓN
# ============================================================

def detectar_intencion(
    question: str
) -> QueryContext:

    question = str(
        question or ""
    ).strip()

    if not question:

        return QueryContext(
            question=""
        )

    q_norm = normalizar_texto(
        question
    )

    q_tokens = tokens_significativos(
        question
    )

    fuentes = (
        _obtener_fuentes_explicitas(
            question,
            q_norm,
            q_tokens
        )
    )

    fuente_principal = (
        fuentes[0]
        if fuentes
        else None
    )

    pagina, ultima = (
        _detectar_pagina(
            question
        )
    )

    comparacion = (
        _detectar_comparacion(
            question
        )
    )

    tipo = (
        _detectar_tipo_consulta(
            question,
            comparacion,
            fuente_principal,
            pagina
        )
    )

    ctx = QueryContext(

        question=question,

        q_tokens=q_tokens,

        fuente_explicita=(
            fuente_principal
        ),

        fuentes_explicitas=fuentes,

        pagina_explicita=pagina,

        ultima_pagina_solicitada=ultima,

        es_comparacion=comparacion,

        tipo_consulta=tipo,
    )

    resolver_ultima_pagina(
        ctx
    )

    ctx.query_variants = (
        construir_query_variants(
            ctx
        )
    )

    return ctx


# ============================================================
# RETRIEVAL HELPERS
# ============================================================

def _normalizar_resultados(
    resultados
) -> list[Document]:

    if not resultados:
        return []

    documentos = []

    distancias = []

    for doc, distancia in resultados:

        try:

            d = float(
                distancia
            )

        except (
            ValueError,
            TypeError
        ):

            d = 0.0

        distancias.append(
            d
        )

        documentos.append(
            doc
        )

    if not distancias:
        return []

    minimo = min(
        distancias
    )

    maximo = max(
        distancias
    )

    for doc, distancia in zip(
        documentos,
        distancias
    ):

        if maximo == minimo:

            score = 1.0

        else:

            score = (
                1.0
                -
                (
                    (distancia - minimo)
                    /
                    (maximo - minimo)
                )
            )

        doc.metadata[
            "_retrieval_distance"
        ] = distancia

        doc.metadata[
            "_retrieval_score"
        ] = score

    return documentos


def _deduplicar_docs(
    docs: list[Document]
) -> list[Document]:

    salida = []

    vistos = set()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            ""
        )

        page = doc.metadata.get(
            "page"
        )

        chunk = doc.metadata.get(
            "chunk_index"
        )

        contenido = (
            doc.page_content
            or ""
        )

        fingerprint = hashlib.md5(
            contenido.encode(
                "utf-8",
                errors="ignore"
            )
        ).hexdigest()

        clave = (
            source,
            page,
            chunk,
            fingerprint
        )

        if clave in vistos:
            continue

        vistos.add(
            clave
        )

        salida.append(
            doc
        )

    return salida


def _filtrar_fuente_y_pagina(
    docs: list[Document],
    ctx: QueryContext
) -> list[Document]:

    salida = docs

    if ctx.fuentes_explicitas:

        fuentes = set(
            ctx.fuentes_explicitas
        )

        salida = [
            doc
            for doc in salida
            if doc.metadata.get(
                "source"
            ) in fuentes
        ]

    elif ctx.fuente_explicita:

        salida = [
            doc
            for doc in salida
            if doc.metadata.get(
                "source"
            )
            ==
            ctx.fuente_explicita
        ]

    if ctx.pagina_explicita is not None:

        salida = [
            doc
            for doc in salida
            if doc.metadata.get(
                "page"
            )
            ==
            ctx.pagina_explicita
        ]

    return salida


# ============================================================
# PASO 2
# RETRIEVAL HÍBRIDO
# ============================================================

def recuperar_candidatos(
    ctx: QueryContext
) -> list[Document]:

    variants = ctx.query_variants or [
        ctx.question
    ]

    acumulados = []

    # ========================================================
    # CASO EXACTO: DOCUMENTO + PÁGINA
    # ========================================================

    if (
        ctx.fuente_explicita
        and
        ctx.pagina_explicita is not None
    ):

        try:

            data = vector_db.get(

                where={
                    "$and": [
                        {
                            "source":
                                ctx.fuente_explicita
                        },
                        {
                            "page":
                                ctx.pagina_explicita
                        }
                    ]
                },

                include=[
                    "documents",
                    "metadatas"
                ]
            )

            documentos = (
                data.get(
                    "documents",
                    []
                )
            )

            metadatas = (
                data.get(
                    "metadatas",
                    []
                )
            )

            for contenido, metadata in zip(
                documentos,
                metadatas
            ):

                metadata = dict(
                    metadata or {}
                )

                metadata[
                    "_retrieval_score"
                ] = 1.0

                metadata[
                    "_retrieval_distance"
                ] = 0.0

                acumulados.append(
                    Document(
                        page_content=(
                            contenido or ""
                        ),
                        metadata=metadata
                    )
                )

            return _deduplicar_docs(
                acumulados
            )

        except Exception as e:

            log(
                "ADVERTENCIA retrieval "
                f"exacto: {e}"
            )

            return []

    # ========================================================
    # VARIAS FUENTES EXPLÍCITAS EN COMPARACIÓN
    # ========================================================

    fuentes_comparacion = (
        ctx.fuentes_explicitas
        if (
            ctx.es_comparacion
            and
            len(
                ctx.fuentes_explicitas
            ) > 1
        )
        else []
    )

    if fuentes_comparacion:

        for fuente in fuentes_comparacion:

            for variant in variants:

                try:

                    resultados = (
                        vector_db
                        .similarity_search_with_score(
                            variant,
                            k=MAX_RESULTS_PER_QUERY,
                            filter={
                                "source":
                                    fuente
                            }
                        )
                    )

                    acumulados.extend(
                        _normalizar_resultados(
                            resultados
                        )
                    )

                except Exception as e:

                    log(
                        f"ADVERTENCIA retrieval "
                        f"'{fuente}': {e}"
                    )

        if ctx.pagina_explicita is not None:

            acumulados = [
                doc
                for doc in acumulados
                if doc.metadata.get(
                    "page"
                )
                ==
                ctx.pagina_explicita
            ]

        return _deduplicar_docs(
            acumulados
        )

    # ========================================================
    # RETRIEVAL GENERAL / DOCUMENTO
    # ========================================================

    filtros = None

    if ctx.fuente_explicita:

        filtros = {
            "source":
                ctx.fuente_explicita
        }

    elif ctx.pagina_explicita is not None:

        filtros = {
            "page":
                ctx.pagina_explicita
        }

    for variant in variants:

        try:

            if filtros:

                resultados = (
                    vector_db
                    .similarity_search_with_score(
                        variant,
                        k=MAX_RESULTS_PER_QUERY,
                        filter=filtros
                    )
                )

            else:

                resultados = (
                    vector_db
                    .similarity_search_with_score(
                        variant,
                        k=MAX_RESULTS_PER_QUERY
                    )
                )

            acumulados.extend(
                _normalizar_resultados(
                    resultados
                )
            )

        except Exception as e:

            log(
                f"ADVERTENCIA retrieval "
                f"'{variant}': {e}"
            )

    # ========================================================
    # Restricciones duras finales
    # ========================================================

    acumulados = (
        _filtrar_fuente_y_pagina(
            acumulados,
            ctx
        )
    )

    return _deduplicar_docs(
        acumulados
    )


# ============================================================
# PASO 3
# RANKING DOCUMENTAL
# ============================================================

def _puntuar_documento(
    ctx: QueryContext,
    docs: list[Document]
) -> float:

    if not docs:
        return 0.0

    if not ctx.q_tokens:
        return 0.0

    scores = []

    for doc in docs:

        tokens_texto = (
            tokens_significativos(
                doc.page_content
            )
        )

        if not tokens_texto:
            continue

        interseccion = (
            ctx.q_tokens &
            tokens_texto
        )

        cobertura = (
            len(interseccion)
            /
            max(
                1,
                len(ctx.q_tokens)
            )
        )

        bono = sum(
            BONO_LEXICAL_DOCUMENTO
            for token in interseccion
            if len(token)
            >= LONGITUD_TOKEN_BONO
        )

        semantico = float(
            doc.metadata.get(
                "_retrieval_score",
                0.0
            )
            or 0.0
        )

        score = (

            0.50 *
            cobertura

            +

            0.20 *
            bono

            +

            0.30 *
            semantico
        )

        scores.append(
            score
        )

    if not scores:
        return 0.0

    scores.sort(
        reverse=True
    )

    resultado = scores[0]

    if len(scores) >= 2:

        resultado += (
            0.35 *
            scores[1]
        )

    if len(scores) >= 3:

        resultado += (
            0.20 *
            scores[2]
        )

    return resultado


def identificar_documento_relevante(
    ctx: QueryContext,
    candidatos: list[Document]
):

    if not candidatos:
        return []

    por_fuente = defaultdict(
        list
    )

    for doc in candidatos:

        source = doc.metadata.get(
            "source"
        )

        if source:

            por_fuente[
                source
            ].append(
                doc
            )

    ranking = [

        (
            _puntuar_documento(
                ctx,
                docs
            ),

            source,

            docs
        )

        for source, docs
        in por_fuente.items()
    ]

    ranking.sort(
        key=lambda x: x[0],
        reverse=True
    )

    log(
        "\n--- Paso 3: ranking de documentos ---"
    )

    for score, source, docs in ranking:

        log(
            f"  {score:.4f} -> "
            f"{source} "
            f"({len(docs)} chunks)"
        )

    return ranking


# ============================================================
# PASO 4
# AISLAMIENTO
# ============================================================

def aislar_documento(
    ctx: QueryContext,
    ranking_documentos
) -> list[Document]:

    if not ranking_documentos:
        return []

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    if ctx.es_comparacion:

        if ctx.fuentes_explicitas:

            permitidos = set(
                ctx.fuentes_explicitas
            )

            docs = []

            for _, source, source_docs in (
                ranking_documentos
            ):

                if source in permitidos:

                    docs.extend(
                        source_docs
                    )

            return docs

        docs = []

        for _, _, source_docs in (
            ranking_documentos[
                :MAX_DOCUMENTS_COMPARISON
            ]
        ):

            docs.extend(
                source_docs
            )

        return docs

    # ========================================================
    # DOCUMENTO EXPLÍCITO
    # ========================================================

    if ctx.fuente_explicita:

        for _, source, docs in (
            ranking_documentos
        ):

            if source == ctx.fuente_explicita:

                return docs

        return []

    # ========================================================
    # DOMINANTE
    # ========================================================

    _, _, docs = (
        ranking_documentos[0]
    )

    return docs


# ============================================================
# PASO 5
# SCORE LEXICAL
# ============================================================

def _puntuacion_lexica_chunk(
    ctx: QueryContext,
    doc: Document
) -> float:

    if not ctx.q_tokens:
        return 0.0

    texto_tokens = (
        tokens_significativos(
            doc.page_content
        )
    )

    if not texto_tokens:
        return 0.0

    interseccion = (
        ctx.q_tokens &
        texto_tokens
    )

    cobertura = (
        len(interseccion)
        /
        max(
            1,
            len(ctx.q_tokens)
        )
    )

    bono = sum(
        BONO_LEXICAL_CHUNK
        for token in interseccion
        if len(token)
        >= LONGITUD_TOKEN_BONO
    )

    return (
        cobertura
        +
        bono
    )


def _puntuacion_semantica_chunk(
    doc: Document
) -> float:

    try:

        return float(
            doc.metadata.get(
                "_retrieval_score",
                0.0
            )
            or 0.0
        )

    except (
        ValueError,
        TypeError
    ):

        return 0.0


# ============================================================
# DIVERSIDAD Y PRIORIZACIÓN
# ============================================================

def _bonus_tipo_consulta(
    ctx: QueryContext,
    doc: Document
) -> float:

    bonus = 0.0

    texto = normalizar_texto(
        doc.page_content
    )

    # Consultas dirigidas
    if ctx.tipo_consulta == "factual":

        patrones = [

            ("dni", [
                "dni",
                "documento nacional",
                "identidad"
            ]),

            ("juez", [
                "juez",
                "jueza",
                "magistrado",
                "magistrada"
            ]),

            ("fecha", [
                "fecha",
                "sentencia",
                "resolucion"
            ]),

            ("acusado", [
                "acusado",
                "imputado",
                "procesado"
            ]),

            ("agraviado", [
                "agraviado",
                "estado",
                "policia nacional"
            ]),

            ("monto", [
                "monto",
                "soles",
                "reparacion civil"
            ]),
        ]

        q = normalizar_texto(
            ctx.question
        )

        for palabra_clave, palabras in patrones:

            if palabra_clave in q:

                if any(
                    palabra in texto
                    for palabra in palabras
                ):

                    bonus += 0.12

    return bonus


# ============================================================
# PASO 5
# EVIDENCIA
# ============================================================

def seleccionar_evidencia(
    ctx: QueryContext,
    docs_aislados: list[Document]
) -> tuple[
    list[Document],
    list[tuple[float, Document]]
]:

    if not docs_aislados:
        return [], []

    docs = list(
        docs_aislados
    )

    # Página = restricción dura
    if ctx.pagina_explicita is not None:

        docs = [
            doc
            for doc in docs
            if doc.metadata.get(
                "page"
            )
            ==
            ctx.pagina_explicita
        ]

        if not docs:

            log(
                "    -> No existe evidencia "
                f"en página "
                f"{ctx.pagina_explicita + 1}."
            )

            return [], []

    candidatos = []

    for doc in docs:

        lexical = (
            _puntuacion_lexica_chunk(
                ctx,
                doc
            )
        )

        semantico = (
            _puntuacion_semantica_chunk(
                doc
            )
        )

        bonus = (
            _bonus_tipo_consulta(
                ctx,
                doc
            )
        )

        score = (

            PESO_SEMANTICO *
            semantico

            +

            PESO_LEXICO *
            lexical

            +

            bonus
        )

        candidatos.append(
            (
                score,
                doc
            )
        )

    candidatos.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # ========================================================
    # Selección final
    # ========================================================

    evidencia = []

    vistos = set()

    for score, doc in candidatos:

        source = doc.metadata.get(
            "source",
            ""
        )

        page = doc.metadata.get(
            "page"
        )

        chunk_index = doc.metadata.get(
            "chunk_index"
        )

        clave = (
            source,
            page,
            chunk_index
        )

        if clave in vistos:
            continue

        vistos.add(
            clave
        )

        evidencia.append(
            doc
        )

        if (
            len(evidencia)
            >= FINAL_CONTEXT_CHUNKS
        ):

            break

    scores_finales = []

    for doc in evidencia:

        lexical = (
            _puntuacion_lexica_chunk(
                ctx,
                doc
            )
        )

        semantico = (
            _puntuacion_semantica_chunk(
                doc
            )
        )

        bonus = (
            _bonus_tipo_consulta(
                ctx,
                doc
            )
        )

        score = (

            PESO_SEMANTICO *
            semantico

            +

            PESO_LEXICO *
            lexical

            +

            bonus
        )

        scores_finales.append(
            (
                score,
                doc
            )
        )

    return (
        evidencia,
        scores_finales
    )


# ============================================================
# PIPELINE
# ============================================================

def ejecutar_pipeline(
    question: str
) -> PipelineResult:

    ctx = detectar_intencion(
        question
    )

    docs_candidatos = (
        recuperar_candidatos(
            ctx
        )
    )

    ranking_documentos = (
        identificar_documento_relevante(
            ctx,
            docs_candidatos
        )
    )

    docs_aislados = (
        aislar_documento(
            ctx,
            ranking_documentos
        )
    )

    docs_evidencia, scores_evidencia = (
        seleccionar_evidencia(
            ctx,
            docs_aislados
        )
    )

    return PipelineResult(

        ctx=ctx,

        docs_candidatos=
            docs_candidatos,

        ranking_documentos=
            ranking_documentos,

        docs_aislados=
            docs_aislados,

        docs_evidencia=
            docs_evidencia,

        scores_evidencia=
            scores_evidencia,
    )


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
Eres el asistente documental local de JUCHLM.

Tu única fuente de verdad es el CONTEXTO RECUPERADO.

REGLAS OBLIGATORIAS:

1. Utiliza exclusivamente información explícitamente presente
   en el contexto.

2. No utilices conocimiento externo.

3. No inventes ni completes información faltante.

4. No conviertas una fecha en otra fecha.

5. No conviertas un número en otro número.

6. No conviertas un cargo en otro cargo.

7. No conviertas una persona en otra persona.

8. No infieras relaciones que el documento no afirma.

9. No utilices conocimiento general para completar el contexto.

10. Si una respuesta requiere una inferencia, indícalo claramente
    y NO presentes la inferencia como un hecho documental.

11. Evita expresiones como:
    "por lo tanto",
    "se puede inferir",
    "esto demuestra",
    "esto significa",
    "probablemente",
    cuando estén introduciendo un dato que no aparece
    explícitamente en el contexto.

12. Si el usuario solicita una página concreta, responde
    solamente con evidencia de esa página.

13. Si el usuario solicita un documento concreto, utiliza
    solamente ese documento.

14. No mezcles documentos salvo que el usuario solicite
    comparación o análisis conjunto.

15. Si existen inconsistencias de OCR, NO corrijas el nombre
    por intuición.

16. Si aparecen nombres diferentes que podrían representar
    a la misma persona, indica la inconsistencia.

17. Para DNI, fechas, montos, expedientes, resoluciones,
    artículos, nombres y demás identificadores, exige
    correspondencia explícita con la evidencia.

18. Si solo existe evidencia parcial, responde parcialmente.

19. No agregues una sección "Fuentes".
    Las fuentes serán añadidas por el sistema.

20. Si no existe evidencia suficiente para ninguna parte
    de la pregunta, responde exactamente:

No encontré información suficiente en los documentos
indexados para responder esa pregunta.

CONTEXTO RECUPERADO:

{context}

PREGUNTA:

{question}

RESPUESTA:
"""


# ============================================================
# PROMPT DE VALIDACIÓN LLM
# ============================================================

CLAIM_VALIDATION_PROMPT = """
Eres un verificador estricto de evidencia documental.

Debes determinar si una afirmación está respaldada
EXPLÍCITAMENTE por el contexto.

Devuelve exclusivamente una de estas palabras:

RESPALDADA
NO_RESPALDADA

No agregues explicación.

CONTEXTO:

{context}

AFIRMACIÓN:

{claim}
"""


# ============================================================
# PROMPT DE REPARACIÓN
# ============================================================

REPAIR_PROMPT_TEMPLATE = """
Eres un revisor de respuestas documentales.

Contexto autorizado:

{context}

Respuesta original:

{answer}

REGLAS:

1. Conserva únicamente afirmaciones explícitamente respaldadas.

2. Elimina afirmaciones inferidas.

3. Elimina datos que no aparezcan en el contexto.

4. No inventes nada.

5. No corrijas nombres OCR por intuición.

6. Si hay contradicción, exprésala.

7. No añadas fuentes.

8. No añadas explicaciones sobre tu proceso de revisión.

Entrega únicamente la respuesta corregida.
"""


# ============================================================
# CONTEXTO
# ============================================================

def formatear_contexto(
    docs: list[Document]
) -> str:

    if not docs:
        return ""

    partes = []

    for index, doc in enumerate(
        docs,
        1
    ):

        source = doc.metadata.get(
            "source",
            "desconocido"
        )

        page = doc.metadata.get(
            "page"
        )

        if page is None:

            page_label = "?"

        else:

            try:

                page_label = str(
                    int(page) + 1
                )

            except (
                ValueError,
                TypeError
            ):

                page_label = str(
                    page
                )

        partes.append(

            f"=== EVIDENCIA {index} ===\n"
            f"Archivo: {source}\n"
            f"Página: {page_label}\n"
            f"Contenido:\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(
        partes
    )


# ============================================================
# MÉTRICAS
# ============================================================

def medir_contexto(
    texto: str
) -> dict:

    caracteres = len(
        texto
    )

    palabras = len(
        texto.split()
    )

    tokens_estimados = max(
        1,
        caracteres // 4
    )

    return {

        "caracteres":
            caracteres,

        "palabras":
            palabras,

        "tokens_estimados":
            tokens_estimados,
    }


# ============================================================
# FUENTES
# ============================================================

def formatear_fuentes(
    docs: list[Document]
) -> list[dict]:

    fuentes = []

    vistos = set()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "?"
        )

        page = doc.metadata.get(
            "page"
        )

        try:

            page_number = (
                int(page) + 1
                if page is not None
                else None
            )

        except (
            ValueError,
            TypeError
        ):

            page_number = None

        clave = (
            source,
            page_number
        )

        if clave in vistos:
            continue

        vistos.add(
            clave
        )

        fuentes.append({

            "source":
                source,

            "page":
                page_number
        })

    return fuentes


# ============================================================
# VALIDACIÓN HEURÍSTICA
# ============================================================

def _detecta_inferencia(
    claim: str
) -> bool:

    q = normalizar_texto(
        claim
    )

    patrones = [

        r"\bse puede inferir\b",

        r"\bpuede inferirse\b",

        r"\bpor lo tanto\b",

        r"\bpor ende\b",

        r"\bpor consiguiente\b",

        r"\besto demuestra\b",

        r"\besto significa\b",

        r"\blo que indica\b",

        r"\blo cual indica\b",

        r"\bde ello se desprende\b",

        r"\ben consecuencia\b",

        r"\bprobablemente\b",

        r"\bse deduce\b",

        r"\bse puede deducir\b",

        r"\bdebe haber sido\b",

        r"\bdebio haber sido\b",

        r"\bdespues de\b",

        r"\bantes de\b",
    ]

    return any(
        re.search(
            patron,
            q
        )
        for patron in patrones
    )


def _validar_claim_basico(
    claim: str,
    context: str
) -> tuple[
    bool,
    float,
    str
]:

    claim = claim.strip()

    if len(claim) < 8:

        return (
            True,
            1.0,
            "fragmento no factual"
        )

    if _detecta_inferencia(
        claim
    ):

        return (
            False,
            0.0,
            "posible inferencia"
        )

    claim_tokens = (
        tokens_significativos(
            claim
        )
    )

    context_tokens = (
        tokens_significativos(
            context
        )
    )

    if not claim_tokens:

        return (
            True,
            1.0,
            "sin tokens críticos"
        )

    interseccion = (
        claim_tokens &
        context_tokens
    )

    overlap = (
        len(interseccion)
        /
        len(claim_tokens)
    )

    # --------------------------------------------------------
    # Números
    # --------------------------------------------------------

    numeros_claim = extraer_numeros(
        claim
    )

    numeros_context = extraer_numeros(
        context
    )

    numeros_faltantes = (
        numeros_claim
        -
        numeros_context
    )

    if numeros_faltantes:

        return (
            False,
            overlap,
            "número no presente"
        )

    # --------------------------------------------------------
    # Identificadores sensibles
    # --------------------------------------------------------

    claim_normal = normalizar_texto(
        claim
    )

    sensible = (

        "dni" in claim_normal

        or

        "expediente" in claim_normal

        or

        "numero de expediente"
        in claim_normal

        or

        "resolucion" in claim_normal
    )

    if sensible:

        if overlap >= MIN_CLAIM_TOKEN_OVERLAP_STRICT:

            return (
                True,
                overlap,
                "afirmación sensible respaldada"
            )

        return (
            False,
            overlap,
            "afirmación sensible débil"
        )

    if overlap >= 0.45:

        return (
            True,
            overlap,
            "alta coincidencia"
        )

    if overlap >= MIN_CLAIM_TOKEN_OVERLAP:

        return (
            True,
            overlap,
            "coincidencia suficiente"
        )

    return (
        False,
        overlap,
        "baja coincidencia"
    )


# ============================================================
# VALIDACIÓN LLM DE CLAIM
# ============================================================

def validar_claim_con_llm(
    llm,
    context: str,
    claim: str
) -> bool:

    if not CLAIM_VALIDATOR_LLM_ENABLED:
        return True

    prompt = ChatPromptTemplate.from_template(
        CLAIM_VALIDATION_PROMPT
    )

    try:

        mensajes = prompt.format_messages(

            context=context,

            claim=claim
        )

        respuesta = llm.invoke(
            mensajes
        )

        contenido = str(
            getattr(
                respuesta,
                "content",
                ""
            )
            or ""
        ).strip().upper()

        return (
            contenido.startswith(
                "RESPALDADA"
            )
        )

    except Exception as e:

        log(
            f"    -> Validator LLM error: {e}"
        )

        # En caso de fallo del verificador,
        # conservamos el resultado heurístico.
        return True


# ============================================================
# VALIDAR RESPUESTA
# ============================================================

def validar_respuesta(
    llm,
    answer: str,
    docs: list[Document]
) -> dict:

    answer_limpio = (
        limpiar_fuentes_generadas(
            answer
        )
    )

    context = (
        formatear_contexto(
            docs
        )
    )

    frases = dividir_en_frases(
        answer_limpio
    )

    claims = []

    unsupported = []

    for index, claim in enumerate(
        frases
    ):

        if (
            index
            >=
            MAX_CLAIMS_TO_VERIFY
        ):

            break

        ok, overlap, reason = (
            _validar_claim_basico(
                claim,
                context
            )
        )

        # ----------------------------------------------------
        # Si heurística detecta una posible afirmación,
        # usa el LLM como segunda validación.
        # ----------------------------------------------------

        if (
            ok
            and
            VALIDATOR_ENABLED
            and
            CLAIM_VALIDATOR_LLM_ENABLED
            and
            len(
                claim
            ) >= 20
        ):

            llm_ok = (
                validar_claim_con_llm(
                    llm,
                    context,
                    claim
                )
            )

            if not llm_ok:

                ok = False

                reason = (
                    "validator LLM: "
                    "no respaldada"
                )

        item = {

            "claim":
                claim,

            "ok":
                ok,

            "overlap":
                overlap,

            "reason":
                reason,
        }

        claims.append(
            item
        )

        if not ok:

            unsupported.append(
                item
            )

    return {

        "ok":
            len(
                unsupported
            ) == 0,

        "claims":
            claims,

        "unsupported":
            unsupported,

        "answer_clean":
            answer_limpio,
    }


# ============================================================
# REPARACIÓN
# ============================================================

def reparar_respuesta(
    llm,
    answer: str,
    docs: list[Document]
) -> str:

    context = (
        formatear_contexto(
            docs
        )
    )

    prompt = ChatPromptTemplate.from_template(
        REPAIR_PROMPT_TEMPLATE
    )

    try:

        mensajes = prompt.format_messages(

            context=context,

            answer=answer
        )

        respuesta = llm.invoke(
            mensajes
        )

        contenido = str(
            getattr(
                respuesta,
                "content",
                ""
            )
            or ""
        ).strip()

        return (
            contenido
            if contenido
            else answer
        )

    except Exception as e:

        log(
            f"    -> Error reparación: {e}"
        )

        return answer


# ============================================================
# RESPUESTA VALIDADA
# ============================================================

def generar_respuesta_validada(
    llm,
    prompt,
    question: str,
    docs: list[Document]
) -> tuple[
    str,
    dict
]:

    context = (
        formatear_contexto(
            docs
        )
    )

    if not context:

        return (

            "No encontré información "
            "suficiente en los documentos "
            "indexados para responder esa "
            "pregunta.",

            {
                "validated":
                    True,

                "repaired":
                    False
            }
        )

    mensajes = prompt.format_messages(

        context=context,

        question=question
    )

    try:

        respuesta = llm.invoke(
            mensajes
        )

        draft = str(
            getattr(
                respuesta,
                "content",
                ""
            )
            or ""
        ).strip()

    except Exception as e:

        log(
            f"\nERROR LLM: {e}"
        )

        return (

            "Error al generar "
            "la respuesta con Ollama.",

            {
                "validated":
                    False,

                "repaired":
                    False,

                "error":
                    True
            }
        )

    if not draft:

        return (

            "No encontré información "
            "suficiente en los documentos "
            "indexados para responder esa "
            "pregunta.",

            {
                "validated":
                    True,

                "repaired":
                    False
            }
        )

    # ========================================================
    # PRIMERA VALIDACIÓN
    # ========================================================

    validacion = (
        validar_respuesta(
            llm,
            draft,
            docs
        )
    )

    if validacion["ok"]:

        return (

            validacion["answer_clean"],

            {
                "validated":
                    True,

                "repaired":
                    False,

                "unsupported":
                    0
            }
        )

    # ========================================================
    # REPARACIÓN
    # ========================================================

    if REPAIR_UNSUPPORTED_CLAIMS:

        repaired = (
            reparar_respuesta(
                llm,
                validacion[
                    "answer_clean"
                ],
                docs
            )
        )

        validacion_reparada = (
            validar_respuesta(
                llm,
                repaired,
                docs
            )
        )

        if validacion_reparada["ok"]:

            return (

                validacion_reparada[
                    "answer_clean"
                ],

                {
                    "validated":
                        True,

                    "repaired":
                        True,

                    "unsupported_before":
                        len(
                            validacion[
                                "unsupported"
                            ]
                        ),

                    "unsupported_after":
                        0
                }
            )

        # ----------------------------------------------------
        # Fallback: conservar solamente claims respaldados
        # ----------------------------------------------------

        claims_validos = [

            item["claim"]

            for item
            in validacion["claims"]

            if item["ok"]
        ]

        if claims_validos:

            return (

                "\n".join(
                    claims_validos
                ),

                {
                    "validated":
                        True,

                    "repaired":
                        True,

                    "partial":
                        True,

                    "unsupported_before":
                        len(
                            validacion[
                                "unsupported"
                            ]
                        )
                }
            )

    # ========================================================
    # NO SE PUDO GARANTIZAR
    # ========================================================

    return (

        "No encontré información suficiente "
        "en los documentos indexados para "
        "responder esa pregunta.",

        {
            "validated":
                False,

            "repaired":
                False,

            "failed_validation":
                True,

            "unsupported":
                len(
                    validacion[
                        "unsupported"
                    ]
                )
        }
    )


# ============================================================
# CONSTRUIR LLM
# ============================================================

def construir_llm():

    return ChatOllama(

        model=LLM_MODEL,

        base_url=OLLAMA_URL,

        temperature=TEMPERATURE,

        num_predict=MAX_TOKENS_RESPUESTA,

        num_ctx=CONTEXTO_LLM,

        reasoning=False,

        streaming=False
    )


# ============================================================
# CHAIN COMPATIBLE
# ============================================================

def construir_chain():

    llm = construir_llm()

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    class ChainCompatible:

        def stream(
            self,
            question,
            docs=None
        ):

            if docs is None:

                resultado = (
                    ejecutar_pipeline(
                        question
                    )
                )

                docs = (
                    resultado.docs_evidencia
                )

            if not docs:

                yield (
                    "No encontré información "
                    "suficiente en los documentos "
                    "indexados para responder esa "
                    "pregunta."
                )

                return

            contexto = (
                formatear_contexto(
                    docs
                )
            )

            metricas = medir_contexto(
                contexto
            )

            log(
                "\n--- Métricas del contexto ---\n"

                f"  Chunks: "
                f"{len(docs)}\n"

                f"  Caracteres: "
                f"{metricas['caracteres']}\n"

                f"  Palabras: "
                f"{metricas['palabras']}\n"

                f"  Tokens estimados: "
                f"{metricas['tokens_estimados']}\n"
            )

            inicio = time.time()

            respuesta, validacion = (
                generar_respuesta_validada(

                    llm,

                    prompt,

                    question,

                    docs
                )
            )

            log(
                "\n--- Validación de respuesta ---"
            )

            log(
                f"  Validada: "
                f"{validacion.get('validated')}"
            )

            log(
                f"  Reparada: "
                f"{validacion.get('repaired')}"
            )

            if (
                "unsupported_before"
                in validacion
            ):

                log(
                    "  Claims no respaldados "
                    "antes de reparar: "
                    f"{validacion['unsupported_before']}"
                )

            if (
                "unsupported"
                in validacion
            ):

                log(
                    "  Claims no respaldados: "
                    f"{validacion['unsupported']}"
                )

            log(
                f"  Tiempo LLM + validación: "
                f"{time.time() - inicio:.2f}s"
            )

            # ------------------------------------------------
            # Compatibilidad con streaming.
            #
            # La respuesta ya está validada antes
            # de ser emitida.
            # ------------------------------------------------

            tamano = 120

            for pos in range(
                0,
                len(respuesta),
                tamano
            ):

                yield respuesta[
                    pos:
                    pos + tamano
                ]

        def invoke(
            self,
            question
        ):

            resultado = (
                ejecutar_pipeline(
                    question
                )
            )

            docs = (
                resultado.docs_evidencia
            )

            if not docs:

                return Document(
                    page_content=(
                        "No encontré información "
                        "suficiente en los documentos "
                        "indexados para responder esa "
                        "pregunta."
                    )
                )

            respuesta, _ = (
                generar_respuesta_validada(

                    llm,

                    prompt,

                    question,

                    docs
                )
            )

            return Document(
                page_content=respuesta
            )

    retriever = vector_db.as_retriever(

        search_kwargs={
            "k":
                RETRIEVER_K
        }
    )

    return (
        ChainCompatible(),
        retriever
    )


# ============================================================
# DEBUG
# ============================================================

def imprimir_debug(
    resultado: PipelineResult
):

    ctx = resultado.ctx

    print(
        "--- Paso 1: intención detectada ---"
    )

    print(
        "  Fuente principal: "
        f"{ctx.fuente_explicita or 'ninguna explícita'}"
    )

    if ctx.fuentes_explicitas:

        print(
            "  Fuentes explícitas:"
        )

        for fuente in (
            ctx.fuentes_explicitas
        ):

            print(
                f"    - {fuente}"
            )

    if ctx.pagina_explicita is not None:

        print(
            "  Página detectada: "
            f"{ctx.pagina_explicita + 1}"
        )

    elif ctx.ultima_pagina_solicitada:

        print(
            "  Página solicitada: "
            "última, no resuelta"
        )

    else:

        print(
            "  Página detectada: "
            "ninguna explícita"
        )

    print(
        "  Comparación: "
        +
        (
            "sí"
            if ctx.es_comparacion
            else "no"
        )
    )

    print(
        "  Tipo de consulta: "
        f"{ctx.tipo_consulta}"
    )

    print(
        "  Query variants:"
    )

    for variant in ctx.query_variants:

        print(
            f"    - {variant}"
        )

    # ========================================================
    # PASO 2
    # ========================================================

    print(
        "\n--- Paso 2: candidatos recuperados ---"
    )

    for doc in resultado.docs_candidatos:

        source = doc.metadata.get(
            "source",
            "?"
        )

        page = doc.metadata.get(
            "page"
        )

        try:

            page_visible = (
                int(page) + 1
            )

        except (
            ValueError,
            TypeError
        ):

            page_visible = page

        distance = doc.metadata.get(
            "_retrieval_distance",
            "?"
        )

        score = doc.metadata.get(
            "_retrieval_score",
            "?"
        )

        preview = (
            doc.page_content[:100]
            .replace(
                "\n",
                " "
            )
        )

        print(

            f"  [{source} | "
            f"pág {page_visible} | "
            f"dist {distance} | "
            f"score {score}] "
            f"{preview}..."
        )

    # ========================================================
    # PASO 5
    # ========================================================

    print(
        "\n--- Paso 5: evidencia final "
        "enviada al LLM ---"
    )

    for score, doc in (
        resultado.scores_evidencia
    ):

        source = doc.metadata.get(
            "source",
            "?"
        )

        page = doc.metadata.get(
            "page"
        )

        try:

            page_visible = (
                int(page) + 1
            )

        except (
            ValueError,
            TypeError
        ):

            page_visible = page

        preview = (
            doc.page_content[:120]
            .replace(
                "\n",
                " "
            )
        )

        print(

            f"  [{source} | "
            f"pág {page_visible} | "
            f"score {score:.4f}] "
            f"{preview}..."
        )

    print(
        "-----------------------------------\n"
    )


# ============================================================
# PREGUNTAR CLI
# ============================================================

def preguntar(
    chain,
    question: str,
    retriever
):

    inicio = time.time()

    resultado = (
        ejecutar_pipeline(
            question
        )
    )

    imprimir_debug(
        resultado
    )

    print(
        f"[Contexto final: "
        f"{len(resultado.docs_evidencia)} "
        "chunks]\n"
    )

    if not resultado.docs_evidencia:

        print(
            "No encontré información "
            "suficiente en los documentos "
            "indexados para responder esa "
            "pregunta."
        )

        return

    respuesta_completa = ""

    primer_fragmento = None

    for pedazo in chain.stream(

        question,

        docs=resultado.docs_evidencia
    ):

        if primer_fragmento is None:

            primer_fragmento = (
                time.time()
            )

            print(
                f"[primer fragmento en "
                f"{primer_fragmento - inicio:.2f}s]\n"
            )

        respuesta_completa += pedazo

        print(
            pedazo,
            end="",
            flush=True
        )

    # ========================================================
    # FUENTES CONTROLADAS POR PYTHON
    # ========================================================

    fuentes = formatear_fuentes(
        resultado.docs_evidencia
    )

    print(
        "\n\nFuentes:"
    )

    for fuente in fuentes:

        print(
            f"- {fuente['source']}, "
            f"página {fuente['page']}"
        )

    if not respuesta_completa.strip():

        log(
            "\nADVERTENCIA: "
            "respuesta vacía."
        )

    else:

        log(
            f"\n[Completado en "
            f"{time.time() - inicio:.2f}s]"
        )


# ============================================================
# FASTAPI STREAM
# ============================================================

def preguntar_stream(
    chain,
    question: str,
    retriever
):

    resultado = (
        ejecutar_pipeline(
            question
        )
    )

    # ========================================================
    # SIN EVIDENCIA
    # ========================================================

    if not resultado.docs_evidencia:

        yield (

            "data: "

            +

            json.dumps(

                {
                    "type":
                        "sources",

                    "data":
                        []
                },

                ensure_ascii=False
            )

            +

            "\n\n"
        )

        mensaje = (
            "No encontré información "
            "suficiente en los documentos "
            "indexados para responder esa "
            "pregunta."
        )

        yield (

            "data: "

            +

            json.dumps(

                {
                    "type":
                        "chunk",

                    "data":
                        mensaje
                },

                ensure_ascii=False
            )

            +

            "\n\n"
        )

        yield (

            "data: "

            +

            json.dumps(

                {
                    "type":
                        "done"
                },

                ensure_ascii=False
            )

            +

            "\n\n"
        )

        return

    # ========================================================
    # FUENTES
    # ========================================================

    fuentes = formatear_fuentes(
        resultado.docs_evidencia
    )

    yield (

        "data: "

        +

        json.dumps(

            {
                "type":
                    "sources",

                "data":
                    fuentes
            },

            ensure_ascii=False
        )

        +

        "\n\n"
    )

    # ========================================================
    # RESPUESTA VALIDADA
    # ========================================================

    for pedazo in chain.stream(

        question,

        docs=resultado.docs_evidencia
    ):

        yield (

            "data: "

            +

            json.dumps(

                {
                    "type":
                        "chunk",

                    "data":
                        pedazo
                },

                ensure_ascii=False
            )

            +

            "\n\n"
        )

    # ========================================================
    # FIN
    # ========================================================

    yield (

        "data: "

        +

        json.dumps(

            {
                "type":
                    "done"
            },

            ensure_ascii=False
        )

        +

        "\n\n"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n======================================"
    )

    print(
        "      RAG-JUCHLM v8"
    )

    print(
        " RAG HÍBRIDO + CLAIM VALIDATOR"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # LISTAR
    # --------------------------------------------------------

    if "--listar" in sys.argv:

        listar_indexados()

        return

    # --------------------------------------------------------
    # SINCRONIZACIÓN
    # --------------------------------------------------------

    sincronizar_carpeta()

    # --------------------------------------------------------
    # SOLO INGESTA
    # --------------------------------------------------------

    if "--solo-ingesta" in sys.argv:

        return

    # --------------------------------------------------------
    # CHAIN
    # --------------------------------------------------------

    chain, retriever = (
        construir_chain()
    )

    print(
        "\n======================================"
    )

    print(
        "Escribí tu pregunta "
        "(o 'salir' para terminar)"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while True:

        pregunta = input(
            "\nPregunta: "
        ).strip()

        if pregunta.lower() in (
            "salir",
            "exit",
            "quit",
            ""
        ):

            break

        print()

        preguntar(
            chain,
            pregunta,
            retriever
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()