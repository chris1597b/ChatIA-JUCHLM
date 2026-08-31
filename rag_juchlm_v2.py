import os
import json
import re
import hashlib
import sys
import time
import unicodedata
from pathlib import Path
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


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

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100

# Recuperamos más candidatos inicialmente para poder seleccionar
# la fuente/documento correcto antes de entregarlo al LLM.
RETRIEVER_K = 30

# Cantidad máxima de chunks que finalmente recibe el LLM.
FINAL_CONTEXT_CHUNKS = 8

MAX_TOKENS_RESPUESTA = 2048
CONTEXTO_LLM = 4096

SCANNED_THRESHOLD_CHARS_POR_PAGINA = 30


# ============================================================
# UTILIDADES
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparar nombres de archivos y consultas.
    No modifica el texto que se envía al LLM.
    """
    texto = str(texto or "").lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokens_significativos(texto: str) -> set[str]:
    stopwords = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "en", "por", "para", "con", "sin",
        "que", "cual", "cuales", "como", "donde", "cuando", "quien",
        "quienes", "es", "son", "fue", "fueron", "se", "su", "sus",
        "sobre", "segun", "este", "esta", "estos", "estas", "mi",
        "me", "te", "y", "o", "u", "un", "lo", "hay", "aparece",
        "dice", "dime", "indica", "informacion", "documento"
    }
    return {
        t for t in normalizar_texto(texto).split()
        if len(t) >= 3 and t not in stopwords
    }


# ============================================================
# MANIFEST
# ============================================================

def cargar_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def guardar_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def hash_archivo(path: Path) -> str:
    stat = path.stat()
    firma = f"{stat.st_size}-{stat.st_mtime}"
    return hashlib.md5(firma.encode()).hexdigest()


# ============================================================
# PDF / OCR
# ============================================================

def es_probable_escaneado(documents: list[Document]) -> bool:
    if not documents:
        return False

    total_chars = sum(len(d.page_content.strip()) for d in documents)
    promedio = total_chars / len(documents)

    return promedio < SCANNED_THRESHOLD_CHARS_POR_PAGINA


def cargar_pdf(path: Path) -> tuple[list[Document], bool]:
    """
    Carga texto normal y, si parece escaneado, intenta RapidOCR.
    """
    docs_texto = PyPDFLoader(str(path)).load()

    if not es_probable_escaneado(docs_texto):
        return docs_texto, False

    log(
        f"    -> '{path.name}' parece escaneado, aplicando OCR "
        f"(esto es más lento, tené paciencia)..."
    )

    try:
        from langchain_community.document_loaders import RapidOCRPDFLoader

        docs_ocr = RapidOCRPDFLoader(str(path)).load()

        if es_probable_escaneado(docs_ocr):
            log(
                f"    -> OCR de '{path.name}' extrajo muy poco texto. "
                f"Revisá manualmente si el PDF está en buen estado."
            )

        return docs_ocr, True

    except ImportError:
        log("    -> ERROR: falta 'rapidocr-onnxruntime'. Instalalo con:")
        log("       pip install rapidocr-onnxruntime")
        log(
            f"    -> Se usará el texto probablemente parcial de "
            f"'{path.name}' sin OCR."
        )
        return docs_texto, False


# ============================================================
# EMBEDDINGS + CHROMA
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
# INGESTA / SINCRONIZACIÓN
# ============================================================

def sincronizar_carpeta():
    if not PDF_FOLDER.exists():
        log(f"\nERROR: no existe la carpeta '{PDF_FOLDER.absolute()}'")
        raise SystemExit(1)

    pdfs_en_disco = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdfs_en_disco:
        log(f"\nNo hay PDFs en '{PDF_FOLDER.absolute()}'.")
        return

    manifest = cargar_manifest()
    nombres_en_disco = {p.name for p in pdfs_en_disco}

    # Eliminar documentos ausentes del disco.
    eliminados = [
        nombre for nombre in manifest
        if nombre not in nombres_en_disco
    ]

    for nombre in eliminados:
        log(
            f"\nEliminando de la base vectorial: '{nombre}' "
            f"(ya no está en la carpeta)"
        )
        vector_db.delete(where={"source": nombre})
        del manifest[nombre]

    nuevos_o_cambiados = []

    for path in pdfs_en_disco:
        firma_actual = hash_archivo(path)
        firma_previa = manifest.get(path.name, {}).get("firma")

        if firma_actual != firma_previa:
            nuevos_o_cambiados.append(path)

    if not nuevos_o_cambiados and not eliminados:
        log(
            f"\nBase vectorial ya actualizada. "
            f"{len(pdfs_en_disco)} PDFs indexados, sin cambios."
        )
        return

    log(
        f"\n{len(nuevos_o_cambiados)} PDF(s) nuevo(s)/modificado(s) "
        f"de {len(pdfs_en_disco)} totales. Procesando...\n"
    )

    for i, path in enumerate(nuevos_o_cambiados, 1):
        log(f"[{i}/{len(nuevos_o_cambiados)}] {path.name}")
        inicio = time.time()

        if path.name in manifest:
            vector_db.delete(where={"source": path.name})

        try:
            docs, fue_ocr = cargar_pdf(path)
        except Exception as e:
            log(
                f"    -> ERROR procesando '{path.name}': {e}. "
                f"Se omite este archivo."
            )
            continue

        # Metadata explícita y estable.
        for d in docs:
            d.metadata["source"] = path.name
            d.metadata["ocr"] = fue_ocr

            # PyPDFLoader/RapidOCR normalmente entrega page 0-based.
            # Conservamos el valor original y agregamos page_number para
            # dejar claro que también es 1-based.
            pagina = d.metadata.get("page")

            if pagina is not None:
                try:
                    pagina_int = int(pagina)
                    d.metadata["page"] = pagina_int
                    d.metadata["page_number"] = pagina_int + 1
                except (ValueError, TypeError):
                    d.metadata["page_number"] = None
            else:
                d.metadata["page_number"] = None

            d.metadata["document_id"] = path.stem

        chunks = splitter.split_documents(docs)

        if not chunks:
            log(
                f"    -> '{path.name}' no generó ningún chunk "
                f"de texto (¿PDF vacío?)."
            )
            continue

        # IDs deterministas por documento y chunk.
        ids = [
            f"{path.name}::{idx}"
            for idx in range(len(chunks))
        ]

        vector_db.add_documents(
            documents=chunks,
            ids=ids
        )

        manifest[path.name] = {
            "firma": hash_archivo(path),
            "chunks": len(chunks),
            "ocr": fue_ocr,
        }

        log(
            f"    -> {len(chunks)} chunks, "
            f"{'OCR' if fue_ocr else 'texto directo'}, "
            f"{time.time() - inicio:.2f}s"
        )

    guardar_manifest(manifest)

    log(
        f"\nSincronización completa. "
        f"Total PDFs indexados: {len(manifest)}"
    )


# ============================================================
# ANÁLISIS GENÉRICO DE CONSULTA
# ============================================================

def detectar_pagina(question: str):
    """
    Detecta restricciones explícitas de página.

    Ejemplos:
      'primera página' -> 0
      'página 1'       -> 0
      'página 5'       -> 4

    Devuelve None si no hay una restricción explícita.
    """
    q = normalizar_texto(question)

    if re.search(r"\bprimera pagina\b", q):
        return 0

    if re.search(r"\bsegunda pagina\b", q):
        return 1

    if re.search(r"\btercera pagina\b", q):
        return 2

    match = re.search(r"\bp(?:a|á)gina\s+(\d+)\b", question.lower())

    if match:
        numero = int(match.group(1))
        if numero >= 1:
            return numero - 1

    return None


def detectar_comparacion(question: str) -> bool:
    q = normalizar_texto(question)

    patrones = [
        "compara",
        "comparar",
        "diferencia entre",
        "diferencias entre",
        "ambos documentos",
        "los dos documentos",
        "entre los documentos",
        "contrasta",
        "relacion entre",
    ]

    return any(p in q for p in patrones)


def obtener_fuente_explicita(question: str):
    """
    Intenta determinar si el usuario está apuntando explícitamente
    a uno de los PDFs indexados.

    No depende de un tipo concreto de documento: funciona con
    expedientes, manuales, tesis, informes, reglamentos, etc.
    """
    pdfs = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdfs:
        return None

    q_norm = normalizar_texto(question)

    # 1) Coincidencia fuerte con el nombre completo.
    for path in pdfs:
        nombre_norm = normalizar_texto(path.name)

        if nombre_norm and nombre_norm in q_norm:
            return path.name

    # 2) Coincidencia por nombre sin extensión.
    for path in pdfs:
        stem_norm = normalizar_texto(path.stem)

        if len(stem_norm) >= 8 and stem_norm in q_norm:
            return path.name

    # 3) Coincidencia por tokens significativos del nombre.
    # Solo se activa si varios tokens del nombre aparecen en la consulta.
    q_tokens = tokens_significativos(question)

    candidatos = []

    for path in pdfs:
        nombre_tokens = tokens_significativos(path.stem)

        if not nombre_tokens:
            continue

        coincidencias = q_tokens & nombre_tokens
        proporcion = len(coincidencias) / max(1, len(nombre_tokens))

        if len(coincidencias) >= 2 and proporcion >= 0.35:
            candidatos.append(
                (proporcion, len(coincidencias), path.name)
            )

    if candidatos:
        candidatos.sort(reverse=True)
        return candidatos[0][2]

    # 4) Si el usuario dice "el expediente", "el manual", etc.,
    # intentamos usar tokens distintivos del nombre, pero solo cuando
    # hay una única coincidencia razonablemente clara.
    palabras_genericas = {
        "expediente", "manual", "reglamento", "tesis", "informe",
        "articulo", "articulos", "contrato", "resolucion",
        "procedimiento", "protocolo", "estudio", "documentacion"
    }

    coincidencias_genericas = []

    for path in pdfs:
        nombre_tokens = tokens_significativos(path.stem)
        encontrados = q_tokens & nombre_tokens & palabras_genericas

        if encontrados:
            coincidencias_genericas.append(
                (len(encontrados), path.name)
            )

    if len(coincidencias_genericas) == 1:
        return coincidencias_genericas[0][1]

    return None


# ============================================================
# RETRIEVAL
# ============================================================

def recuperar_candidatos(question: str):
    """
    Primera recuperación amplia.

    Si existe una fuente explícita, se filtra desde el inicio.
    Si existe una página explícita, también se utiliza metadata.
    """
    fuente = obtener_fuente_explicita(question)
    pagina = detectar_pagina(question)

    filtros = {}

    if fuente:
        filtros["source"] = fuente

    if pagina is not None:
        filtros["page"] = pagina

    try:
        if filtros:
            docs = vector_db.similarity_search(
                question,
                k=RETRIEVER_K,
                filter=filtros
            )
        else:
            docs = vector_db.similarity_search(
                question,
                k=RETRIEVER_K
            )
    except Exception as e:
        log(f"ADVERTENCIA retrieval: {e}")
        docs = vector_db.similarity_search(
            question,
            k=RETRIEVER_K
        )

    return docs, fuente, pagina


def puntuacion_lexica(question: str, doc: Document) -> float:
    """
    Puntaje lexical auxiliar.

    No reemplaza embeddings. Sirve para priorizar nombres, números,
    términos exactos y entidades que los embeddings pueden recuperar
    de forma imperfecta.
    """
    q_tokens = tokens_significativos(question)

    texto = doc.page_content
    texto_tokens = tokens_significativos(texto)

    if not q_tokens or not texto_tokens:
        return 0.0

    interseccion = q_tokens & texto_tokens

    # Cobertura de términos de la consulta.
    cobertura = len(interseccion) / len(q_tokens)

    # Pequeño bono por coincidencias exactas de tokens largos.
    bono = sum(
        0.15 for token in interseccion
        if len(token) >= 7
    )

    return cobertura + bono


def seleccionar_contexto(
    question: str,
    docs: list[Document],
    fuente_explicita=None,
    pagina_explicita=None
):
    """
    Selecciona el contexto final.

    Objetivos:
      - no mezclar PDFs arbitrariamente;
      - respetar documento explícito;
      - respetar página explícita;
      - conservar varios documentos solo cuando la pregunta pide
        comparación/relación;
      - combinar similitud semántica con una señal lexical.
    """
    if not docs:
        return []

    # Si se especificó una fuente, todos los resultados deben pertenecer
    # a ella. Esto es una garantía de aislamiento.
    if fuente_explicita:
        docs = [
            d for d in docs
            if d.metadata.get("source") == fuente_explicita
        ]

    # Si se especificó página, respetarla estrictamente.
    if pagina_explicita is not None:
        docs_pagina = [
            d for d in docs
            if str(d.metadata.get("page", "")) == str(pagina_explicita)
        ]

        # Si hay resultados de esa página, usar exclusivamente esos.
        if docs_pagina:
            docs = docs_pagina

    if not docs:
        return []

    # Calcular señal lexical y mantener el orden de similitud de Chroma
    # como componente principal. Cuanto más arriba estaba el chunk,
    # mayor será su score semántico aproximado.
    candidatos = []

    for posicion, d in enumerate(docs):
        lexical = puntuacion_lexica(question, d)

        # Score de posición: 1 para el primer resultado y decrece.
        semantico_proxy = 1.0 / (1.0 + posicion * 0.15)

        score = (0.75 * semantico_proxy) + (0.25 * lexical)

        candidatos.append((score, d))

    candidatos.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Para preguntas normales queremos una sola fuente dominante.
    # Para comparación permitimos varias.
    comparacion = detectar_comparacion(question)

    if fuente_explicita or pagina_explicita is not None:
        seleccionados = [d for _, d in candidatos]
    elif comparacion:
        seleccionados = [d for _, d in candidatos]
    else:
        por_fuente = defaultdict(list)

        for score, d in candidatos:
            fuente = d.metadata.get("source", "desconocido")
            por_fuente[fuente].append((score, d))

        ranking_fuentes = []

        for fuente, elementos in por_fuente.items():
            scores = [x[0] for x in elementos]

            # Peso mayor al mejor resultado, pero también cuenta
            # la consistencia de los siguientes.
            score_fuente = (
                scores[0]
                + 0.45 * (scores[1] if len(scores) > 1 else 0)
                + 0.25 * (scores[2] if len(scores) > 2 else 0)
            )

            ranking_fuentes.append(
                (score_fuente, fuente)
            )

        ranking_fuentes.sort(reverse=True)

        fuente_principal = ranking_fuentes[0][1]

        # Para consultas normales aislamos la fuente dominante.
        seleccionados = [
            d for _, d in candidatos
            if d.metadata.get("source") == fuente_principal
        ]

    # Límite final.
    return seleccionados[:FINAL_CONTEXT_CHUNKS]


def recuperar_contexto(question: str):
    docs, fuente, pagina = recuperar_candidatos(question)

    docs_finales = seleccionar_contexto(
        question,
        docs,
        fuente_explicita=fuente,
        pagina_explicita=pagina
    )

    return docs_finales, docs, fuente, pagina


# ============================================================
# PROMPT GENERALISTA
# ============================================================

PROMPT_TEMPLATE = """
Eres el asistente de consulta documental de JUCHLM.

Tu función es responder preguntas utilizando EXCLUSIVAMENTE la
evidencia contenida en el CONTEXTO RECUPERADO.

REGLAS OBLIGATORIAS:

1. Usa únicamente la información que aparece explícitamente en el
   contexto recuperado. No uses conocimiento externo.

2. No inventes, completes, supongas ni deduzcas datos que no estén
   explícitamente respaldados por el contexto.

3. Para datos concretos como fechas, nombres, números de expediente,
   montos, cargos, artículos, resoluciones, lugares o identificadores,
   solo proporciona un valor cuando la evidencia lo respalde
   explícitamente.

4. NO infieras un dato específico a partir de otro dato.
   Ejemplo: si aparece "2016" en un número de expediente, eso NO
   demuestra que una sentencia haya sido emitida en 2016.

5. Distingue datos que puedan parecer similares:
   fecha de los hechos ≠ fecha de la sentencia;
   fecha de publicación ≠ fecha del documento;
   autor ≠ juez;
   agraviado ≠ testigo;
   acusado ≠ funcionario;
   número de expediente ≠ número de resolución.

6. Respeta estrictamente las restricciones expresadas por el usuario:
   documento, archivo, página, sección, capítulo, apartado, tabla,
   figura u otra parte específica.

7. Si el usuario solicita información de un documento específico,
   responde únicamente con evidencia perteneciente a ese documento.

8. No mezcles información de diferentes documentos salvo que el usuario
   solicite explícitamente una comparación, relación o síntesis entre
   documentos.

9. Cada bloque del contexto está identificado por archivo y página.
   Trata esas etiquetas como la procedencia de la evidencia.

10. Si una afirmación aparece en un documento distinto al documento
    solicitado por el usuario, NO la utilices.

11. Si el contexto contiene información insuficiente para responder,
    dilo claramente. No rellenes el vacío mediante inferencias.

12. Si el usuario solicita una página específica y la evidencia de esa
    página no contiene el dato solicitado, no utilices otra página para
    sustituirla sin indicarlo.

13. Si existen evidencias contradictorias dentro del contexto, no
    elijas arbitrariamente una. Expón la contradicción y señala las
    páginas correspondientes.

14. Cuando respondas, indica al final:
    "Fuentes: [archivo, página]"

15. Si NO existe evidencia suficiente para responder, utiliza esta
    respuesta:
    "No encontré información suficiente en los documentos indexados para responder esa pregunta."

16. No digas que viste, analizaste o recibiste un documento que no
    aparece en el contexto recuperado.

17. Prioriza precisión y fidelidad documental sobre cantidad de texto.

CONTEXTO RECUPERADO:
{context}

PREGUNTA:
{question}

RESPUESTA:
"""


# ============================================================
# FORMATEO DE CONTEXTO
# ============================================================

def formatear_contexto(docs: list[Document]) -> str:
    if not docs:
        return ""

    partes = []

    for i, d in enumerate(docs, 1):
        fuente = d.metadata.get("source", "desconocido")
        pagina = d.metadata.get("page")

        if pagina is None:
            pagina_label = "?"
        else:
            try:
                pagina_label = str(int(pagina) + 1)
            except (ValueError, TypeError):
                pagina_label = str(pagina)

        partes.append(
            f"=== EVIDENCIA {i} ===\n"
            f"Archivo: {fuente}\n"
            f"Página: {pagina_label}\n"
            f"Contenido:\n{d.page_content}"
        )

    return "\n\n".join(partes)


def formatear_fuentes(docs: list[Document]) -> list[dict]:
    fuentes = []
    vistos = set()

    for d in docs:
        source = d.metadata.get("source", "?")
        page = d.metadata.get("page")

        try:
            page_number = int(page) + 1 if page is not None else None
        except (ValueError, TypeError):
            page_number = None

        clave = (source, page_number)

        if clave in vistos:
            continue

        vistos.add(clave)

        fuentes.append({
            "source": source,
            "page": page_number
        })

    return fuentes


# ============================================================
# CADENA RAG
# ============================================================

def construir_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        num_predict=MAX_TOKENS_RESPUESTA,
        num_ctx=CONTEXTO_LLM,
        reasoning=False,
    )


def construir_chain():
    llm = construir_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # La recuperación ahora se realiza dinámicamente antes de generar.
    # Esto permite aplicar filtros por documento/página.
    def generar_respuesta(question: str):
        docs_finales, _, _, _ = recuperar_contexto(question)

        contexto = formatear_contexto(docs_finales)

        if not contexto:
            return (
                "No encontré información suficiente en los documentos "
                "indexados para responder esa pregunta."
            )

        mensajes = prompt.format_messages(
            context=contexto,
            question=question
        )

        return llm.invoke(mensajes).content

    class ChainCompatible:
        """
        Adaptador pequeño para conservar:
          chain.stream(question)
        usado por preguntar() y preguntar_stream().
        """

        def stream(self, question):
            docs_finales, _, _, _ = recuperar_contexto(question)
            contexto = formatear_contexto(docs_finales)

            if not contexto:
                yield (
                    "No encontré información suficiente en los "
                    "documentos indexados para responder esa pregunta."
                )
                return

            mensajes = prompt.format_messages(
                context=contexto,
                question=question
            )

            # ChatOllama permite streaming del mensaje.
            for pedazo in llm.stream(mensajes):
                contenido = getattr(pedazo, "content", "")
                if contenido:
                    yield contenido

        def invoke(self, question):
            return generar_respuesta(question)

    # El retriever se conserva como objeto de compatibilidad para app.py,
    # pero la recuperación efectiva la controla recuperar_contexto().
    retriever = vector_db.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )

    return ChainCompatible(), retriever


# ============================================================
# DIAGNÓSTICO
# ============================================================

def imprimir_debug(question: str, docs_recuperados, docs_finales,
                    fuente, pagina):
    print("--- Análisis de consulta ---")

    if fuente:
        print(f"  Fuente detectada: {fuente}")
    else:
        print("  Fuente detectada: ninguna explícita")

    if pagina is not None:
        print(f"  Página detectada: {pagina + 1}")
    else:
        print("  Página detectada: ninguna explícita")

    print(f"  Comparación: {'sí' if detectar_comparacion(question) else 'no'}")

    print("\n--- Candidatos recuperados ---")

    for d in docs_recuperados:
        fuente_d = d.metadata.get("source", "?")
        pagina_d = d.metadata.get("page", "?")
        preview = d.page_content[:100].replace("\n", " ")
        print(
            f"  [{fuente_d} | pág {pagina_d}] "
            f"{preview}..."
        )

    print("\n--- Contexto FINAL enviado al LLM ---")

    for d in docs_finales:
        fuente_d = d.metadata.get("source", "?")
        pagina_d = d.metadata.get("page", "?")

        try:
            pagina_visible = int(pagina_d) + 1
        except (ValueError, TypeError):
            pagina_visible = pagina_d

        preview = d.page_content[:120].replace("\n", " ")

        print(
            f"  [{fuente_d} | pág {pagina_visible}] "
            f"{preview}..."
        )

    print("-----------------------------------\n")


def preguntar(chain, question: str, retriever):
    inicio = time.time()

    docs_finales, docs_recuperados, fuente, pagina = recuperar_contexto(
        question
    )

    # IMPORTANTE:
    # este diagnóstico muestra tanto lo recuperado inicialmente como
    # lo que realmente se entrega al LLM.
    imprimir_debug(
        question,
        docs_recuperados,
        docs_finales,
        fuente,
        pagina
    )

    print(f"[Contexto final: {len(docs_finales)} chunks]\n")

    respuesta_completa = ""
    primer_token = None

    for pedazo in chain.stream(question):
        if primer_token is None:
            primer_token = time.time()
            print(
                f"[primer token en "
                f"{primer_token - inicio:.2f}s]\n"
            )

        respuesta_completa += pedazo
        print(pedazo, end="", flush=True)

    duracion = time.time() - inicio

    print()

    if not respuesta_completa.strip():
        log(
            "\nADVERTENCIA: respuesta vacía. Revisá "
            "num_predict/num_ctx o el modelo."
        )
    else:
        log(f"\n[Completado en {duracion:.2f}s]")


def preguntar_stream(chain, question: str, retriever):
    """
    Compatible con el app.py actual.

    IMPORTANTE:
    El frontend recibe ahora las fuentes del CONTEXTO FINAL,
    no simplemente todos los candidatos iniciales.
    """
    docs_finales, docs_recuperados, fuente, pagina = recuperar_contexto(
        question
    )

    fuentes = formatear_fuentes(docs_finales)

    yield (
        f"data: {json.dumps({'type': 'sources', 'data': fuentes}, ensure_ascii=False)}"
        "\n\n"
    )

    for pedazo in chain.stream(question):
        yield (
            f"data: {json.dumps({'type': 'chunk', 'data': pedazo}, ensure_ascii=False)}"
            "\n\n"
        )

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ============================================================
# MAIN
# ============================================================

def listar_indexados():
    manifest = cargar_manifest()

    if not manifest:
        print("\nNo hay PDFs indexados todavía.")
        return

    print(f"\n{len(manifest)} PDF(s) indexados:\n")

    for nombre, info in manifest.items():
        tipo = "OCR" if info.get("ocr") else "texto"
        print(
            f"  - {nombre}: "
            f"{info.get('chunks')} chunks ({tipo})"
        )


def main():
    print("\n======================================")
    print("   RAG-JUCHLM v2 - MULTI-PDF")
    print("======================================")

    if "--listar" in sys.argv:
        listar_indexados()
        return

    sincronizar_carpeta()

    if "--solo-ingesta" in sys.argv:
        return

    chain, retriever = construir_chain()

    print("\n======================================")
    print("Escribí tu pregunta (o 'salir' para terminar)")
    print("======================================")

    while True:
        pregunta = input("\nPregunta: ").strip()

        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break

        print()
        preguntar(chain, pregunta, retriever)


if __name__ == "__main__":
    main()
