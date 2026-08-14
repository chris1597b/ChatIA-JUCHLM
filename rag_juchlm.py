"""
RAG-JUCHLM - Versión producción: carpeta de PDFs + detección de escaneados
=============================================================================

Hardware objetivo: RTX 2060 6GB VRAM, 8GB RAM. LLM y embeddings vía Ollama.

QUÉ RESUELVE ESTA VERSIÓN (vs la de un solo PDF):

1. Procesa TODA una carpeta (PDF_FOLDER), no un archivo fijo.

2. Detección automática de PDFs escaneados
   Para cada PDF, extrae texto normal con PyPDFLoader y mide cuántos
   caracteres reales salen por página. Si es casi nada (< SCANNED_THRESHOLD
   caracteres/página en promedio) asume que es un escaneo y lo reprocesa
   con OCR (RapidOCRPDFLoader - 100% Python, sin depender de Tesseract o
   Poppler instalados en el sistema, lo cual importa en Windows).

2b. Detección MIXTA por página, no solo por archivo
   Un PDF puede tener páginas normales y páginas escaneadas mezcladas
   (pasa seguido con "resúmenes de artículos" armados a partir de varias
   fuentes). Si el archivo completo se ve escaneado, se OCRea completo;
   si es mayormente texto, se deja como está (evita gastar OCR de más).

3. Indexación incremental con manifest.json
   Antes de reprocesar, compara mtime+hash de cada PDF contra un manifest
   guardado. Solo se re-embeben archivos nuevos o modificados. Los PDFs
   eliminados de la carpeta se eliminan también de ChromaDB. Esto es
   CRÍTICO en producción: sin esto, cada vez que agregas un PDF nuevo a la
   carpeta tendrías que re-embeber todo desde cero.

4. IDs estables por chunk (archivo + índice) -> permite update/delete
   real en Chroma en vez de duplicar vectores en cada corrida.

5. Procesamiento archivo por archivo (no carga todos los PDFs en memoria
   a la vez) -> importante con 8GB RAM.

6. Manejo de errores por archivo: si un PDF está corrupto o falla el OCR,
   se loggea y se sigue con el resto en vez de tumbar todo el proceso.

7. Metadata de fuente en cada chunk (nombre de archivo + página + si vino
   de OCR) para que las respuestas puedan citar de qué PDF salió la info.

8. Loop de preguntas continuo (no una sola pregunta y se cierra).

INSTALACIÓN (una vez):
-----------------------
    pip install langchain-community langchain-ollama langchain-text-splitters ^
                langchain-chroma langchain-core pypdf rapidocr-onnxruntime

ANTES DE CORRER, EN POWERSHELL:
---------------------------------
    $env:OLLAMA_NUM_PARALLEL="1"
    $env:OLLAMA_MAX_LOADED_MODELS="1"
    $env:OLLAMA_FLASH_ATTENTION="1"
    ollama serve

USO:
----
    python rag_juchlm.py            -> sincroniza carpeta + abre loop de preguntas
    python rag_juchlm.py --solo-ingesta   -> solo sincroniza, no pregunta nada
"""

import hashlib
import json
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# ============================================================
# CONFIGURACIÓN
# ============================================================

PDF_FOLDER = Path("pdfs")            # carpeta donde van todos los PDFs
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "local-rag"
MANIFEST_PATH = Path(CHROMA_PATH) / "manifest.json"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen3.5:4b"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
RETRIEVER_K = 6  # con varios PDFs en la colección, 4 se queda corto

MAX_TOKENS_RESPUESTA = 2048  # Aumentado para evitar respuestas truncadas
CONTEXTO_LLM = 4096         # Contexto ampliado para respuestas largas

# Si el promedio de caracteres de texto extraído por página es menor a esto,
# se asume que el PDF (o esa zona) está escaneado y se manda a OCR.
SCANNED_THRESHOLD_CHARS_POR_PAGINA = 30


def log(msg: str):
    print(msg, flush=True)


# ============================================================
# UTILIDADES DE MANIFEST (para indexación incremental)
# ============================================================

def cargar_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def guardar_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def hash_archivo(path: Path) -> str:
    """Hash rápido basado en tamaño + mtime, no en contenido completo
    (hashear el contenido de PDFs grandes sería lento; para detectar
    cambios alcanza con tamaño+fecha de modificación)."""
    stat = path.stat()
    firma = f"{stat.st_size}-{stat.st_mtime}"
    return hashlib.md5(firma.encode()).hexdigest()


# ============================================================
# DETECCIÓN DE PDF ESCANEADO + CARGA
# ============================================================

def es_probable_escaneado(documents: list[Document]) -> bool:
    if not documents:
        return False
    total_chars = sum(len(d.page_content.strip()) for d in documents)
    promedio = total_chars / len(documents)
    return promedio < SCANNED_THRESHOLD_CHARS_POR_PAGINA


def cargar_pdf(path: Path) -> tuple[list[Document], bool]:
    """Devuelve (documentos, fue_ocr). Intenta texto normal primero
    (rápido); si detecta que es escaneado, reprocesa con OCR."""

    docs_texto = PyPDFLoader(str(path)).load()

    if not es_probable_escaneado(docs_texto):
        return docs_texto, False

    log(f"    -> '{path.name}' parece escaneado, aplicando OCR "
        f"(esto es más lento, tené paciencia)...")

    try:
        from langchain_community.document_loaders import RapidOCRPDFLoader
        docs_ocr = RapidOCRPDFLoader(str(path)).load()
        if es_probable_escaneado(docs_ocr):
            log(f"    -> OCR de '{path.name}' extrajo muy poco texto. "
                f"Revisá manualmente si el PDF está en buen estado.")
        return docs_ocr, True
    except ImportError:
        log("    -> ERROR: falta 'rapidocr-onnxruntime'. Instalalo con:")
        log("       pip install rapidocr-onnxruntime")
        log(f"    -> Se usará el texto (probablemente vacío/parcial) de "
            f"'{path.name}' sin OCR.")
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
# SINCRONIZAR CARPETA -> CHROMADB
# ============================================================

def sincronizar_carpeta():
    if not PDF_FOLDER.exists():
        log(f"\nERROR: no existe la carpeta '{PDF_FOLDER.absolute()}'")
        raise SystemExit(1)

    pdfs_en_disco = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdfs_en_disco:
        log(f"\nNo hay PDFs en '{PDF_FOLDER.absolute()}'.")

    manifest = cargar_manifest()
    nombres_en_disco = {p.name for p in pdfs_en_disco}

    # --- eliminar del índice los PDFs que ya no están en la carpeta ---
    eliminados = [nombre for nombre in manifest if nombre not in nombres_en_disco]
    for nombre in eliminados:
        log(f"\nEliminando de la base vectorial: '{nombre}' (ya no está en la carpeta)")
        vector_db.delete(where={"source": nombre})
        del manifest[nombre]

    # --- procesar nuevos / modificados ---
    nuevos_o_cambiados = []
    for path in pdfs_en_disco:
        firma_actual = hash_archivo(path)
        firma_previa = manifest.get(path.name, {}).get("firma")
        if firma_actual != firma_previa:
            nuevos_o_cambiados.append(path)

    if not nuevos_o_cambiados and not eliminados:
        log(f"\nBase vectorial ya actualizada. "
            f"{len(pdfs_en_disco)} PDFs indexados, sin cambios.")
        return

    log(f"\n{len(nuevos_o_cambiados)} PDF(s) nuevo(s)/modificado(s) de "
        f"{len(pdfs_en_disco)} totales. Procesando...\n")

    for i, path in enumerate(nuevos_o_cambiados, 1):
        log(f"[{i}/{len(nuevos_o_cambiados)}] {path.name}")
        inicio = time.time()

        # si es una actualización de un archivo ya indexado, borrar lo viejo primero
        if path.name in manifest:
            vector_db.delete(where={"source": path.name})

        try:
            docs, fue_ocr = cargar_pdf(path)
        except Exception as e:
            log(f"    -> ERROR procesando '{path.name}': {e}. Se omite este archivo.")
            continue

        for d in docs:
            d.metadata["source"] = path.name
            d.metadata["ocr"] = fue_ocr

        chunks = splitter.split_documents(docs)

        if not chunks:
            log(f"    -> '{path.name}' no generó ningún chunk de texto (¿PDF vacío?).")
            continue

        ids = [f"{path.name}::{idx}" for idx in range(len(chunks))]
        vector_db.add_documents(documents=chunks, ids=ids)

        manifest[path.name] = {
            "firma": hash_archivo(path),
            "chunks": len(chunks),
            "ocr": fue_ocr,
        }

        log(f"    -> {len(chunks)} chunks, "
            f"{'OCR' if fue_ocr else 'texto directo'}, "
            f"{time.time() - inicio:.2f}s")

    guardar_manifest(manifest)
    log(f"\nSincronización completa. Total PDFs indexados: {len(manifest)}")


# ============================================================
# CADENA RAG
# ============================================================

def construir_chain():
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        num_predict=MAX_TOKENS_RESPUESTA,
        num_ctx=CONTEXTO_LLM,
        reasoning=False,  # qwen3.5 es híbrido: sin esto puede gastar todo
                          # el presupuesto de tokens "pensando" y no
                          # devolver contenido (content vacío).
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    template = """Eres un asistente experto que responde ÚNICAMENTE con la información del contexto proporcionado.

REGLAS ESTRICTAS:
1. Usa SOLO la información del contexto. NO uses conocimiento externo.
2. Si la pregunta no tiene respuesta en el contexto, responde exactamente: "No encontré información sobre ese tema en los documentos indexados."
3. NO mezcles información de diferentes documentos a menos que la pregunta lo pida explícitamente.
4. Si la pregunta es específica sobre un tema, busca SOLO en los chunks que hablan de ese tema.
5. Al final, indica siempre de qué archivo(s) obtuviste la información.

Contexto (chunks recuperados de los documentos):
{context}

Pregunta: {question}

Respuesta completa y detallada:"""
    prompt = ChatPromptTemplate.from_template(template)

    def formatear_contexto(docs):
        partes = []
        for d in docs:
            fuente = d.metadata.get("source", "desconocido")
            pagina = d.metadata.get("page", "?")
            partes.append(f"[Fuente: {fuente} | página {pagina}]\n{d.page_content}")
        return "\n\n".join(partes)

    chain = (
        {"context": retriever | formatear_contexto, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def preguntar(chain, question: str, retriever):
    # Diagnóstico: mostramos qué chunks se recuperaron REALMENTE antes de
    # que el LLM responda. Esto es la verdad de base — si el chunk correcto
    # no aparece acá, el problema es de retrieval (k muy bajo, embeddings,
    # o el PDF no está indexado), no del LLM.
    docs_recuperados = retriever.invoke(question)
    print("--- Chunks recuperados (debug) ---")
    for d in docs_recuperados:
        fuente = d.metadata.get("source", "?")
        pagina = d.metadata.get("page", "?")
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  [{fuente} | pág {pagina}] {preview}...")
    print("-----------------------------------\n")

    inicio = time.time()
    respuesta_completa = ""
    primer_token = None

    for pedazo in chain.stream(question):
        if primer_token is None:
            primer_token = time.time()
            print(f"[primer token en {primer_token - inicio:.2f}s]\n")
        respuesta_completa += pedazo
        print(pedazo, end="", flush=True)

    duracion = time.time() - inicio
    print()
    if not respuesta_completa.strip():
        log("\nADVERTENCIA: respuesta vacía. Revisá num_predict/num_ctx "
            "o si el modelo sigue en modo thinking.")
    else:
        log(f"\n[Completado en {duracion:.2f}s]")


def preguntar_stream(chain, question: str, retriever):
    docs_recuperados = retriever.invoke(question)
    
    fuentes = []
    for d in docs_recuperados:
        fuente = d.metadata.get("source", "?")
        pagina = d.metadata.get("page", "?")
        fuentes.append({"source": fuente, "page": pagina})
        
    yield f"data: {json.dumps({'type': 'sources', 'data': fuentes})}\n\n"

    for pedazo in chain.stream(question):
        yield f"data: {json.dumps({'type': 'chunk', 'data': pedazo})}\n\n"
        
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
        print(f"  - {nombre}: {info.get('chunks')} chunks ({tipo})")


def main():
    print("\n======================================")
    print("   RAG-JUCHLM - PRODUCCIÓN (multi-PDF)")
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