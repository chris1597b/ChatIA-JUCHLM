import os
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_juchlm import (
    construir_chain,
    preguntar_stream,
    sincronizar_carpeta,
    PDF_FOLDER,
    cargar_manifest
)

app = FastAPI(title="ChatJUCHLM API")

# Initialize RAG components
# We ensure the folder is synced when the app starts
sincronizar_carpeta()
chain, retriever = construir_chain()

templates = Jinja2Templates(directory="templates")
os.makedirs("pdfs", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse(request=request, name="chat.html")

@app.get("/documentos", response_class=HTMLResponse)
async def documents_interface(request: Request):
    return templates.TemplateResponse(request=request, name="documents.html")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    if not question:
        return {"error": "Question is required"}

    def event_generator():
        for chunk in preguntar_stream(chain, question, retriever):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are allowed"}
        
    file_path = PDF_FOLDER / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # Trigger synchronization
    sincronizar_carpeta()
    return {"message": "File uploaded and processed successfully"}

@app.get("/api/documents")
async def list_documents():
    manifest = cargar_manifest()
    docs = []
    for nombre, info in manifest.items():
        docs.append({
            "name": nombre,
            "chunks": info.get("chunks", 0),
            "ocr": info.get("ocr", False)
        })
    return {"documents": docs}

@app.delete("/api/delete/{filename}")
async def delete_document(filename: str):
    file_path = PDF_FOLDER / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Archivo no encontrado"})
    
    try:
        os.remove(file_path)
        # sincronizar_carpeta detects missing files and removes them from ChromaDB + manifest
        sincronizar_carpeta()
        return {"message": f"'{filename}' eliminado correctamente de disco y ChromaDB"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error al eliminar: {str(e)}"})

@app.get("/api/download/{filename}")
async def download_document(filename: str):
    file_path = PDF_FOLDER / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Archivo no encontrado"})
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/pdf"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
