from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import backend services
from backend.ingestion.file_scanner import FileScanner
from backend.orchestration.query_graph import QueryPipeline
from backend.orchestration.ingestion_graph import IngestionPipeline
from backend.utils.file_utils import FileUtils
from backend.config.settings import settings

app = FastAPI(title="AI File Explorer API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = QueryPipeline()
ingestion = IngestionPipeline()
scanner = FileScanner()
file_utils = FileUtils()


class TextPayload(BaseModel):
    query: str
    top_k: Optional[int] = None
    rerank: bool = False

class ReindexPayload(BaseModel):
    path: Optional[str] = None

class DeletePayload(BaseModel):
    path: str
    delete_file: bool = True


def _format_result(item: Dict, fallback_type: str = "text") -> Dict:
    path = item.get("file_path") or item.get("path") or item.get("image") or ""
    name = item.get("name") or (Path(path).name if path else "Result")
    similarity = item.get("similarity") or 0.0
    summary = (
        item.get("summary")
        or item.get("chunk_content")
        or item.get("content_preview")
        or item.get("content")
        or ""
    )
    tags = item.get("tags") or []
    result_type = item.get("type") or fallback_type

    return {
        "name": name,
        "path": path,
        "type": result_type,
        "similarity": similarity,
        "summary": summary,
        "tags": tags,
        "size": item.get("size", 0),
    }


@app.get("/api/files")
def list_files(root: Optional[str] = None):
    indexed_map = {}
    stores = [
        getattr(pipeline.store_manager, "faiss_text", None),
        getattr(pipeline.store_manager, "faiss_images", None),
        getattr(pipeline.store_manager, "faiss_faces", None),
        getattr(pipeline.store_manager, "faiss_objects", None),
    ]
    for store in stores:
        if store is None:
            continue
        metadata_iter = []
        if hasattr(store, "metadata_map"):
            metadata_iter = list(store.metadata_map.values())
        elif hasattr(store, "metadata_store"):
            metadata_iter = getattr(store, "metadata_store", [])
        for meta in metadata_iter:
            path = meta.get("file_path") or meta.get("image") or meta.get("path")
            if not path:
                continue
            try:
                p = Path(path).expanduser().resolve()
            except Exception:
                p = Path(path)
            key = str(p)
            if key in indexed_map:
                continue
            indexed_map[str(p)] = {
                "name": Path(key).name,
                "path": key,
                "type": meta.get("type") or file_utils.get_file_type(p),
                "size": p.stat().st_size if p.exists() else meta.get("size", 0),
            }
    if indexed_map:
        return {"files": list(indexed_map.values())}

    scan_root = Path(root) if root else settings.data_dir
    if not scan_root.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {scan_root}")
    all_files, _, _ = scanner.scan_folder(str(scan_root))
    for f in all_files:
        try:
            p = Path(f).expanduser().resolve()
        except Exception:
            p = Path(f)
        key = str(p)
        if key in indexed_map:
            continue
        indexed_map[key] = {
            "name": p.name,
            "path": key,
            "type": file_utils.get_file_type(p),
            "size": p.stat().st_size if p.exists() else 0,
        }
    return {"files": list(indexed_map.values())}


@app.post("/api/search/text")
def search_text(payload: TextPayload):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    result = pipeline.run(query)
    reranked = (result.get("reranked_results") or [])[: payload.top_k or settings.top_k_results]
    return {"results": [_format_result(r, "text") for r in reranked], "answer": result.get("final_response")}


@app.post("/api/search/text-image")
def search_text_image(payload: TextPayload):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    result = pipeline.run(query)
    reranked = (result.get("reranked_results") or [])[: payload.top_k or settings.top_k_results]
    return {"results": [_format_result(r, "image") for r in reranked], "answer": result.get("final_response")}


@app.post("/api/search/object")
def search_object(payload: TextPayload):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    result = pipeline.run(query, query_type="object_search", skip_llm=True)
    reranked = (result.get("reranked_results") or [])[: payload.top_k or settings.top_k_results]
    return {"results": [_format_result(r, "object") for r in reranked], "answer": result.get("final_response")}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), top_k: Optional[int] = Form(None)):
    data = await file.read()
    try:
        # Use direct image embedding + FAISS search to avoid graph aggregation conflicts on pure image queries
        embedding = pipeline.embedding_manager.generate_image_embedding_from_bytes(data)
        results = pipeline.store_manager.search_images(embedding, top_k or settings.top_k_results)
        return {"results": [_format_result(r, "image") for r in results], "answer": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image search failed: {exc}")


@app.post("/api/search/face")
async def search_face(file: UploadFile = File(...), top_k: Optional[int] = Form(None)):
    data = await file.read()
    result = pipeline.run("face search", data)
    reranked = (result.get("reranked_results") or [])[: top_k or settings.top_k_results]
    return {"results": [_format_result(r, "face") for r in reranked], "answer": result.get("final_response")}


@app.post("/api/reindex")
def reindex(payload: ReindexPayload | None = None):
    target = Path(payload.path) if payload and payload.path else settings.data_dir
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    try:
        result = ingestion.run(str(target)) or {}
        # Build a JSON-safe summary to avoid serialization errors
        summary = {
            "files_processed": result.get("files_processed"),
            "total_files": result.get("total_files"),
            "errors": [str(e) for e in result.get("errors", [])],
        }
        return {"status": "completed", "path": str(target), "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.get("/api/file")
def serve_file(path: str):
    """Serve a file (e.g., image) from an absolute path."""
    decoded = unquote(path)
    file_path = Path(decoded)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        return FileResponse(file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read file: {exc}")


@app.post("/api/files/delete")
def delete_file(payload: DeletePayload):
    target = Path(payload.path)
    deleted_embeddings = pipeline.store_manager.delete_path(str(target))
    file_deleted = False
    if payload.delete_file and target.exists():
        try:
            target.unlink()
            file_deleted = True
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {exc}")
    return {"deleted_embeddings": deleted_embeddings, "file_deleted": file_deleted}


@app.post("/api/files/add")
async def add_file(file: UploadFile = File(...)):
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / file.filename
    data = await file.read()
    try:
        with open(dest, "wb") as f:
            f.write(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")
    try:
        ingestion.run(str(uploads_dir))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    return {"path": str(dest)}


# Serve built React
frontend_dist = Path(__file__).parent.parent / "frontend_web" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not found")
