from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from services.ingest_service import ingest_document
from services.chroma_service import list_documents, delete_document_by_source, collection_count
import config

router = APIRouter()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    overwrite: bool = Form(True),
):
    """
    Upload and ingest a document (PDF, TXT, MD, DOCX, CSV).
    Supports English and Hindi content.
    """
    import os
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {list(config.ALLOWED_EXTENSIONS)}"
        )

    try:
        data = await file.read()
        result = ingest_document(
            file_data=data,
            filename=filename,
            language=language,
            overwrite=overwrite,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/documents")
async def get_documents():
    """List all uploaded documents with chunk counts."""
    docs = list_documents()
    return {
        "documents": docs,
        "total_chunks": collection_count(),
        "total_documents": len(docs),
    }


@router.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """Delete a document and all its chunks from ChromaDB."""
    deleted = delete_document_by_source(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    return {"message": f"✅ Deleted '{filename}' ({deleted} chunks removed)", "chunks_deleted": deleted}
