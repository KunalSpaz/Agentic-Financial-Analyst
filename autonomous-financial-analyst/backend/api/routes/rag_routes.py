"""
rag_routes.py
-------------
FastAPI routes for RAG document management.

Endpoints:
    POST /documents/ingest        — Ingest a document from JSON body text
    POST /documents/ingest-file   — Ingest a .txt file upload
    GET  /documents               — List all indexed documents
    GET  /documents/search        — Query the RAG knowledge base
    DELETE /documents/{doc_id}    — Remove a document record
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.financial_document import FinancialDocument
from backend.services.rag_service import RAGService
from backend.utils.logger import get_logger

router = APIRouter(prefix="/documents", tags=["Document Library"])
logger = get_logger(__name__)

_rag = RAGService()

_VALID_DOC_TYPES = {"earnings_transcript", "annual_report", "analyst_note", "news_article", "research_report", "other"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class IngestDocumentRequest(BaseModel):
    """Request body for text-based document ingestion."""
    title: str = Field(..., min_length=3, max_length=512)
    content: str = Field(..., min_length=50)
    ticker: Optional[str] = Field(default=None, min_length=1, max_length=10)
    doc_type: str = Field(default="other")
    source: Optional[str] = Field(default=None, max_length=256)
    document_date: Optional[str] = Field(default=None, max_length=32)


@router.post("/ingest", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def ingest_document(
    request: Request,
    body: IngestDocumentRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Ingest a financial document from raw text into the RAG knowledge base.

    Chunks the content, embeds it via OpenAI, adds it to the FAISS index,
    and persists the document record to the database.
    """
    doc_type = body.doc_type if body.doc_type in _VALID_DOC_TYPES else "other"
    ticker = body.ticker.upper() if body.ticker else None

    chunks_added = await asyncio.to_thread(_rag.add_documents, [{
        "content": body.content,
        "title": body.title,
        "ticker": ticker or "",
        "doc_type": doc_type,
    }])

    if chunks_added == 0:
        raise HTTPException(
            status_code=422,
            detail="Document could not be embedded. Check your OPENAI_API_KEY and content length.",
        )

    try:
        doc = FinancialDocument(
            ticker=ticker,
            doc_type=doc_type,
            title=body.title,
            content=body.content,
            source=body.source,
            document_date=body.document_date,
            faiss_indexed=True,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id
    except Exception as exc:
        logger.warning("Failed to persist document to DB: %s", exc)
        doc_id = None

    logger.info("Ingested document '%s' — %d chunks added to FAISS.", body.title, chunks_added)
    return {
        "status": "success",
        "doc_id": doc_id,
        "title": body.title,
        "ticker": ticker,
        "doc_type": doc_type,
        "chunks_added": chunks_added,
        "total_index_size": _rag._index.ntotal if _rag._index else 0,
    }


@router.post("/ingest-file", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def ingest_file(
    request: Request,
    file: UploadFile = File(..., description="Plain text file (.txt)"),
    title: str = Form(...),
    ticker: str = Form(default=""),
    doc_type: str = Form(default="other"),
    source: str = Form(default=""),
    document_date: str = Form(default=""),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Ingest a .txt file upload into the RAG knowledge base.

    Accepts a plain-text file plus form metadata fields.
    """
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=415, detail="Only .txt files are supported.")

    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1", errors="replace")

    if len(content.strip()) < 50:
        raise HTTPException(status_code=422, detail="File content is too short (minimum 50 characters).")

    _ticker = ticker.upper().strip() if ticker.strip() else None
    _doc_type = doc_type if doc_type in _VALID_DOC_TYPES else "other"

    chunks_added = await asyncio.to_thread(_rag.add_documents, [{
        "content": content,
        "title": title,
        "ticker": _ticker or "",
        "doc_type": _doc_type,
    }])

    if chunks_added == 0:
        raise HTTPException(
            status_code=422,
            detail="File could not be embedded. Check your OPENAI_API_KEY.",
        )

    try:
        doc = FinancialDocument(
            ticker=_ticker,
            doc_type=_doc_type,
            title=title,
            content=content,
            source=source.strip() or file.filename,
            document_date=document_date.strip() or None,
            faiss_indexed=True,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id
    except Exception as exc:
        logger.warning("Failed to persist document to DB: %s", exc)
        doc_id = None

    logger.info("Ingested file '%s' (%s) — %d chunks added.", file.filename, title, chunks_added)
    return {
        "status": "success",
        "doc_id": doc_id,
        "title": title,
        "ticker": _ticker,
        "doc_type": _doc_type,
        "filename": file.filename,
        "content_length": len(content),
        "chunks_added": chunks_added,
        "total_index_size": _rag._index.ntotal if _rag._index else 0,
    }


@router.get("/search", response_model=Dict[str, Any])
@limiter.limit("30/minute")
async def search_documents(request: Request, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the RAG knowledge base and return the most relevant document chunks.

    Args:
        query:  Natural language question or search string.
        top_k:  Number of chunks to return (1–20).
    """
    if not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    top_k = max(1, min(top_k, 20))

    if _rag._index is None or _rag._index.ntotal == 0:
        return {
            "query": query,
            "results": [],
            "message": "Knowledge base is empty. Ingest some documents first.",
        }

    context = await asyncio.to_thread(_rag.retrieve, query, top_k=top_k)
    # Split context back into individual chunks for structured response
    chunks = [c.strip() for c in context.split("\n\n---\n\n") if c.strip()]

    return {
        "query": query,
        "top_k": top_k,
        "chunks_returned": len(chunks),
        "results": chunks,
        "total_index_size": _rag._index.ntotal,
    }


@router.get("", response_model=List[Dict[str, Any]])
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    limit: int = 50,
    ticker: Optional[str] = None,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    List all documents stored in the database.

    Args:
        limit:  Maximum number of documents to return (max 200).
        ticker: Optional filter by ticker symbol.
    """
    limit = max(1, min(limit, 200))
    query = db.query(FinancialDocument).order_by(FinancialDocument.created_at.desc())
    if ticker:
        query = query.filter(FinancialDocument.ticker == ticker.upper())
    docs = query.limit(limit).all()

    return [
        {
            "id": d.id,
            "ticker": d.ticker,
            "doc_type": d.doc_type,
            "title": d.title,
            "source": d.source,
            "document_date": d.document_date,
            "faiss_indexed": d.faiss_indexed,
            "content_preview": d.content[:300] + "…" if len(d.content) > 300 else d.content,
            "content_length": len(d.content),
            "created_at": str(d.created_at),
        }
        for d in docs
    ]


@router.delete("/{doc_id}", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def delete_document(request: Request, doc_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Remove a document record from the database.

    Note: FAISS does not support per-vector deletion. The embeddings remain
    in the index but will no longer appear in document listings. Rebuild the
    index via re-ingestion to fully remove the document from retrieval.
    """
    doc = db.query(FinancialDocument).filter(FinancialDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    title = doc.title
    db.delete(doc)
    db.commit()
    logger.info("Deleted document id=%d title='%s'", doc_id, title)

    return {
        "status": "deleted",
        "doc_id": doc_id,
        "title": title,
        "note": "Document removed from DB. FAISS embeddings persist until index rebuild.",
    }
