"""
rag_service.py
--------------
Retrieval-Augmented Generation (RAG) pipeline using FAISS + OpenAI embeddings.

Documents are chunked, embedded via OpenAI, stored in a FAISS index, and
retrieved at query time to augment LLM reasoning.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import faiss
import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.utils.config import settings
from backend.utils.logger import get_logger

_EMBED_RETRY = dict(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)

logger = get_logger(__name__)

CHUNK_SIZE = 512        # chars per chunk
CHUNK_OVERLAP = 64      # overlap between consecutive chunks
EMBEDDING_DIM = 1536    # text-embedding-3-small dimension


class RAGService:
    """
    Manages a FAISS index of financial document embeddings.

    Usage::

        svc = RAGService()
        svc.add_documents([{"content": "...", "title": "Q4 Earnings", "ticker": "AAPL"}])
        context = svc.retrieve("What was AAPL revenue growth?")
    """

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: List[Dict[str, Any]] = []
        self._index_path = settings.faiss_index_path
        self._lock = threading.Lock()  # guards index + metadata mutations
        self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Load persisted FAISS index and metadata from disk if available."""
        idx_file = f"{self._index_path}.idx"
        meta_file = f"{self._index_path}.meta"
        if os.path.exists(idx_file) and os.path.exists(meta_file):
            try:
                self._index = faiss.read_index(idx_file)
                with open(meta_file, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                logger.info("FAISS index loaded: %d vectors.", self._index.ntotal)
            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s — starting fresh.", exc)
                self._initialise_index()
        else:
            self._initialise_index()

    def _initialise_index(self) -> None:
        self._index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self._metadata = []

    def _save_index(self) -> None:
        """Persist FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, f"{self._index_path}.idx")
        with open(f"{self._index_path}.meta", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Chunk, embed, and index a list of financial documents.

        Args:
            documents: List of dicts with at least ``content`` key.
                       Optional keys: ``title``, ``ticker``, ``doc_type``.

        Returns:
            Number of new chunks added to the index.
        """
        chunks = []
        for doc in documents:
            text = doc.get("content", "")
            doc_chunks = self._chunk_text(text)
            for chunk_text in doc_chunks:
                chunks.append({
                    "text": chunk_text,
                    "title": doc.get("title", ""),
                    "ticker": doc.get("ticker", ""),
                    "doc_type": doc.get("doc_type", ""),
                })

        if not chunks:
            return 0

        embeddings = self._embed_texts([c["text"] for c in chunks])
        if embeddings is None:
            return 0

        with self._lock:
            self._index.add(embeddings)
            self._metadata.extend(chunks)
            self._save_index()
            total = self._index.ntotal

        logger.info("Added %d chunks to FAISS index (total=%d).", len(chunks), total)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve the most relevant document chunks for *query*.

        Args:
            query: Natural language question or search string.
            top_k: Number of chunks to retrieve. Defaults to ``settings.rag_top_k``.

        Returns:
            Concatenated context string from the top-k most similar chunks.
        """
        k = top_k or settings.rag_top_k
        if self._index is None or self._index.ntotal == 0:
            return "No financial documents available in the knowledge base."

        query_vec = self._embed_texts([query])
        if query_vec is None:
            return ""

        with self._lock:
            distances, indices = self._index.search(query_vec, min(k, self._index.ntotal))
            metadata_snapshot = list(self._metadata)

        context_parts = []
        for idx in indices[0]:
            if idx < len(metadata_snapshot):
                meta = metadata_snapshot[idx]
                text = meta.get("text", "")
                if not text:
                    continue
                header = f"[{meta.get('ticker', '')} | {meta.get('title', '')}]"
                context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Call OpenAI Embeddings API and return float32 numpy array."""
        try:
            @retry(**_EMBED_RETRY)
            def _call():
                return self._client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=texts,
                )

            response = _call()
            vectors = [item.embedding for item in response.data]
            return np.array(vectors, dtype=np.float32)
        except Exception as exc:
            logger.error("OpenAI embedding error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Split *text* into overlapping chunks of CHUNK_SIZE characters."""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
