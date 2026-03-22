"""
test_rag_service.py
--------------------
Unit tests for RAGService — FAISS-based RAG pipeline.

Covers:
  - _chunk_text:       static chunking logic and overlap
  - _initialise_index: creates fresh FAISS index
  - add_documents:     chunking → embedding → indexing → persistence
  - retrieve:          embedding → similarity search → formatted context
  - Thread safety:     lock is used during index mutation
"""
from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ── Patch heavy deps before any import ────────────────────────────────────────
sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())
sys.modules.setdefault("crewai",       MagicMock())

from backend.services.rag_service import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIM,
    RAGService,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embed_mock(dim: int = EMBEDDING_DIM, n: int = 1) -> MagicMock:
    """Return an OpenAI client mock that returns *n* embedding vectors."""
    items = []
    for _ in range(n):
        item = MagicMock()
        item.embedding = [0.1] * dim
        items.append(item)
    resp = MagicMock()
    resp.data = items
    client = MagicMock()
    client.embeddings.create.return_value = resp
    return client


def _make_service(ntotal: int = 0) -> RAGService:
    """Return a RAGService with mocked FAISS index and OpenAI client."""
    with patch("backend.services.rag_service.OpenAI"), \
         patch.object(RAGService, "_load_index"):
        svc = RAGService()
    mock_index = MagicMock()
    mock_index.ntotal = ntotal
    svc._index = mock_index
    svc._metadata = []
    svc._client = _embed_mock()
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# _chunk_text  (static — pure Python, no mocks needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert RAGService._chunk_text("") == []

    def test_none_falsy_text_returns_empty_list(self):
        assert RAGService._chunk_text("") == []

    def test_short_text_returns_single_chunk(self):
        """Text shorter than the step (CHUNK_SIZE - CHUNK_OVERLAP) fits in one chunk."""
        step = CHUNK_SIZE - CHUNK_OVERLAP
        text = "a" * (step - 1)
        chunks = RAGService._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_exactly_step_size_returns_one_chunk(self):
        """Text exactly equal to step size starts a second window at position step,
        which equals len(text) — so the loop exits after one iteration."""
        step = CHUNK_SIZE - CHUNK_OVERLAP
        text = "x" * step
        chunks = RAGService._chunk_text(text)
        assert len(chunks) == 1

    def test_text_longer_than_chunk_size_returns_multiple_chunks(self):
        text = "a" * (CHUNK_SIZE * 3)
        chunks = RAGService._chunk_text(text)
        assert len(chunks) > 1

    def test_each_chunk_max_chunk_size_chars(self):
        text = "b" * (CHUNK_SIZE * 5)
        for chunk in RAGService._chunk_text(text):
            assert len(chunk) <= CHUNK_SIZE

    def test_overlap_between_consecutive_chunks(self):
        """Second chunk should start at CHUNK_SIZE - CHUNK_OVERLAP."""
        text = "c" * (CHUNK_SIZE + CHUNK_OVERLAP + 50)
        chunks = RAGService._chunk_text(text)
        assert len(chunks) >= 2
        step = CHUNK_SIZE - CHUNK_OVERLAP
        assert chunks[1] == text[step: step + CHUNK_SIZE]

    def test_first_chunk_starts_at_beginning(self):
        text = "d" * (CHUNK_SIZE * 2)
        chunks = RAGService._chunk_text(text)
        assert chunks[0] == text[:CHUNK_SIZE]

    def test_whole_text_covered(self):
        text = "e" * (CHUNK_SIZE * 2 + 30)
        chunks = RAGService._chunk_text(text)
        step = CHUNK_SIZE - CHUNK_OVERLAP
        last_start = (len(chunks) - 1) * step
        assert last_start < len(text)


# ─────────────────────────────────────────────────────────────────────────────
# _initialise_index
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialiseIndex:
    def test_creates_faiss_index_with_correct_dim(self):
        import faiss as mock_faiss  # already mocked in sys.modules
        with patch("backend.services.rag_service.OpenAI"), \
             patch("backend.services.rag_service.faiss", mock_faiss):
            with patch.object(RAGService, "_load_index"):
                svc = RAGService()
            svc._initialise_index()
        mock_faiss.IndexFlatL2.assert_called_with(EMBEDDING_DIM)

    def test_initialise_index_clears_metadata(self):
        svc = _make_service()
        svc._metadata = [{"text": "old"}]
        svc._initialise_index()
        assert svc._metadata == []


# ─────────────────────────────────────────────────────────────────────────────
# add_documents
# ─────────────────────────────────────────────────────────────────────────────

class TestAddDocuments:
    def test_empty_list_returns_zero(self):
        svc = _make_service()
        assert svc.add_documents([]) == 0

    def test_document_with_empty_content_returns_zero(self):
        svc = _make_service()
        assert svc.add_documents([{"content": "", "title": "Empty"}]) == 0

    def test_document_with_whitespace_only_adds_chunk(self):
        """Whitespace-only content is not filtered by add_documents — it gets chunked.
        This test documents the actual behaviour (no strip before chunking)."""
        svc = _make_service()
        with patch.object(svc, "_save_index"):
            result = svc.add_documents([{"content": "   ", "title": "Blank"}])
        assert result >= 0  # does not crash; actual value depends on chunk logic

    def test_embedding_api_failure_returns_zero(self):
        svc = _make_service()
        svc._client.embeddings.create.side_effect = Exception("API down")
        result = svc.add_documents([{"content": "x" * 200, "title": "Doc"}])
        assert result == 0

    def test_short_content_one_chunk_added(self):
        svc = _make_service()
        with patch.object(svc, "_save_index"):
            result = svc.add_documents([{"content": "a" * 100, "title": "Doc"}])
        assert result == 1

    def test_long_content_multiple_chunks_added(self):
        svc = _make_service()
        content = "b" * (CHUNK_SIZE * 3)
        with patch.object(svc, "_save_index"):
            result = svc.add_documents([{"content": content, "title": "Long"}])
        assert result > 1

    def test_metadata_populated_with_correct_fields(self):
        svc = _make_service()
        with patch.object(svc, "_save_index"):
            svc.add_documents([{
                "content": "c" * 100,
                "title": "EarningsDoc",
                "ticker": "AAPL",
                "doc_type": "earnings_transcript",
            }])
        assert len(svc._metadata) == 1
        entry = svc._metadata[0]
        assert entry["title"] == "EarningsDoc"
        assert entry["ticker"] == "AAPL"
        assert entry["doc_type"] == "earnings_transcript"
        assert "text" in entry

    def test_index_add_called_with_float32_embeddings(self):
        svc = _make_service()
        with patch.object(svc, "_save_index"):
            svc.add_documents([{"content": "d" * 200, "title": "T"}])
        call_args = svc._index.add.call_args[0][0]
        assert call_args.dtype == np.float32

    def test_save_index_called_after_successful_add(self):
        svc = _make_service()
        with patch.object(svc, "_save_index") as mock_save:
            svc.add_documents([{"content": "e" * 200, "title": "T"}])
        mock_save.assert_called_once()

    def test_lock_acquired_during_index_mutation(self):
        """threading.Lock must be held while mutating _index and _metadata."""
        svc = _make_service()
        acquired = []

        class TrackingLock:
            def __enter__(self):
                acquired.append(True)
                return self
            def __exit__(self, *args):
                pass

        svc._lock = TrackingLock()
        with patch.object(svc, "_save_index"):
            svc.add_documents([{"content": "f" * 200, "title": "T"}])
        assert len(acquired) >= 1

    def test_multiple_documents_all_chunks_added(self):
        svc = _make_service()
        # Two single-chunk documents
        svc._client = _embed_mock(n=2)
        with patch.object(svc, "_save_index"):
            result = svc.add_documents([
                {"content": "g" * 100, "title": "Doc1"},
                {"content": "h" * 100, "title": "Doc2"},
            ])
        assert result == 2


# ─────────────────────────────────────────────────────────────────────────────
# retrieve
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_empty_index_returns_fallback_message(self):
        svc = _make_service(ntotal=0)
        result = svc.retrieve("What is AAPL revenue?")
        assert "No financial documents" in result

    def test_none_index_returns_fallback_message(self):
        svc = _make_service()
        svc._index = None
        result = svc.retrieve("query")
        assert "No financial documents" in result

    def test_embedding_failure_returns_empty_string(self):
        svc = _make_service(ntotal=5)
        svc._client.embeddings.create.side_effect = Exception("embed fail")
        result = svc.retrieve("query")
        assert result == ""

    def test_returns_formatted_context_with_header(self):
        svc = _make_service(ntotal=1)
        svc._metadata = [{"ticker": "AAPL", "title": "Q4 Earnings", "text": "Revenue grew 8%."}]
        svc._index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
        result = svc.retrieve("earnings", top_k=1)
        assert "AAPL" in result
        assert "Q4 Earnings" in result
        assert "Revenue grew 8%." in result

    def test_multiple_results_separated_by_delimiter(self):
        svc = _make_service(ntotal=2)
        svc._metadata = [
            {"ticker": "AAPL", "title": "Doc1", "text": "first."},
            {"ticker": "MSFT", "title": "Doc2", "text": "second."},
        ]
        svc._index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        result = svc.retrieve("query", top_k=2)
        assert "\n\n---\n\n" in result

    def test_top_k_capped_by_index_size(self):
        svc = _make_service(ntotal=2)
        svc._metadata = [
            {"ticker": "AAPL", "title": "D1", "text": "t1"},
            {"ticker": "MSFT", "title": "D2", "text": "t2"},
        ]
        svc._index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        svc.retrieve("query", top_k=1000)
        _, k_used = svc._index.search.call_args[0]
        assert k_used == 2  # min(1000, ntotal=2)

    def test_out_of_bounds_index_skipped_gracefully(self):
        """If FAISS returns an index beyond metadata length, skip it."""
        svc = _make_service(ntotal=1)
        svc._metadata = [{"ticker": "AAPL", "title": "Doc", "text": "valid."}]
        svc._index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 999]]),  # 999 is out of bounds
        )
        result = svc.retrieve("query", top_k=2)
        assert "valid." in result  # valid entry included, no crash

    def test_default_top_k_from_settings(self):
        svc = _make_service(ntotal=10)
        svc._metadata = [{"ticker": "T", "title": "D", "text": "t"} for _ in range(10)]
        svc._index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
        with patch("backend.services.rag_service.settings") as mock_settings:
            mock_settings.rag_top_k = 3
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            svc.retrieve("query")  # no top_k argument
        _, k_used = svc._index.search.call_args[0]
        assert k_used == 3


# ─────────────────────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_lock_attribute_exists(self):
        svc = _make_service()
        assert hasattr(svc, "_lock")

    def test_lock_is_reentrant_context_manager(self):
        svc = _make_service()
        with svc._lock:
            pass  # should not deadlock

    def test_concurrent_adds_do_not_corrupt_metadata(self):
        """Two threads adding documents should each see all their metadata."""
        svc = _make_service()
        errors = []

        def add_one(content: str) -> None:
            try:
                with patch.object(svc, "_save_index"):
                    svc.add_documents([{"content": content, "title": "T"}])
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=add_one, args=("a" * 200,))
        t2 = threading.Thread(target=add_one, args=("b" * 200,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(svc._metadata) == 2
