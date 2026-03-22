"""
test_rag_routes.py
------------------
Unit tests for all 5 /documents endpoints.

    POST /documents/ingest
    POST /documents/ingest-file
    GET  /documents/search
    GET  /documents
    DELETE /documents/{doc_id}
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import sys

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())
sys.modules.setdefault("crewai",       MagicMock())

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import create_app  # noqa: E402
from backend.database.connection import Base, engine  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with patch("backend.api.main.run_migrations"), \
         patch("backend.api.main.start_scheduler"), \
         patch("backend.api.main.shutdown_scheduler"), \
         patch("backend.api.main._prewarm_finbert"):
        app = create_app()
        Base.metadata.create_all(bind=engine)
        with TestClient(app) as c:
            yield c
        Base.metadata.drop_all(bind=engine)


def _mock_rag(ntotal: int = 5, chunks_added: int = 3):
    """Return a mock RAGService instance."""
    mock = MagicMock()
    mock._index.ntotal = ntotal
    mock.add_documents.return_value = chunks_added
    mock.retrieve.return_value = "Chunk A\n\n---\n\nChunk B\n\n---\n\nChunk C"
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# POST /documents/ingest
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestDocument:
    VALID_BODY = {
        "title": "Apple Q4 2024 Earnings Call",
        "content": "Apple reported record revenue of $119.6 billion in Q4 2024. " * 5,
        "ticker": "AAPL",
        "doc_type": "earnings_transcript",
        "source": "apple.com",
        "document_date": "2024-11-01",
    }

    def test_returns_200_on_success(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=self.VALID_BODY)
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=self.VALID_BODY)
        data = resp.json()
        for key in ("status", "doc_id", "title", "ticker", "doc_type", "chunks_added", "total_index_size"):
            assert key in data, f"Missing key: {key}"

    def test_status_is_success(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=self.VALID_BODY)
        assert resp.json()["status"] == "success"

    def test_ticker_uppercased(self, client):
        body = {**self.VALID_BODY, "ticker": "aapl"}
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=body)
        assert resp.json()["ticker"] == "AAPL"

    def test_unknown_doc_type_defaults_to_other(self, client):
        body = {**self.VALID_BODY, "doc_type": "spaceship_report"}
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=body)
        assert resp.json()["doc_type"] == "other"

    def test_zero_chunks_returns_422(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag(chunks_added=0)):
            resp = client.post("/documents/ingest", json=self.VALID_BODY)
        assert resp.status_code == 422

    def test_short_content_returns_422(self, client):
        body = {**self.VALID_BODY, "content": "Too short."}
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=body)
        assert resp.status_code == 422

    def test_missing_title_returns_422(self, client):
        body = {k: v for k, v in self.VALID_BODY.items() if k != "title"}
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post("/documents/ingest", json=body)
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /documents/ingest-file
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestFile:
    CONTENT = b"Apple Inc. Q4 2024 earnings transcript. Revenue was $119.6 billion. " * 5

    def _upload(self, client, rag_mock, filename: str = "transcript.txt", content: bytes = None):
        data = content or self.CONTENT
        return client.post(
            "/documents/ingest-file",
            files={"file": (filename, io.BytesIO(data), "text/plain")},
            data={"title": "Apple Transcript", "ticker": "AAPL",
                  "doc_type": "earnings_transcript", "source": "SEC", "document_date": "2024-11-01"},
        )

    def test_returns_200_on_success(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = self._upload(client, _mock_rag())
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = self._upload(client, _mock_rag())
        data = resp.json()
        for key in ("status", "doc_id", "title", "chunks_added", "content_length", "filename"):
            assert key in data

    def test_non_txt_file_returns_415(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.post(
                "/documents/ingest-file",
                files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
                data={"title": "Some Report"},
            )
        assert resp.status_code == 415

    def test_too_short_file_returns_422(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = self._upload(client, _mock_rag(), content=b"Short.")
        assert resp.status_code == 422

    def test_zero_chunks_returns_422(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag(chunks_added=0)):
            resp = self._upload(client, _mock_rag(chunks_added=0))
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /documents/search
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchDocuments:
    def test_returns_200_with_results(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag(ntotal=10)):
            resp = client.get("/documents/search?query=Apple+earnings")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag(ntotal=10)):
            resp = client.get("/documents/search?query=revenue+growth")
        data = resp.json()
        assert "query" in data
        assert "results" in data
        assert "chunks_returned" in data

    def test_chunks_returned_matches_results_length(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag(ntotal=10)):
            resp = client.get("/documents/search?query=revenue&top_k=3")
        data = resp.json()
        assert data["chunks_returned"] == len(data["results"])

    def test_empty_index_returns_empty_results(self, client):
        mock = _mock_rag(ntotal=0)
        mock._index.ntotal = 0
        with patch("backend.api.routes.rag_routes._rag", mock):
            resp = client.get("/documents/search?query=Apple")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_empty_query_returns_422(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.get("/documents/search?query=")
        assert resp.status_code == 422

    def test_top_k_capped_at_20(self, client):
        mock = _mock_rag(ntotal=100)
        mock.retrieve.return_value = "\n\n---\n\n".join([f"Chunk {i}" for i in range(20)])
        with patch("backend.api.routes.rag_routes._rag", mock):
            resp = client.get("/documents/search?query=test&top_k=999")
        assert resp.status_code == 200
        # retrieve must have been called with top_k <= 20
        mock.retrieve.assert_called_once()
        args, kwargs = mock.retrieve.call_args
        actual_top_k = kwargs.get("top_k") or (args[1] if len(args) > 1 else None)
        assert actual_top_k is not None and actual_top_k <= 20


# ─────────────────────────────────────────────────────────────────────────────
# GET /documents
# ─────────────────────────────────────────────────────────────────────────────

class TestListDocuments:
    def test_returns_200_and_list(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.get("/documents")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_empty_db_returns_list(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.get("/documents")
        assert isinstance(resp.json(), list)

    def test_limit_param_accepted(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.get("/documents?limit=10")
        assert resp.status_code == 200

    def test_ticker_filter_accepted(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.get("/documents?ticker=AAPL")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /documents/{doc_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteDocument:
    def test_nonexistent_doc_returns_404(self, client):
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            resp = client.delete("/documents/99999")
        assert resp.status_code == 404

    def test_delete_returns_status_deleted(self, client):
        """Ingest a doc then delete it — should return status=deleted."""
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            ingest_resp = client.post(
                "/documents/ingest",
                json={
                    "title": "To be deleted",
                    "content": "This document will be deleted from the system. " * 5,
                    "doc_type": "other",
                },
            )
        if ingest_resp.status_code != 200:
            pytest.skip("Ingest failed, skipping delete test")
        doc_id = ingest_resp.json().get("doc_id")
        if not doc_id:
            pytest.skip("No doc_id returned from ingest")

        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            del_resp = client.delete(f"/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"
        assert del_resp.json()["doc_id"] == doc_id

    def test_double_delete_returns_404(self, client):
        """Deleting the same document twice returns 404 on second call."""
        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            ingest_resp = client.post(
                "/documents/ingest",
                json={
                    "title": "Double delete test doc",
                    "content": "Content for double delete test scenario. " * 5,
                    "doc_type": "other",
                },
            )
        if ingest_resp.status_code != 200 or not ingest_resp.json().get("doc_id"):
            pytest.skip("Ingest failed")
        doc_id = ingest_resp.json()["doc_id"]

        with patch("backend.api.routes.rag_routes._rag", _mock_rag()):
            client.delete(f"/documents/{doc_id}")
            second = client.delete(f"/documents/{doc_id}")
        assert second.status_code == 404
