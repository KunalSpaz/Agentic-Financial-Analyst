"""
8_document_library.py
----------------------
Document Library page — upload financial documents into the RAG knowledge base
so the AI agents can reference them during stock analysis.

Tabs:
    1. Paste Text      — Paste raw text + fill metadata form
    2. Upload File     — Upload a .txt file
    3. Search          — Query the knowledge base
    4. Library         — Browse and delete indexed documents
"""
from __future__ import annotations

import httpx
import streamlit as st

from components.sidebar import render_sidebar

st.set_page_config(page_title="Document Library", page_icon="📚", layout="wide")
api_url = render_sidebar()

st.title("📚 Document Library")
st.markdown(
    "Upload financial documents — earnings transcripts, analyst notes, research reports — "
    "into the AI knowledge base. Documents are chunked and embedded into the FAISS vector "
    "index so the **Stock Analysis** agent can reference them automatically."
)

# ── Index size banner ─────────────────────────────────────────────────────────
try:
    idx_resp = httpx.get(f"{api_url}/documents?limit=1", timeout=10)
    if idx_resp.status_code == 200:
        docs_list = idx_resp.json()
        count_resp = httpx.get(f"{api_url}/documents?limit=200", timeout=10)
        total_docs = len(count_resp.json()) if count_resp.status_code == 200 else "?"
        st.info(f"**Knowledge base:** {total_docs} document(s) indexed")
    else:
        st.info("Knowledge base status unknown — is the backend running?")
except Exception:
    st.info("Knowledge base status unknown — is the backend running?")

st.divider()

tab_paste, tab_file, tab_search, tab_library = st.tabs([
    "📝 Paste Text",
    "📁 Upload File",
    "🔍 Search Knowledge Base",
    "📖 Document Library",
])

# ── DOC TYPES ─────────────────────────────────────────────────────────────────
DOC_TYPES = [
    "earnings_transcript",
    "annual_report",
    "analyst_note",
    "news_article",
    "research_report",
    "other",
]

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Paste Text
# ═══════════════════════════════════════════════════════════════════════════════
with tab_paste:
    st.subheader("📝 Paste Document Text")
    st.markdown("Paste the full text of a financial document below and fill in the metadata.")

    col_form, col_help = st.columns([2, 1])

    with col_form:
        pt_title = st.text_input(
            "Document Title *",
            placeholder="e.g. Apple Q4 2024 Earnings Call Transcript",
            key="pt_title",
        )
        pt_ticker = st.text_input(
            "Ticker Symbol (optional)",
            placeholder="e.g. AAPL",
            key="pt_ticker",
        ).upper()
        col_type, col_date = st.columns(2)
        with col_type:
            pt_doc_type = st.selectbox("Document Type", DOC_TYPES, key="pt_doc_type")
        with col_date:
            pt_date = st.text_input("Document Date (optional)", placeholder="2024-11-01", key="pt_date")
        pt_source = st.text_input(
            "Source URL or Name (optional)",
            placeholder="e.g. SEC EDGAR / earnings.apple.com",
            key="pt_source",
        )
        pt_content = st.text_area(
            "Document Content *",
            height=300,
            placeholder="Paste the full document text here (minimum 50 characters)…",
            key="pt_content",
        )
        char_count = len(pt_content)
        st.caption(f"{char_count:,} characters entered")

        if st.button("⬆ Ingest Document", type="primary", key="btn_paste_ingest"):
            if not pt_title.strip():
                st.warning("Please enter a document title.")
            elif char_count < 50:
                st.warning("Content is too short — minimum 50 characters.")
            else:
                with st.spinner("Embedding and indexing document…"):
                    try:
                        resp = httpx.post(
                            f"{api_url}/documents/ingest",
                            json={
                                "title": pt_title.strip(),
                                "content": pt_content,
                                "ticker": pt_ticker.strip() or None,
                                "doc_type": pt_doc_type,
                                "source": pt_source.strip() or None,
                                "document_date": pt_date.strip() or None,
                            },
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(
                                f"✅ **'{data['title']}'** ingested successfully! "
                                f"{data['chunks_added']} chunks added to FAISS "
                                f"(index total: {data['total_index_size']} vectors)."
                            )
                        else:
                            st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    with col_help:
        st.markdown("### Tips")
        st.markdown("""
**What to upload:**
- 📊 Earnings call transcripts
- 📋 10-K / 10-Q annual reports
- 🔬 Analyst research notes
- 📰 Long-form financial news

**Why it matters:**
The AI agents retrieve relevant excerpts from these documents during stock analysis to ground their recommendations in specific company data.

**Best results:**
- Include full text, not summaries
- One document per submission
- Accurate ticker and date help the agent filter by company
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload File
# ═══════════════════════════════════════════════════════════════════════════════
with tab_file:
    st.subheader("📁 Upload a Text File")
    st.markdown("Upload a `.txt` file containing the document. Fill in the metadata fields below.")

    col_meta, col_upload = st.columns([1, 1])

    with col_meta:
        fu_title = st.text_input(
            "Document Title *",
            placeholder="e.g. Microsoft FY2024 Annual Report",
            key="fu_title",
        )
        fu_ticker = st.text_input(
            "Ticker Symbol (optional)",
            placeholder="e.g. MSFT",
            key="fu_ticker",
        ).upper()
        col_t2, col_d2 = st.columns(2)
        with col_t2:
            fu_doc_type = st.selectbox("Document Type", DOC_TYPES, key="fu_doc_type")
        with col_d2:
            fu_date = st.text_input("Document Date (optional)", placeholder="2024-07-30", key="fu_date")
        fu_source = st.text_input(
            "Source URL or Name (optional)",
            placeholder="e.g. SEC EDGAR",
            key="fu_source",
        )

    with col_upload:
        uploaded = st.file_uploader(
            "Choose a .txt file",
            type=["txt"],
            help="Plain text files only. Maximum recommended size: 5 MB.",
            key="fu_file",
        )
        if uploaded:
            raw_preview = uploaded.read(500).decode("utf-8", errors="replace")
            uploaded.seek(0)
            st.markdown("**Preview (first 500 chars):**")
            st.code(raw_preview, language=None)

    if st.button("⬆ Upload & Ingest File", type="primary", key="btn_file_ingest"):
        if not fu_title.strip():
            st.warning("Please enter a document title.")
        elif uploaded is None:
            st.warning("Please select a .txt file to upload.")
        else:
            with st.spinner(f"Uploading and indexing '{uploaded.name}'…"):
                try:
                    uploaded.seek(0)
                    resp = httpx.post(
                        f"{api_url}/documents/ingest-file",
                        files={"file": (uploaded.name, uploaded.read(), "text/plain")},
                        data={
                            "title": fu_title.strip(),
                            "ticker": fu_ticker.strip(),
                            "doc_type": fu_doc_type,
                            "source": fu_source.strip(),
                            "document_date": fu_date.strip(),
                        },
                        timeout=90,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ **'{data['title']}'** uploaded successfully! "
                            f"{data['chunks_added']} chunks added "
                            f"({data['content_length']:,} chars · index total: {data['total_index_size']})."
                        )
                    else:
                        st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Connection error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Search Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("🔍 Search the Knowledge Base")
    st.markdown(
        "Ask a natural-language question. The system will retrieve the most relevant "
        "document chunks from the FAISS index using semantic similarity."
    )

    col_q, col_k = st.columns([3, 1])
    with col_q:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g. What was Apple's revenue growth in Q4 2024?",
            key="search_query",
        )
    with col_k:
        top_k = st.number_input("Top K Results", min_value=1, max_value=20, value=5, key="search_k")

    if st.button("🔍 Search", type="primary", key="btn_search"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching knowledge base…"):
                try:
                    resp = httpx.get(
                        f"{api_url}/documents/search",
                        params={"query": search_query, "top_k": top_k},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        if not results:
                            st.info(data.get("message", "No results found. Try ingesting some documents first."))
                        else:
                            st.success(
                                f"Found **{data['chunks_returned']}** relevant chunks "
                                f"(index size: {data['total_index_size']} vectors)"
                            )
                            for i, chunk in enumerate(results, 1):
                                with st.expander(f"Result {i}", expanded=(i == 1)):
                                    st.markdown(chunk)
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Document Library
# ═══════════════════════════════════════════════════════════════════════════════
with tab_library:
    st.subheader("📖 Indexed Documents")

    col_filter, col_refresh = st.columns([2, 1])
    with col_filter:
        filter_ticker = st.text_input(
            "Filter by Ticker (leave blank for all)",
            placeholder="e.g. AAPL",
            key="lib_ticker_filter",
        ).upper()
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_lib = st.button("🔄 Refresh List", key="btn_refresh_lib")

    if "lib_docs" not in st.session_state or refresh_lib:
        with st.spinner("Loading document library…"):
            try:
                params: dict = {"limit": 200}
                if filter_ticker.strip():
                    params["ticker"] = filter_ticker.strip()
                resp = httpx.get(f"{api_url}/documents", params=params, timeout=15)
                if resp.status_code == 200:
                    st.session_state["lib_docs"] = resp.json()
                else:
                    st.session_state["lib_docs"] = []
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.session_state["lib_docs"] = []
                st.error(f"Connection error: {e}")

    docs = st.session_state.get("lib_docs", [])

    if not docs:
        st.info("No documents found. Upload documents in the **Paste Text** or **Upload File** tab.")
    else:
        st.caption(f"Showing {len(docs)} document(s)")
        for doc in docs:
            ticker_tag = f" `{doc['ticker']}`" if doc.get("ticker") else ""
            label = f"**{doc['title']}**{ticker_tag} — *{doc['doc_type']}*"
            with st.expander(label):
                col_meta2, col_action = st.columns([3, 1])
                with col_meta2:
                    st.markdown(f"**ID:** {doc['id']} | **Date:** {doc.get('document_date') or 'N/A'} | **Source:** {doc.get('source') or 'N/A'}")
                    st.markdown(f"**Indexed:** {'✅ Yes' if doc['faiss_indexed'] else '⚠️ No'} | **Content:** {doc['content_length']:,} chars | **Added:** {doc['created_at'][:10]}")
                    st.markdown("**Preview:**")
                    st.text(doc.get("content_preview", ""))
                with col_action:
                    if st.button(
                        "🗑 Delete",
                        key=f"del_{doc['id']}",
                        help="Removes from DB. FAISS embeddings persist until index rebuild.",
                    ):
                        try:
                            del_resp = httpx.delete(f"{api_url}/documents/{doc['id']}", timeout=10)
                            if del_resp.status_code == 200:
                                st.success(f"Deleted '{doc['title']}'")
                                # Remove from session state and rerun
                                st.session_state["lib_docs"] = [
                                    d for d in st.session_state["lib_docs"] if d["id"] != doc["id"]
                                ]
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {del_resp.text}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
