"""
chat_routes.py
--------------
FastAPI route for the AI chatbot interface.

Endpoint:
    POST /chat  — Send a message; the AI responds using RAG context + live market data.
    GET  /chat/history  — Return recent conversation history (stored in memory for session).
"""
from __future__ import annotations

import asyncio

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

limiter = Limiter(key_func=get_remote_address)

from backend.database.connection import get_db
from backend.models.user_query import UserQuery
from backend.services.market_data_service import MarketDataService
from backend.services.news_service import NewsService
from backend.services.rag_service import RAGService
from backend.utils.config import settings
from backend.utils.logger import get_logger

router = APIRouter(prefix="/chat", tags=["AI Chatbot"])
logger = get_logger(__name__)

_rag = RAGService()
_mds = MarketDataService()
_ns = NewsService()

_SYSTEM_PROMPT = """You are an expert AI Financial Analyst assistant embedded in the Autonomous Financial Analyst platform.

You have access to:
- Real-time stock quotes and technical indicators
- Latest financial news headlines
- A knowledge base of financial documents (earnings transcripts, analyst notes, reports)

Guidelines:
- Answer clearly and concisely with financial precision
- When discussing stocks, always mention relevant risks
- Cite the context you used (news headlines, document excerpts) when applicable
- If you don't have data on something, say so clearly rather than hallucinating
- Format responses with markdown (bold key metrics, use bullet points for lists)"""


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=5000)


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000)
    ticker: Optional[str] = Field(default=None, max_length=10, description="Optional ticker to focus analysis on")
    history: List[ChatMessage] = Field(default_factory=list, max_length=20, description="Prior conversation turns (max 20 turns)")


@router.post("", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Send a message to the AI financial analyst chatbot.

    Automatically retrieves:
    - RAG document context relevant to the query
    - Live stock quote if a ticker is provided
    - Recent news headlines for context

    Returns the AI response plus the context sources used.
    """
    ticker = body.ticker.upper().strip() if body.ticker else None
    context_parts: List[str] = []
    sources_used: List[str] = []

    # ── 1. RAG document context ───────────────────────────────────────────────
    try:
        if _rag._index and _rag._index.ntotal > 0:
            rag_ctx = await asyncio.to_thread(_rag.retrieve, body.message, top_k=settings.rag_top_k)
            if rag_ctx and len(rag_ctx) > 50:
                context_parts.append(f"## Relevant Document Excerpts\n{rag_ctx}")
                sources_used.append("knowledge_base")
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)

    # ── 2. Live stock quote (if ticker given) ─────────────────────────────────
    if ticker:
        try:
            quote = await asyncio.to_thread(_mds.get_quote, ticker)
            if quote:
                q_lines = [
                    f"**{quote.get('company_name', ticker)} ({ticker})** — Live Quote",
                    f"- Price: ${quote.get('price', 'N/A')}",
                    f"- Change: {quote.get('change_pct', 'N/A')}%",
                    f"- Market Cap: ${quote.get('market_cap'):,}" if isinstance(quote.get("market_cap"), (int, float)) else "- Market Cap: N/A",
                    f"- 52-week High: ${quote.get('52w_high', 'N/A')}",
                    f"- 52-week Low: ${quote.get('52w_low', 'N/A')}",
                    f"- Sector: {quote.get('sector', 'N/A')}",
                ]
                context_parts.append("## Live Market Data\n" + "\n".join(q_lines))
                sources_used.append("live_market_data")
        except Exception as exc:
            logger.warning("Market data fetch failed for %s: %s", ticker, exc)

    # ── 3. Recent news ────────────────────────────────────────────────────────
    try:
        if ticker:
            articles = await asyncio.to_thread(_ns.get_stock_news, ticker, max_articles=5)
        else:
            articles = await asyncio.to_thread(_ns.get_top_financial_news, max_articles=5)
        if articles:
            headlines = "\n".join(
                f"- {a.get('title', '')} ({a.get('source', {}).get('name', '') if isinstance(a.get('source'), dict) else a.get('source', '')})"
                for a in articles[:5]
            )
            context_parts.append(f"## Recent News Headlines\n{headlines}")
            sources_used.append("news")
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)

    # ── 4. Build messages list for OpenAI ─────────────────────────────────────
    context_block = "\n\n".join(context_parts)
    system_content = _SYSTEM_PROMPT
    if context_block:
        system_content += f"\n\n---\n## Current Context\n{context_block}"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]

    # Add prior conversation history (last 10 turns max to stay within context)
    for turn in body.history[-10:]:
        messages.append({"role": turn.role, "content": turn.content})

    messages.append({"role": "user", "content": body.message})

    # ── 5. Call OpenAI (with retry on transient errors) ───────────────────────
    try:
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        @retry(
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        async def _call_llm():
            return await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.4,
                max_tokens=1024,
            )

        resp = await _call_llm()
        reply = resp.choices[0].message.content
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Please wait a moment.")
    except Exception as exc:
        logger.error("OpenAI chat completion failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable.")

    # ── 6. Persist query ──────────────────────────────────────────────────────
    try:
        db.add(UserQuery(
            endpoint="/chat",
            ticker=ticker,
            query_payload=body.message[:500],
            response_summary=(reply or "")[:500],
        ))
        db.commit()
    except Exception as exc:
        logger.warning("Failed to persist query log: %s", exc)

    return {
        "response": reply,
        "ticker": ticker,
        "sources_used": sources_used,
        "context_available": bool(context_parts),
        "rag_vectors": _rag._index.ntotal if _rag._index else 0,
    }
