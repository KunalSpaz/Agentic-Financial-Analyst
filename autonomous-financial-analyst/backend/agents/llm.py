"""
llm.py
------
Shared LangChain chat-model factory. Replaces the raw ``llm=settings.openai_model``
string CrewAI accepted directly — LangGraph nodes call a real ``ChatOpenAI``
instance's ``.invoke()``.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from backend.utils.config import settings


@lru_cache(maxsize=1)
def get_chat_model(temperature: float = 0.3) -> ChatOpenAI:
    """Return a cached :class:`ChatOpenAI` instance configured from settings."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
