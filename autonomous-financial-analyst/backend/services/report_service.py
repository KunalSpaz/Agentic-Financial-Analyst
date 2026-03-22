"""
report_service.py
-----------------
Generates daily AI market briefings using GPT-4o.
The briefing includes top opportunities, sentiment summary, key news,
and recommended trades.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.utils.config import settings
from backend.utils.logger import get_logger

_OPENAI_RETRY = dict(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)

logger = get_logger(__name__)


class ReportService:
    """
    Creates narrative AI market reports using structured data + GPT-4o.
    """

    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)

    def generate_daily_briefing(
        self,
        opportunities: List[Dict[str, Any]],
        market_indices: Dict[str, Any],
        top_news: List[Dict[str, Any]],
        overall_sentiment: str = "NEUTRAL",
    ) -> Dict[str, Any]:
        """
        Generate the daily AI market briefing.

        Args:
            opportunities:     Ranked list of market opportunities.
            market_indices:    Current state of SPY/QQQ/DIA/VIX.
            top_news:          Top financial headlines.
            overall_sentiment: Aggregate market sentiment label.

        Returns:
            Dict with ``date``, ``narrative``, ``top_picks``, and ``sentiment``.
        """
        today = datetime.date.today().isoformat()

        # Summarise inputs for the prompt
        opp_lines = "\n".join(
            f"  {i+1}. {o.get('ticker')} — {o.get('recommendation')} "
            f"(confidence={o.get('confidence_score', 0):.0f})"
            for i, o in enumerate(opportunities[:5])
        )
        news_lines = "\n".join(
            f"  • {n.get('title', '')}" for n in top_news[:5]
        )
        indices_lines = "\n".join(
            f"  {sym}: ${data.get('price', 'N/A')} ({data.get('change_pct', 0):.2f}%)"
            for sym, data in market_indices.items()
        )

        system_prompt = (
            "You are a senior financial analyst AI. Write a concise, professional "
            "daily market briefing in 4 sections: Market Overview, Top Investment "
            "Opportunities, Key Risks, and Recommended Actions. Be specific, "
            "data-driven, and professional. 300-400 words total."
        )

        user_prompt = f"""
Date: {today}
Overall Market Sentiment: {overall_sentiment}

Market Indices:
{indices_lines}

Top Investment Opportunities:
{opp_lines}

Key Headlines:
{news_lines}

Generate the daily AI market briefing.
""".strip()

        try:
            @retry(**_OPENAI_RETRY)
            def _call() -> str:
                resp = self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                )
                return resp.choices[0].message.content.strip()

            narrative = _call()
        except Exception as exc:
            logger.error("GPT-4o report generation failed: %s", exc)
            narrative = "Daily market briefing temporarily unavailable."

        return {
            "date": today,
            "narrative": narrative,
            "overall_sentiment": overall_sentiment,
            "top_picks": [o.get("ticker") for o in opportunities[:5]],
            "market_indices": market_indices,
        }
