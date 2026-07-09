"""
opportunity_scanner_agent.py
-----------------------------
LangGraph Opportunity Scanner node — the final step of MarketScanGraph.
Runs once over the full ranked opportunity list (not per-ticker) to produce
a thematic/sector-clustering market narrative.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import OpportunityScanState

ROLE = "Market Opportunity Scanner"
GOAL = (
    "Systematically scan the predefined stock universe, compute confidence scores "
    "for each, and produce a ranked list of the best current investment "
    "opportunities with clear rationale."
)
BACKSTORY = (
    "You are a quantitative research analyst who built systematic screening "
    "tools at a hedge fund. You excel at filtering the universe of stocks "
    "to surface high-conviction opportunities efficiently."
)


def create_opportunity_scanner_node() -> Callable[[OpportunityScanState], Dict[str, str]]:
    """Build the LangGraph node function for the Opportunity Scanner agent."""
    llm = get_chat_model()

    def node(state: OpportunityScanState) -> Dict[str, str]:
        opportunities = state.get("opportunities") or []
        ranked_lines = "\n".join(
            f"{o['rank']}. {o['ticker']} — {o['recommendation']} "
            f"(confidence={o['confidence_score']:.0f}, sector={o.get('sector', 'Unknown')})"
            for o in opportunities
        ) or "No opportunities found."
        task = (
            "The following is the full ranked list of scanned investment opportunities. "
            "Identify sector clustering or thematic patterns among the top picks, note "
            "any diversification concerns if several top picks share a sector, and summarise "
            f"the overall market read in 3-4 sentences.\n\n{ranked_lines}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"market_narrative": response.content}

    return node
