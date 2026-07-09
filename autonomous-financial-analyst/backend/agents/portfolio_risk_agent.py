"""
portfolio_risk_agent.py
-----------------------
LangGraph Portfolio Risk node — interprets a completed PortfolioRiskService
analysis (volatility, beta, Sharpe, VaR, correlation, sector exposure) into
a plain-language risk narrative. Wired into POST /portfolio-analysis via
backend/agents/portfolio_graph.py.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import PortfolioRiskNarrativeState

ROLE = "Portfolio Risk Manager"
GOAL = (
    "Analyse the risk profile of any submitted portfolio. Compute "
    "volatility, beta, Sharpe ratio, max drawdown, VaR(95%), correlation "
    "matrix, and sector exposure. Provide plain-language risk interpretation "
    "and actionable diversification recommendations."
)
BACKSTORY = (
    "You are a chief risk officer with 20 years of experience managing "
    "multi-billion dollar portfolio risk. You specialise in identifying "
    "hidden concentrations, tail risks, and correlation breakdown scenarios."
)


def create_portfolio_risk_node() -> Callable[[PortfolioRiskNarrativeState], Dict[str, str]]:
    """Build the LangGraph node function for the Portfolio Risk agent."""
    llm = get_chat_model()

    def node(state: PortfolioRiskNarrativeState) -> Dict[str, str]:
        result = state.get("risk_result") or {}
        metrics_lines = "\n".join(
            f"- {k}: {v}" for k, v in result.items()
            if k not in ("correlation_matrix", "holdings", "sector_exposure")
        )
        sector_lines = "\n".join(f"  - {k}: {v:.1%}" for k, v in (result.get("sector_exposure") or {}).items())
        task = (
            "Interpret the following portfolio risk metrics. Explain what the "
            "volatility/beta/Sharpe/VaR combination means in plain language, flag any "
            "concentration or correlation risk from the sector exposure below, and give "
            f"actionable diversification recommendations.\n\n{metrics_lines}\n\nSector exposure:\n{sector_lines}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"narrative": response.content}

    return node
