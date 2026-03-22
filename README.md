# Autonomous Financial Analyst

An AI-powered financial intelligence platform combining a **12-agent CrewAI pipeline** with deterministic Python services for technical analysis, NLP sentiment analysis (FinBERT), RAG document intelligence, backtesting, portfolio risk, and strategy optimisation — all served through a FastAPI backend and an interactive Streamlit dashboard.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Backend](#backend)
  - [Services](#services)
  - [CrewAI Agents](#crewai-agents)
  - [API Endpoints](#api-endpoints)
  - [Database](#database)
  - [Utilities](#utilities)
- [Frontend](#frontend)
  - [Dashboard Pages](#dashboard-pages)
  - [Components](#components)
- [Tests](#tests)
- [CI/CD](#cicd)
- [Environment Variables](#environment-variables)
- [Docker](#docker)

---

## Architecture

```
Data Sources
  yfinance (OHLCV + quotes)  ·  NewsAPI (headlines)  ·  Uploaded Documents
          │
          ▼
Python Service Layer  ─── deterministic computation, no LLM
  MarketDataService      → live quotes, OHLCV history via yfinance
  TechnicalAnalysisService → RSI, MACD, SMA 50/200, Bollinger Bands, volume ratio
  NewsService            → top headlines fetched from NewsAPI
  SentimentService       → FinBERT (ProsusAI/finbert) — POSITIVE/NEUTRAL/NEGATIVE
  RAGService             → FAISS + OpenAI embeddings — document retrieval
  ConfidenceScoreService → weighted score: technical 50% + sentiment 30% + momentum 20%
  RecommendationService  → maps score → STRONG BUY / BUY / HOLD / SELL / STRONG SELL
  BacktestingService     → RSI+MACD+Sentiment strategy, equity curve, trade log
  PortfolioRiskService   → VaR, beta, Sharpe, max drawdown, correlation, sector exposure
  StrategyOptimisationService → grid-search over RSI thresholds + MACD + MA filter
  ReportService          → GPT-4o daily market briefings
          │
          ▼
CrewAI Intelligence Layer  ─── LLM reasoning & synthesis (12 agents)
  6 active agents in the stock analysis crew (sequential)
  6 specialist agents (market data, news, technical, sentiment, backtesting,
    opportunity scanner, portfolio risk, strategy optimisation, report writer)
          │
          ▼
FastAPI Backend  ─── 19 REST endpoints + APScheduler (daily jobs at 08:00 ET)
          │
          ▼
Streamlit Dashboard  ─── 9 interactive pages
```

### Design Principle

The system separates **computation** from **reasoning**:

- **Python services** handle everything deterministic — fetching OHLCV data, computing RSI/MACD/Bollinger Bands via pandas-ta, running FinBERT inference, executing backtest loops, and calculating VaR/Sharpe/beta. These produce reliable, reproducible numbers.
- **CrewAI agents** handle everything interpretive — synthesising pre-fetched data into structured intelligence, explaining what a combination of signals means, weighing risks, and writing the final investment narrative. These are pure reasoning tasks where LLMs add genuine value.

Asking an LLM to "fetch a stock quote" or "compute RSI" produces hallucinated numbers. Asking it to "interpret what RSI=28 combined with a MACD bullish divergence means for an entry decision" produces useful insight.

---

## Features

| Feature | Details |
|---|---|
| **12-Agent CrewAI Pipeline** | Sequential 6-agent stock analysis crew + 6 specialist agents for market data, news, sentiment, backtesting, portfolio risk, and optimisation |
| **AI Chatbot** | GPT-4o conversational analyst grounded in live market data + RAG document context |
| **Technical Indicators** | RSI(14), MACD(12/26/9), SMA(50/200), Bollinger Bands(20,2σ), Volume ratio |
| **FinBERT Sentiment** | ProsusAI/finbert — POSITIVE / NEUTRAL / NEGATIVE classification per article, aggregated compound score |
| **Confidence Scoring** | Weighted signal engine: technical 50% + sentiment 30% + momentum 20%, clamped [0, 100] |
| **Recommendations** | STRONG BUY / BUY / HOLD / SELL / STRONG SELL with color-coded badges |
| **Opportunity Scanner** | Ranks entire stock universe by confidence score with refresh-on-demand |
| **Strategy Backtesting** | RSI+MACD+Sentiment strategy — equity curve, Sharpe ratio, max drawdown, win rate, trade log |
| **Strategy Optimisation** | Grid-search over RSI thresholds, MACD confirmation, MA filter — 3 objectives (return, Sharpe, drawdown) |
| **Portfolio Risk** | Volatility, beta vs SPY, Sharpe, max drawdown, VaR(95%), correlation heatmap, sector exposure |
| **RAG Document Library** | FAISS + OpenAI embeddings — ingest earnings transcripts, analyst reports, SEC filings |
| **Daily Briefings** | GPT-4o market briefings auto-generated at 08:00 ET via APScheduler |
| **Tenacity Retry** | All OpenAI calls (chat, embeddings, report) automatically retry on RateLimitError / APIConnectionError with exponential backoff (1→10s, 3 attempts) |
| **Request ID Tracing** | Every HTTP request gets an `X-Request-ID` header propagated through all log lines via `contextvars` |
| **Security** | Optional `X-API-Key` header auth, configurable CORS, per-endpoint rate limiting via slowapi, security headers middleware |
| **Structured Logging** | JSON log output with timestamp, level, logger, message, and `request_id` correlation field |
| **Docker** | `Dockerfile` + `docker-compose.yml` for one-command deployment |

---

## Quick Start

### Requirements

- Python **3.10–3.13** (CrewAI does not support 3.14+)
- OpenAI API key
- NewsAPI key (optional — system degrades gracefully without it)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/your-org/autonomous-financial-analyst.git
cd autonomous-financial-analyst

# Use Python 3.12 explicitly if your system default is newer
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Development (latest compatible versions):
pip install -r requirements.txt

# Production / CI (exact pinned versions — guaranteed reproducible):
pip install -r requirements.lock
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set at minimum:
# OPENAI_API_KEY=sk-proj-...
# NEWS_API_KEY=your_newsapi_key
```

### 4. Start backend

```bash
# Always invoke through the venv python to avoid subprocess Python version mismatch
.venv/bin/python -m uvicorn backend.api.main:app --reload --reload-dir backend --port 8000
```

> `--reload-dir backend` restricts hot-reload watching to source files only, preventing the SQLite DB and FAISS index writes from triggering unnecessary restarts.

### 5. Start frontend

```bash
# In a second terminal (with venv active)
.venv/bin/python -m streamlit run frontend/app.py
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |
| ReDoc API Docs | http://localhost:8000/redoc |

> **First run:** FinBERT (~500 MB) downloads from HuggingFace automatically on startup. The SQLite database and `data/` directory are also created automatically.

---

## Project Structure

```
autonomous-financial-analyst/
│
├── backend/
│   ├── agents/
│   │   ├── crew_orchestrator.py          # FinancialAnalystCrew — wires agents + tasks
│   │   ├── market_data_agent.py          # Fetches and structures OHLCV data
│   │   ├── news_intelligence_agent.py    # Synthesises news into structured intelligence
│   │   ├── technical_analysis_agent.py   # Interprets indicator combinations
│   │   ├── sentiment_analysis_agent.py   # Aggregates and contextualises sentiment
│   │   ├── document_intelligence_agent.py # Extracts insights from RAG chunks
│   │   ├── financial_analysis_agent.py   # Unified assessment from all inputs
│   │   ├── investment_decision_agent.py  # Final recommendation + rationale
│   │   ├── report_writer_agent.py        # Formats final investment report
│   │   ├── opportunity_scanner_agent.py  # Ranks tickers by opportunity
│   │   ├── portfolio_risk_agent.py       # Interprets portfolio risk metrics
│   │   ├── backtesting_agent.py          # Interprets backtest results
│   │   └── strategy_optimization_agent.py # Recommends optimal parameters
│   │
│   ├── services/
│   │   ├── market_data_service.py        # yfinance — quotes, OHLCV history
│   │   ├── technical_analysis_service.py # pandas-ta — RSI, MACD, SMA, BB
│   │   ├── news_service.py               # NewsAPI — article fetching
│   │   ├── sentiment_service.py          # FinBERT inference + aggregation
│   │   ├── rag_service.py                # FAISS index — ingest, chunk, embed, retrieve
│   │   ├── confidence_score_service.py   # Weighted signal scoring engine
│   │   ├── recommendation_service.py     # Score → recommendation mapping
│   │   ├── backtesting_service.py        # Strategy backtest engine
│   │   ├── portfolio_risk_service.py     # Risk metrics calculation
│   │   ├── strategy_optimization_service.py # Grid-search optimiser
│   │   └── report_service.py             # GPT-4o daily briefing generation
│   │
│   ├── api/
│   │   ├── main.py                       # FastAPI app factory, middleware, exception handlers
│   │   └── routes/
│   │       ├── stock_routes.py           # /analyze-stock, /stock/{ticker}
│   │       ├── market_routes.py          # /market-report, /top-news, /market-opportunities
│   │       ├── portfolio_routes.py       # /portfolio-analysis
│   │       ├── backtest_routes.py        # /backtest
│   │       ├── optimization_routes.py    # /optimize-strategy
│   │       ├── rag_routes.py             # /documents/* (ingest, search, list, delete)
│   │       └── chat_routes.py            # /chat
│   │
│   ├── database/
│   │   ├── connection.py                 # SQLAlchemy engine, session factory, Base
│   │   └── migrations.py                 # create_all on startup
│   │
│   ├── models/                           # SQLAlchemy ORM models (one file per table)
│   │   ├── analysis_report.py
│   │   ├── backtest_result.py
│   │   ├── financial_document.py
│   │   ├── market_opportunity.py
│   │   ├── news_article.py
│   │   ├── portfolio_risk_report.py
│   │   ├── sentiment_score.py
│   │   ├── stock.py
│   │   ├── strategy_optimization.py
│   │   └── user_query.py
│   │
│   └── utils/
│       ├── config.py                     # Pydantic Settings — all env vars with defaults
│       ├── logger.py                     # JSON structured logger + request_id ContextVar
│       └── scheduler.py                  # APScheduler — daily market scan + briefing
│
├── frontend/
│   ├── app.py                            # Streamlit entrypoint + landing page
│   ├── pages/
│   │   ├── 1_market_overview.py          # Live indices, news, sentiment
│   │   ├── 2_stock_analysis.py           # Full AI analysis pipeline with charts
│   │   ├── 3_opportunity_scanner.py      # Ranked investment opportunities
│   │   ├── 4_daily_market_report.py      # AI-generated daily briefing
│   │   ├── 5_portfolio_advisor.py        # Risk metrics, correlation, sector exposure
│   │   ├── 6_strategy_backtesting.py     # Backtest with equity curve + trade log
│   │   ├── 7_strategy_optimization.py    # Grid-search parameter tuning
│   │   ├── 8_document_library.py         # Upload, browse, search RAG documents
│   │   └── 9_ai_chatbot.py               # GPT-4o chatbot with RAG + live data
│   └── components/
│       ├── sidebar.py                    # Global sidebar: branding, nav, backend status
│       ├── charts.py                     # Plotly chart factories (candlestick, RSI, MACD…)
│       └── metrics.py                    # Confidence gauge, metric cards, recommendation badge
│
├── tests/                                # 516 pytest tests across 20 files — all passing
├── data/                                 # Auto-created: SQLite DB + FAISS index
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                      # Unpinned — latest compatible versions
├── requirements.lock                     # Pinned — exact reproducible versions
├── .env.example
└── CLAUDE.md                             # AI assistant conventions and guidelines
```

---

## Backend

### Services

Each service is a focused, stateless class with no HTTP or database concerns. Route handlers call services and handle persistence.

#### `MarketDataService`
Wraps `yfinance` for all market data needs.
- `get_quote(ticker)` — live price, change %, market cap, P/E, volume, 52-week high/low
- `get_historical_data(ticker, period)` — OHLCV DataFrame for a given period (1mo, 3mo, 6mo, 1y, 2y)
- `get_market_indices()` — live quotes for SPY, QQQ, DIA, VIX

#### `TechnicalAnalysisService`
Computes technical indicators via `pandas-ta`.
- RSI(14) — momentum oscillator
- MACD(12/26/9) — trend + momentum
- SMA(50) and SMA(200) — trend direction
- Bollinger Bands(20, 2σ) — volatility envelope
- Volume ratio — current vs 20-day average

#### `NewsService`
Fetches articles from `NewsAPI` with configurable `max_articles` limit. Gracefully returns empty list when `NEWS_API_KEY` is absent.

#### `SentimentService`
Runs `ProsusAI/finbert` (loaded once at startup, pre-warmed in a background thread) on article titles and descriptions. Truncates input to 512 tokens. Aggregates per-article POSITIVE/NEUTRAL/NEGATIVE labels into a compound float score. Loads on CPU by default; moves to GPU automatically if available.

#### `RAGService`
FAISS-backed vector store for financial documents.
- `add_documents(docs)` — chunks text (512-char chunks, 64-char overlap), generates OpenAI embeddings, adds to FAISS index
- `retrieve(query, top_k)` — embeds query, returns top-k most relevant chunks with metadata header `[TICKER | TITLE]`
- Thread-safe: mutations guarded by `threading.Lock`; reads also hold the lock to prevent race conditions
- Persists index to disk at `FAISS_INDEX_PATH` on every write

#### `ConfidenceScoreService`
Produces a [0, 100] confidence score from three weighted sub-scorers:
- **Technical (50%)**: RSI oversold/overbought, golden/death cross (SMA50/200), Bollinger Band position, MACD crossover and histogram direction
- **Sentiment (30%)**: FinBERT compound score mapped to +15/0/−15 adjustment
- **Momentum (20%)**: Price vs SMA50 and SMA200 position, volume confirmation

#### `RecommendationService`
Maps confidence score to a five-tier recommendation:
| Score | Recommendation | Color |
|---|---|---|
| ≥ 70 | STRONG BUY | `#00C853` |
| 55–69 | BUY | `#69F0AE` |
| 40–54 | HOLD | `#FFD740` |
| 25–39 | SELL | `#FF6D00` |
| < 25 | STRONG SELL | `#D50000` |

#### `BacktestingService`
Simulates a configurable RSI+MACD+Sentiment strategy on historical data.
- Entry: RSI < `rsi_buy_threshold` + optional MACD confirmation + optional MA filter
- Exit: RSI > `rsi_sell_threshold`
- Computes: total return, annualised return, Sharpe ratio, max drawdown, win rate, trade count
- Returns equity curve (list of portfolio values) and full trade log

#### `PortfolioRiskService`
Calculates risk metrics for a multi-ticker portfolio.
- Annualised volatility and return
- Beta vs SPY benchmark
- Sharpe ratio (risk-free rate 4.5%)
- Max drawdown
- VaR(95%) — parametric
- Correlation matrix between all holdings
- Sector exposure via yfinance metadata
- Guards against zero-weight portfolios and zero-peak drawdown producing NaN

#### `StrategyOptimisationService`
Grid-searches over: RSI buy thresholds [25, 30, 35], RSI sell thresholds [65, 70, 75], MACD confirmation [true/false], MA filter [true/false] = 36 combinations. Supports three optimisation objectives: `maximize_return`, `maximize_sharpe`, `minimize_drawdown`.

#### `ReportService`
Generates GPT-4o daily market briefings. Wraps OpenAI call with tenacity retry (exponential backoff, 3 attempts) on rate limit or connection errors.

---

### CrewAI Agents

All 12 agents are defined as factory functions in `backend/agents/`. Each is instantiated fresh per request inside the crew orchestrator. All use `settings.openai_model` for the LLM.

#### Stock Analysis Crew (6 agents, sequential)

When `/analyze-stock` is called, these agents run in series, each receiving the output of the previous:

| # | Agent | Role | What it produces |
|---|---|---|---|
| 1 | `DocumentIntelligenceAgent` | Financial document analyst | Key themes and signals extracted from RAG-retrieved document chunks |
| 2 | `NewsIntelligenceAgent` | News intelligence specialist | Structured intelligence: material events, sentiment direction, risk flags, contradictions |
| 3 | `TechnicalAnalysisAgent` | Technical analysis expert | Narrative interpretation of RSI, MACD, SMA, Bollinger Band combinations and divergences |
| 4 | `FinancialAnalysisAgent` | Senior financial analyst | Unified assessment synthesising document intelligence, news, and technical signals |
| 5 | `InvestmentDecisionAgent` | Portfolio manager | Confirmed recommendation with investment rationale, entry conditions, and risk factors |
| 6 | `ReportWriterAgent` | Financial report writer | Final investment report in structured markdown ready for display |

#### Specialist Agents (6 agents)

Used in other crew contexts (market scan, portfolio analysis, etc.):

| Agent | Purpose |
|---|---|
| `MarketDataAgent` | Fetches and validates OHLCV data for downstream agents |
| `SentimentAnalysisAgent` | Aggregates and contextualises sentiment scores |
| `OpportunityScannerAgent` | Ranks tickers by opportunity signal strength |
| `PortfolioRiskAgent` | Interprets risk metric outputs in portfolio context |
| `BacktestingAgent` | Interprets backtest results and suggests parameter refinements |
| `StrategyOptimisationAgent` | Recommends optimal strategy parameters from grid-search results |

---

### API Endpoints

All endpoints require `X-API-Key` header if `API_SECRET_KEY` is set in the environment. Rate limits apply per IP.

#### Stock & Market

| Method | Endpoint | Rate Limit | Description |
|---|---|---|---|
| `POST` | `/analyze-stock` | 5/min | Full pipeline: live data → indicators → sentiment → 6-agent CrewAI narrative |
| `GET` | `/stock/{ticker}` | 30/min | Live quote + latest cached analysis from DB |
| `GET` | `/market-report` | 10/min | AI-generated daily market briefing (cached in DB) |
| `GET` | `/top-news` | 20/min | Top financial headlines with optional `limit` param |
| `GET` | `/market-opportunities` | 10/min | Ranked opportunity scanner results; `?refresh=true` re-runs crew |

**`POST /analyze-stock` request body:**
```json
{
  "ticker": "AAPL",
  "period": "1y"
}
```

#### Portfolio, Backtest, Optimisation

| Method | Endpoint | Rate Limit | Description |
|---|---|---|---|
| `POST` | `/portfolio-analysis` | 5/min | Portfolio risk metrics: VaR, beta, Sharpe, correlation, sector exposure |
| `POST` | `/backtest` | 10/min | Run strategy backtest with equity curve + trade log |
| `POST` | `/optimize-strategy` | 5/min | Grid-search parameter optimisation |

**`POST /backtest` request body:**
```json
{
  "ticker": "NVDA",
  "period": "1y",
  "rsi_buy_threshold": 30,
  "rsi_sell_threshold": 70,
  "macd_confirmation": true,
  "ma_filter": false,
  "initial_capital": 10000,
  "sentiment_label": "POSITIVE"
}
```

#### Document Library (RAG)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/ingest` | Ingest document from text body; returns chunks added + total index size |
| `POST` | `/documents/ingest-file` | Ingest `.txt` file upload (multipart form) |
| `GET` | `/documents/search?query=...&top_k=5` | Semantic search over FAISS index |
| `GET` | `/documents?limit=50&ticker=AAPL` | List indexed documents, optional ticker filter |
| `DELETE` | `/documents/{id}` | Delete a document record |

#### AI Chatbot

| Method | Endpoint | Rate Limit | Description |
|---|---|---|---|
| `POST` | `/chat` | 20/min | GPT-4o chat grounded in RAG + live market data + conversation history |

**`POST /chat` request body:**
```json
{
  "message": "What is the outlook for NVDA?",
  "ticker": "NVDA",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

#### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns `{"status": "ok", "version": "1.0.0", "env": "..."}` |

---

### Database

10 SQLite tables, auto-created on first startup via `run_migrations()`. All use SQLAlchemy 2.x `mapped_column` style.

| Table | Purpose | Populated by |
|---|---|---|
| `analysis_reports` | Full stock analysis results with CrewAI narrative | `POST /analyze-stock` |
| `market_opportunities` | Ranked opportunity scanner results | `GET /market-opportunities` + daily scheduler |
| `financial_documents` | Document metadata for RAG library | `POST /documents/ingest*` |
| `backtest_results` | Backtest metrics, equity curve, trade log | `POST /backtest` |
| `strategy_optimizations` | Grid-search results and best parameters | `POST /optimize-strategy` |
| `portfolio_risk_reports` | Portfolio risk metric snapshots | `POST /portfolio-analysis` |
| `news_articles` | Cached news article data | Daily scheduler + news fetch |
| `sentiment_scores` | Per-article FinBERT sentiment scores | Daily scheduler |
| `user_queries` | Audit log of all API calls | Every request |
| `stocks` | Stock metadata cache | Market data fetches |

---

### Utilities

#### `config.py`
Pydantic `Settings` class. All environment variables are declared here with types and defaults. Import `settings` from `backend.utils.config` everywhere — never read `os.environ` directly.

#### `logger.py`
JSON structured logger using Python's standard `logging`. Every log record includes `timestamp`, `level`, `logger`, and `message`. When inside an HTTP request context, also includes `request_id` for distributed tracing. Uses `contextvars.ContextVar` so the request ID is isolated per async task — no cross-request bleed.

```python
from backend.utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Processing ticker %s", ticker)
# Output: {"timestamp": "...", "level": "INFO", "logger": "...", "message": "Processing ticker AAPL", "request_id": "abc-123"}
```

#### `scheduler.py`
APScheduler `BackgroundScheduler` with two daily jobs fired at `DAILY_REPORT_HOUR:DAILY_REPORT_MINUTE` ET:
1. **Daily Market Scan** — runs the opportunity scanner over the full stock universe and persists to DB
2. **Daily AI Market Briefing** — calls `ReportService` to generate and cache a GPT-4o market briefing

---

## Frontend

### Dashboard Pages

#### 1 — Market Overview (`1_market_overview.py`)
Live market snapshot. Displays SPY, QQQ, DIA, and VIX price metrics, the AI-generated daily briefing narrative, an overall sentiment badge, and top financial news headlines in expandable cards.

#### 2 — Stock Analysis (`2_stock_analysis.py`)
Full AI analysis pipeline for any ticker. Displays:
- Live quote metrics (price, change, market cap, P/E, volume)
- Confidence gauge (Plotly, 0–100) with recommendation badge
- Candlestick chart with SMA/Bollinger overlays
- RSI chart with overbought/oversold zones
- MACD chart with histogram
- Full 6-agent CrewAI narrative report in markdown
- Results cached in `session_state["analysis_result"]` — no re-run on widget change

#### 3 — Opportunity Scanner (`3_opportunity_scanner.py`)
Ranks all tickers in the stock universe by confidence score. Shows a horizontal bar chart coloured by recommendation, plus a sortable data table. Optional `Refresh` button re-runs the full crew scan.

#### 4 — Daily Market Report (`4_daily_market_report.py`)
Displays the AI-generated daily briefing with market indices, narrative text, top picks, and sentiment. One-click "Regenerate" forces a fresh briefing.

#### 5 — Portfolio Advisor (`5_portfolio_advisor.py`)
Enter tickers and weights to get a full risk analysis:
- Key metrics: volatility, beta, Sharpe, max drawdown, VaR(95%)
- Correlation heatmap (RdBu colourscale, −1 to +1)
- Sector exposure donut chart
- Results cached in `session_state["portfolio_risk"]`

#### 6 — Strategy Backtesting (`6_strategy_backtesting.py`)
Configure and run a backtest for any ticker:
- Adjustable RSI buy/sell thresholds, MACD confirmation, MA filter
- Equity curve chart vs initial capital reference line
- Summary metrics: total return, Sharpe, max drawdown, win rate, trade count
- Full trade log table (entry date, exit date, return per trade)
- Results cached in `session_state["backtest_result"]`

#### 7 — Strategy Optimisation (`7_strategy_optimization.py`)
Automatically find optimal parameters:
- Select ticker, period, and optimisation objective (maximise return / Sharpe / minimise drawdown)
- Runs 36-combination grid search
- Displays best parameters and best metric achieved
- Full results table with all 36 combinations, sorted by objective
- Results cached in `session_state["opt_result"]`

#### 8 — Document Library (`8_document_library.py`)
RAG knowledge base management:
- Paste text or upload a `.txt` file with metadata (title, ticker, document type, source, date)
- Semantic search with configurable `top_k`
- Browse all indexed documents with delete capability
- Displays total document count and FAISS index size

#### 9 — AI Chatbot (`9_ai_chatbot.py`)
Conversational analyst powered by GPT-4o:
- Full conversation history maintained in `session_state["chat_history"]`
- Grounded in: RAG-retrieved relevant document chunks + live market quote for the selected ticker
- Displays source citations when RAG context is used
- Pre-set example questions for quick start
- Handles 401 (invalid API key) and 429 (rate limit) explicitly

---

### Components

#### `sidebar.py`
Rendered on every page via `render_sidebar()`. Returns the API base URL.
- Branded header with gradient background
- Backend URL input (reads `BACKEND_URL` env var as default)
- Live backend status pill (green/red) — checks `GET /` with 2s timeout
- Stock universe displayed as styled ticker badges in a 2-column grid
- Custom navigation replacing Streamlit's default sidebar nav: styled page links with hover effects

#### `charts.py`
Plotly chart factories, all using `template="plotly_dark"`:
- `candlestick_chart(df, ticker, show_volume, show_ma, show_bb)` — OHLCV candlestick with optional SMA 50/200 and Bollinger Band overlays, volume bar chart in sub-panel
- `rsi_chart(df)` — RSI line with 70/30 reference bands
- `macd_chart(df)` — MACD line, signal line, and histogram (green above zero / red below)
- `equity_curve_chart(equity_curve, ticker, strategy_name)` — portfolio value over time with initial capital reference line
- `opportunity_bar_chart(opportunities)` — horizontal bar chart coloured by recommendation, reference line at 50
- `correlation_heatmap(corr_matrix)` — annotated heatmap with RdBu colorscale, zmid=0
- `sector_pie_chart(sector_exposure)` — donut chart with label+percent annotations

#### `metrics.py`
- `confidence_gauge(score, recommendation, color)` — Plotly gauge with five colour zones (red 0–34, orange 34–49, yellow 49–64, light green 64–79, green 79–100)
- `metric_card(label, value, delta)` — thin wrapper around `st.metric`
- `recommendation_badge(recommendation, color)` — HTML inline badge with background colour

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=backend --cov=frontend --cov-report=term-missing

# Single test file
pytest tests/test_backtesting_service.py -v
```

**516 tests across 20 files — all passing.**

| Test File | Coverage |
|---|---|
| `test_agents.py` | All 12 agent factory functions — role, goal, backstory, tool config |
| `test_crew_orchestrator.py` | Task graph wiring, successful kickoff, result propagation |
| `test_api_routes.py` | All 19 endpoints — happy path, validation errors, auth, rate limiting, security headers |
| `test_market_routes.py` | Market report, top news, opportunity scanner — caching, refresh, sentiment defaults |
| `test_chat_routes.py` | Chat endpoint — RAG context, history payload, error handling |
| `test_rag_routes.py` | Document ingest, file upload, search, list, delete |
| `test_market_data_service.py` | Quote fetching, OHLCV history, index data, yfinance error handling |
| `test_technical_analysis_service.py` | All indicators computed correctly, empty DataFrame handling |
| `test_sentiment_service.py` | FinBERT classification, aggregation, edge cases (empty articles, missing keys) |
| `test_confidence_score_service.py` | All sub-scorers in isolation, score clamping, weighted combination |
| `test_recommendation_service.py` | All five tiers, boundary scores, colour mapping |
| `test_backtesting_service.py` | Strategy logic, equity curve, Sharpe calculation, no-trade scenarios |
| `test_portfolio_risk_service.py` | VaR, beta, Sharpe, drawdown, zero-weight guard, single-asset portfolio |
| `test_strategy_optimization_service.py` | Grid search, all three objectives, best result selection |
| `test_rag_service.py` | Chunking boundary conditions, add/retrieve, thread safety |
| `test_report_service.py` | GPT-4o call, tenacity retry on rate limit, fallback message |
| `test_news_service.py` | Article fetching, max_articles limit, missing API key handling |
| `test_frontend_components.py` | Chart factories, gauge colour zones, recommendation colour map |
| `test_logger.py` | JSON output format, request_id propagation, ContextVar isolation |
| `test_scheduler.py` | Job registration, scheduler start/stop, job configuration |

---

## CI/CD

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.lock

      - name: Run tests
        env:
          OPENAI_API_KEY: "sk-test-placeholder"
          NEWS_API_KEY: "test-placeholder"
          APP_ENV: "test"
        run: pytest tests/ -q --tb=short
```

Enable branch protection on `main` (GitHub → Settings → Branches) and require the `test` job to pass before merging.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for GPT-4o and embeddings |
| `NEWS_API_KEY` | No | — | NewsAPI.org key — system degrades gracefully without it |
| `OPENAI_MODEL` | No | `gpt-4o` | GPT model used by all agents and chatbot |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model for RAG |
| `FINBERT_MODEL` | No | `ProsusAI/finbert` | HuggingFace FinBERT model identifier |
| `DATABASE_URL` | No | `sqlite:///./data/financial_analyst.db` | SQLAlchemy connection string |
| `ALLOWED_ORIGINS` | No | `http://localhost:8501` | Comma-separated CORS origins |
| `API_SECRET_KEY` | No | *(blank = auth disabled)* | Enables `X-API-Key` header authentication |
| `APP_ENV` | No | `development` | `development` or `production` — affects error detail exposure |
| `SCHEDULER_ENABLED` | No | `true` | Enable APScheduler daily jobs |
| `DAILY_REPORT_HOUR` | No | `8` | Hour (ET) for daily scheduler jobs |
| `DAILY_REPORT_MINUTE` | No | `0` | Minute for daily scheduler jobs |
| `STOCK_UNIVERSE` | No | `AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD,NFLX,INTC` | Comma-separated tickers for opportunity scanner |
| `FAISS_INDEX_PATH` | No | `./data/faiss_index` | Path to persist the FAISS index |
| `RAG_TOP_K` | No | `5` | Number of document chunks returned per RAG query |
| `CHUNK_SIZE` | No | `512` | Character size of each RAG document chunk |
| `CHUNK_OVERLAP` | No | `64` | Character overlap between consecutive chunks |
| `LOG_FORMAT` | No | `json` | `json` for structured output, `text` for human-readable |
| `BACKEND_URL` | No | `http://localhost:8000` | Used by Streamlit frontend to locate the API |

---

## Docker

```bash
# Build and start both services
docker-compose up --build

# Backend only
docker build -t financial-analyst-api .
docker run -p 8000:8000 --env-file .env financial-analyst-api
```

The `docker-compose.yml` starts the FastAPI backend on port 8000. Run the Streamlit frontend separately or extend the compose file with a second service.

---

## License

MIT
