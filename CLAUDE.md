# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Web Application

```bash
# Install dependencies
source .venv/bin/activate && pip install -r requirements.txt

# Run Flask development server
python run.py

# Import CSV data into SQLite
python scripts/import_csv.py
python scripts/import_csv.py --period 2021_01_01-2024_12_31
python scripts/import_csv.py --dry-run

# Run tests
python -m pytest tests/ -v

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

### Configuration
- Environment variables: Set `BINANCE_API_KEY` and `BINANCE_API_SECRET` in `.env` file
- App config: `app/config.py` (DB path, API keys, ML parameters, trading constants)
- Database: SQLite at `instance/tradebot.db`

## Architecture

### Project Structure

```
app/                    # Flask application
├── __init__.py         # create_app() factory
├── config.py           # Configuration classes
├── extensions.py       # SQLAlchemy, APScheduler instances
├── models/             # SQLAlchemy ORM models
├── services/           # Business logic (pure Python, no Flask dependency)
├── views/              # Flask Blueprints (routes)
├── tasks/              # Background jobs (APScheduler)
├── templates/          # Jinja2 templates (Bootstrap 5)
└── static/             # CSS, JS

scripts/                # CLI utilities
tests/                  # pytest test suite
migrations/             # Alembic database migrations
legacy/                 # Archived original code (reference only)
datasets/               # CSV source data
instance/               # SQLite database (gitignored)
```

### Database Models (SQLAlchemy)

- **Candle** — Multi-timeframe OHLCV data (symbol, timeframe, open_time unique)
- **Level** — Technical levels (HTF, Fractal, Fibonacci, Volume Profile) with touch tracking
- **Feature** — Computed ML features per candle (1:1 with candle)
- **MLModel** — Trained model registry with metrics
- **Prediction** — Prediction history with probabilities
- **PipelineRun** — Pipeline execution log

### Views (Blueprints)

- `dashboard_bp` → `/` — Main dashboard with data coverage stats
- `data_bp` → `/data/` — Data management and status
- `charts_bp` → `/charts/` — TradingView Lightweight Charts
- `features_bp` → `/features/` — Feature engineering status
- `models_bp` → `/models/` — ML model registry
- `api_bp` → `/api/` — JSON API endpoints

### Key API Endpoints

- `GET /api/health` — Health check
- `GET /api/stats` — Dashboard statistics
- `GET /api/candles?tf=1h&start=...&end=...&limit=500` — Candle data
- `GET /api/levels?start=...&end=...&active_only=true` — Level data

### Technology Stack

| Concern | Choice |
|---------|--------|
| Database | SQLite + SQLAlchemy ORM |
| Web framework | Flask + Blueprints |
| UI | Bootstrap 5 + Jinja2 |
| Charts | Lightweight Charts (TradingView) |
| Background tasks | APScheduler |
| Testing | pytest |
| Migrations | Alembic |
| Container | Podman |

## Project Context

Bitcoin (BTCUSDT) fractal prediction system using ML. Combines multi-timeframe technical analysis with level tracking to predict swing highs/lows.

### Core Concepts

- **Fractals**: 5-candle swing highs/lows detected by indicator pipeline
- **Levels**: HTF structure, Fibonacci retracements, Volume Profile (POC/VAH/VAL)
- **Features**: Candle N-1 interactions with zones derived from N-2 close price
- **Targets**: 0=no_fractal, 1=bullish_fractal (swing low), 2=bearish_fractal (swing high)

### Legacy Code

All original business logic is preserved in `legacy/` for reference during service migration. Key files:
- `legacy/indicators.py` — Fractal detection, HTF levels, Fibonacci, Volume Profile
- `legacy/create_ml_features.py` — Feature engineering pipeline
- `legacy/target_variable.py` — Target variable generation
- `legacy/ml_models/` — Model training, evaluation, prediction

### Build Phases

1. **Phase 1** (DONE): Project restructure + SQLite foundation
2. **Phase 2**: Dashboard + data status views
3. **Phase 3**: Indicators pipeline integration
4. **Phase 4**: Interactive charts
5. **Phase 5**: Feature engineering pipeline
6. **Phase 6**: ML training & evaluation UI
7. **Phase 7**: Predictions & automation
8. **Phase 8**: Polish, testing & deployment
