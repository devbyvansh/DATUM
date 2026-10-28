# DATUM project structure

This tree keeps the backend code grouped by what it does. The API entry point remains `backend.app.main:app`, so existing run commands continue to work.

```text
DATUM/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app and API routes
│   │   ├── config.py               # Reads application settings
│   │   ├── errors.py               # Defines safe application errors
│   │   ├── database.py             # Stores dataset and run information
│   │   ├── data_processing.py      # Loads, profiles, and processes datasets
│   │   ├── ai_provider.py          # Connects DATUM to an LLM
│   │   ├── features.py             # Registered feature-engineering operations
│   │   ├── notebook_generator.py    # Builds portable notebooks from completed runs
│   │   ├── connectors/             # Safe SQL source connections and queries
│   │   │   └── sql_database.py
│   │   ├── schemas/                # Defines agent and provider data shapes
│   │   │   └── agent.py
│   │   ├── agents/                 # Contains the four analysis agents
│   │   │   └── agents.py
│   │   └── workflows/              # LangGraph Analyze Only and Prepare for ML graphs
│   │       └── graphs.py
│   └── __init__.py
├── data/
│   ├── uploads/                    # Original uploaded files (generated, not source code)
│   ├── prepared/                   # Prepared CSV files (generated)
│   ├── reports/                    # Generated JSON reports
│   ├── datum.db                    # Local application database
│   └── checkpoints.db              # LangGraph checkpoint database
├── tests/                          # Automated backend tests
├── frontend/                       # React UI for upload, runs, SQL, and reports
│   ├── src/App.tsx                 # Main screens and backend interactions
│   ├── src/api.ts                  # Typed API client
│   └── src/styles.css              # DATUM application design system
├── docs/                           # Setup, API, architecture, security, and handoff docs
├── .env.example                    # Safe configuration template
├── plan.md                         # Completed work and remaining work
├── pyproject.toml                  # Dependencies, package settings, test and lint config
└── README.md                       # Setup and usage guide
```

## Where to start

- Start the server from [backend/app/main.py](../backend/app/main.py).
- Read settings in [backend/app/config.py](../backend/app/config.py).
- Read the deterministic engine in [backend/app/data_processing.py](../backend/app/data_processing.py).
- Read feature engineering in [backend/app/features.py](../backend/app/features.py).
- Read SQL source support in [backend/app/connectors/sql_database.py](../backend/app/connectors/sql_database.py).
- Read database operations in [backend/app/database.py](../backend/app/database.py).
- Read the four agent roles in [backend/app/agents/agents.py](../backend/app/agents/agents.py).
- Read provider configuration and model calls in [backend/app/ai_provider.py](../backend/app/ai_provider.py).
- Read workflow order and approval interrupts in [backend/app/workflows/graphs.py](../backend/app/workflows/graphs.py).
- Run tests from [tests/](../tests/).

Folders under `data/` are runtime storage. Do not put source code or secrets there. The local database, checkpoint database, uploaded files, prepared files, reports, and `.env` are excluded from version control. VS Code hides these generated files with [.vscode/settings.json](../.vscode/settings.json), but the application can still use them.
