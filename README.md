# DATUM backend

DATUM is a local-first data analysis backend. The deterministic engine ingests tabular files, profiles quality, prepares a separate ML-ready output, calculates statistics and Plotly-compatible chart data, and independently verifies the generated report. Real LangGraph Analyze Only and Prepare for ML workflows coordinate four typed agents. LLM explanations are optional and are explicitly marked unavailable when no provider is configured.

## Run on Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m uvicorn backend.app.main:app --reload
```

OpenAPI is available at `http://127.0.0.1:8000/api/docs`. The backend is currently intended for a single local user and has no authentication.

When a Prepare for ML request contains transformations, the run pauses at `needs_approval`. Retrieve the plan with `GET /api/runs/{run_id}/preparation-plan`, then approve or reject it with `POST /api/runs/{run_id}/approval` and `{"decision":"approved"}` or `{"decision":"rejected"}`.

## API example

```powershell
$upload = Invoke-RestMethod -Uri http://127.0.0.1:8000/api/datasets -Method Post -Form @{ file = Get-Item .\sample.csv }
$workflow = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/datasets/$($upload.id)/workflows" -Method Post -ContentType application/json -Body '{"mode":"analyze_only"}'
Invoke-RestMethod "http://127.0.0.1:8000/api/runs/$($workflow.run_id)"
```

Supported formats are CSV, XLSX/XLS, JSON, JSONL, and Parquet. The default upload limit is 50 MB. Read-only SQLite, PostgreSQL, and MySQL query endpoints are also available. Use `prepare_for_ml` with a body such as `{"mode":"prepare_for_ml","preparation":{"drop_duplicates":true,"impute_median":["age"],"one_hot_encode":["segment"]}}`.

Both workflows generate a portable, dataset-specific Jupyter notebook after completion. Download it with `GET /api/runs/{run_id}/notebook`. The notebook includes only applicable analyses and transformations that actually ran; cells are not executed automatically.

To enable OpenRouter through the existing OpenAI-compatible provider, copy `.env.example` to `.env` and set your key only in that untracked file. The example uses `https://openrouter.ai/api/v1`, `nvidia/nemotron-3-ultra:free`, a 60-second timeout, and two retries. Verify that the selected model is currently available on OpenRouter before relying on it. For Ollama, use `LLM_BASE_URL=http://localhost:11434/v1` and the exact installed model tag. Deterministic processing does not require these values.

## Development

```powershell
pytest
ruff check .
```

See [docs/how_datum_works.md](docs/how_datum_works.md), [docs/api_guide.md](docs/api_guide.md), [docs/ai_setup.md](docs/ai_setup.md), and [docs/security.md](docs/security.md).

SQL setup and preparation operations are described in [docs/sql_setup.md](docs/sql_setup.md) and [docs/feature_engineering.md](docs/feature_engineering.md).

## Frontend

The React frontend is in [frontend/](frontend/). Start the backend first, then run:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

The frontend connects to the existing `/api` routes and supports upload, analysis
and preparation runs, approval, report evidence, SQL snapshots, prepared CSV
downloads, and notebook downloads. It does not store or expose the backend LLM key.

## Project structure

The most important code is grouped by responsibility: deterministic processing is in `backend/app/data_processing.py`, SQLite persistence is in `backend/app/database.py`, the four agents are in `backend/app/agents/`, and LangGraph workflows are in `backend/app/workflows/`. API routes and the FastAPI application remain in `backend/app/main.py`, so the start command does not change. See [docs/project_structure.md](docs/project_structure.md) for the complete beginner-friendly tree.
