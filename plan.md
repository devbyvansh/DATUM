# Implementation plan

## Completed

- FastAPI application with `/api` prefix, CORS configuration, structured errors, and OpenAPI.
- Local SQLite metadata store for datasets and workflow runs.
- Safe, size-limited ingestion for CSV, Excel, JSON, JSONL, and Parquet.
- Deterministic profiling, quality findings, preparation operations, descriptive statistics, chart data, and verification.
- Analyze-only and prepare-for-ML workflows with persisted status and prepared dataset downloads.
- Focused unit and API tests.
- Real LangGraph Analyze Only and Prepare for ML graphs with typed agent outputs, approval interrupts, and resume.
- Provider boundary with explicit unavailable and failed states.
- OpenAI-compatible calls with optional API keys for local Ollama, bounded retry behavior, and validated model output.
- SQLite-backed LangGraph checkpoints with restart/resume coverage and safe API artifact responses.
- SQLAlchemy-based read-only SQLite, PostgreSQL, and MySQL source connector with bounded query materialization.
- Feature-engineering registry for imputation, indicators, text/date features, one-hot encoding, scaling, and safe ratios.
- Expanded quality scorecards, missingness metrics, Spearman correlation, categorical frequencies, and time trend chart data.

## Remaining

- Replace the in-process background task with a durable worker if deployment requires multi-process execution.
- Add authentication/authorization for non-local deployments.
- Add a persisted SQLAlchemy/Alembic schema when multi-user deployment is required.
- Add a durable LangGraph checkpointer/worker, independent verification paths, target-aware leakage checks, and optional external LLM adapters.
- Run a configured live-provider integration test when credentials and a reachable model endpoint are available.
- Install the optional `sql-postgres` or `sql-mysql` extra when using those source database types.
