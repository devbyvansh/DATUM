# Architecture

`backend.app.main` owns HTTP validation and workflow entry points. `backend.app.data_processing` is deterministic and contains ingestion, profiling, quality, preparation, statistics, and verification functions. `backend.app.database.Store` persists dataset and run metadata in SQLite. Uploaded files are immutable; prepare-for-ML writes a separate CSV under `data/prepared`.

`backend.app.features` contains registered, deterministic feature-engineering operations. `backend.app.connectors.sql_database` supports bounded, read-only SQL source inspection and materialization for SQLite, PostgreSQL, and MySQL when the matching SQLAlchemy driver is installed.

Workflow stages are represented by persisted run statuses (`queued`, `running`, `needs_approval`, `rejected`, `completed`, `failed`). A run always computes quality from the source and statistics/verification from the selected working frame. LangGraph checkpoints are stored in the separate SQLite path configured by `CHECKPOINT_DATABASE_PATH`, allowing approval resume after application reinitialization on the same machine. The in-process FastAPI background task is adequate for local development; it must be replaced by a durable queue for horizontally scaled deployment.

When a run reaches its final stage, `notebook_generator.py` writes a portable, run-specific
Jupyter notebook under `data/reports/`. Section selection uses the actual dataset column
types and values. The notebook includes only transformations recorded as executed after
approval; rejected or skipped preparation is documented without being emitted as executable
transformation code. The notebook is downloaded through
`GET /api/runs/{run_id}/notebook` and is never executed by the server.
