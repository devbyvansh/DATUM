# API contract

All endpoints use `/api`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| POST | `/datasets` | Multipart dataset upload |
| GET | `/datasets/{id}` | Metadata and profile |
| GET | `/datasets/{id}/preview?limit=20` | Bounded records preview |
| POST | `/datasets/{id}/workflows` | Start `analyze_only` or `prepare_for_ml` |
| GET | `/runs/{id}` | Status and result |
| GET | `/runs/{id}/report` | Completed report |
| GET | `/runs/{id}/download` | Prepared CSV download |
| GET | `/runs/{id}/notebook` | Download the completed, dataset-specific Jupyter notebook |
| GET | `/runs/{id}/preparation-plan` | Preparation plan or availability status |
| POST | `/runs/{id}/approval` | Approve or reject a paused preparation graph |
| GET | `/runs/{id}/statistics` | Calculated statistics and charts |
| GET | `/runs/{id}/verification` | Verification checks |
| POST | `/sql/test` | Test a supported analytical SQL connection |
| POST | `/sql/tables` | List accessible tables and views |
| POST | `/sql/table` | Inspect table columns and SQL types |
| POST | `/sql/query` | Run one bounded read-only query |
| POST | `/sql/datasets` | Materialize a bounded read-only query as a DATUM dataset |

SQL request bodies contain `connection_url`, and query requests also contain `query` and optional `limit`. Connection URLs are never stored in dataset metadata or returned in responses. Errors use `{"detail":{"code":"...","message":"..."}}`. A prepare run with consequential operations returns `needs_approval` before transformations. No LLM credentials are required for deterministic operation. Workflow responses include quality findings, transformations, statistics, Plotly-compatible chart records, verification checks, and an explicit `ai.status`.

Completed runs include `result.notebook` with `available`, `filename`, and the run-scoped download URL. The notebook is generated from the actual profile, quality checks, statistics, executed transformations, and verification result.
