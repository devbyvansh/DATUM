# Frontend handoff

The backend is available under `/api`.

The production frontend lives in `frontend/` and uses React, TypeScript, and
Vite. From that directory, run `npm install` and `npm run dev`; the development
server proxies `/api` to `http://127.0.0.1:8000`. It intentionally keeps SQL
connection input in memory only and never exposes the backend LLM provider key.
Provider configuration remains a backend-only concern and is not shown as a
frontend settings screen.

1. `POST /datasets` uploads a multipart file and returns `id`, `filename`, `size_bytes`, and profile metadata.
2. `GET /datasets/{id}` returns safe metadata; `GET /datasets/{id}/preview?limit=20` returns bounded rows.
3. `POST /datasets/{id}/workflows` accepts `{"mode":"analyze_only"}` or `{"mode":"prepare_for_ml","preparation":{...}}`.
4. `GET /runs/{id}` exposes `status`, `stage`, `stage_status`, `approval_status`, and the result when available.
5. A prepare run may return `needs_approval`; retrieve its plan with `GET /runs/{id}/preparation-plan` and submit `{"decision":"approved"}` or `{"decision":"rejected"}` to `POST /runs/{id}/approval`.
6. Use `/runs/{id}/statistics`, `/verification`, and `/report` for calculated output.
7. `GET /runs/{id}/download` downloads only a completed prepared CSV.
8. `GET /runs/{id}/notebook` downloads the completed dataset-specific Jupyter notebook for either workflow.

The frontend uses these responses as evidence. It does not create recent-run
history because the current backend has no list-runs endpoint, and it does not
show a prepared-data download until the run is completed.

AI status is explicit in reports: `unavailable` means deterministic fallback, `completed` means a provider response was validated, and `failed` means provider execution failed. Errors use `{"detail":{"code":"...","message":"..."}}`. Authentication is not implemented; use only in a trusted local environment.
