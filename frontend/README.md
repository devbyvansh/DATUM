# DATUM frontend

This is the React + TypeScript + Vite frontend for the existing DATUM FastAPI
backend. It uses the backend routes under `/api` and does not contain provider
credentials or SQL connection strings in persistent browser state.

## Run locally

From `D:\DATUM\frontend`:

```powershell
npm install
npm run dev
```

The Vite development server proxies `/api` to
`http://127.0.0.1:8000`. Start the backend separately:

```powershell
python -m uvicorn backend.app.main:app --reload
```

For another backend URL, set `VITE_API_BASE_URL` when building or running the
frontend. The OpenRouter key remains a backend-only `.env` setting.
