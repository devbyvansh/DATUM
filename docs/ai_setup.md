# AI pipeline

The current release keeps calculations deterministic and reports `ai.status = unavailable` when `LLM_PROVIDER=none` or when the configured provider is unavailable. This prevents fabricated narratives and keeps analysis usable without credentials.

Four typed agents live in `backend.app.agents.agents`: quality, preparation, statistical, and verification. They receive calculated evidence and return Pydantic-validated outputs. The unavailable provider path creates only deterministic, evidence-linked summaries; it never claims that a model was called.

The intended provider boundary is after profiling and deterministic calculations: a future provider may receive bounded metadata and evidence, validate structured responses, and explain existing findings. It must never calculate numbers, execute uploaded content, or execute model-generated code. Provider failures must remain visible in run status and report metadata. An adapter must implement `LLMProvider.invoke(payload, output_schema)` and return data that validates against the requested schema. Network retries/timeouts belong inside the adapter, not the deterministic processing layer.

The OpenAI-compatible adapter sends the bearer header only when `LLM_API_KEY` is non-empty, so local Ollama-compatible endpoints can run without a dummy credential. Permanent HTTP authentication, billing, and client errors are not retried; timeout and transient server failures use the bounded `LLM_MAX_RETRIES` setting.

## OpenRouter configuration

OpenRouter can be used through the existing OpenAI-compatible adapter. Copy `.env.example`
to `.env`, then set the real key only in the local `.env` file:

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-3-ultra:free
LLM_API_KEY=replace_with_your_openrouter_key
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
```

The model name is configuration, not a guarantee of continued availability. OpenRouter
authentication, rate-limit, billing, unavailable-model, timeout, and malformed-response
errors remain visible through the provider status; DATUM does not turn them into successful
AI output. The API key is never written to reports or notebooks, and should not be placed
in source code, frontend variables, documentation, logs, or commits.

The adapter sends requests to `/chat/completions` with the configured model and timeout.
Authentication and other permanent client errors are not retried. Timeouts, transient
HTTP failures, and malformed responses use the bounded retry setting.
