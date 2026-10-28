# Security

The backend is local-only by default and has no authentication. Do not expose it publicly without authentication, authorization, TLS, and a durable job/persistence design.

Uploads are extension-checked, size-limited, parsed by supported readers, stored under configured directories, and only bounded evidence is sent to a configured LLM. Preview output is bounded and ordinary metadata responses do not expose internal storage paths. Error responses omit stack traces. Do not log complete rows or put credentials in `.env.example`. Retention and deletion policy for uploaded files should be added before production deployment.

SQL sources are restricted to supported SQLAlchemy schemes, one bounded read-only `SELECT` or `WITH` query, and safe reflected identifiers. Use read-only database credentials as an additional protection. Connection URLs are not persisted in dataset metadata or returned by the API.
