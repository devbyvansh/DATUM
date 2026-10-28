# SQL data sources

DATUM keeps its own metadata and workflow databases in `data/`. SQL source databases are separate and are accessed read-only by the connector layer.

## Supported connections

- SQLite: `sqlite:///C:/path/to/source.db`
- PostgreSQL: `postgresql+psycopg://user:password@host:5432/database`
- MySQL: `mysql+pymysql://user:password@host:3306/database`

Install the matching optional driver when needed:

```powershell
python -m pip install psycopg[binary]
python -m pip install pymysql
```

The API accepts a connection URL only in the request body and never returns it:

- `POST /api/sql/test` — test the connection.
- `POST /api/sql/tables` — list tables and views.
- `POST /api/sql/table` — inspect safe table metadata.
- `POST /api/sql/query` — execute one bounded read-only `SELECT` or `WITH` query.

Mutating statements, multiple statements, unsupported schemes, and unsafe identifiers are rejected. Use database permissions that are read-only in addition to the application checks. Credentials should come from a secret store or protected environment configuration, not source code.
