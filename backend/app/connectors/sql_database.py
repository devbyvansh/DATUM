from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

from backend.app.errors import DatumError

ALLOWED_SCHEMES = {"sqlite", "postgresql", "postgresql+psycopg", "mysql", "mysql+pymysql"}
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke|attach|pragma)\b",
    re.IGNORECASE,
)


def create_read_only_engine(connection_url: str) -> Engine:
    try:
        parsed = make_url(connection_url)
    except Exception as exc:
        raise DatumError("The SQL connection URL is invalid.", "invalid_sql_connection") from exc
    if parsed.drivername not in ALLOWED_SCHEMES:
        raise DatumError(
            "Only SQLite, PostgreSQL, and MySQL analytical connections are supported.",
            "unsupported_sql_database",
        )
    try:
        return create_engine(parsed, pool_pre_ping=True)
    except Exception as exc:
        raise DatumError("The SQL connection could not be created.", "sql_connection_failed") from exc


def test_connection(connection_url: str) -> dict[str, Any]:
    engine = create_read_only_engine(connection_url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "connected", "dialect": engine.dialect.name}
    except Exception as exc:
        raise DatumError("The SQL connection test failed.", "sql_connection_failed") from exc
    finally:
        engine.dispose()


def list_tables(connection_url: str, schema: str | None = None) -> dict[str, Any]:
    engine = create_read_only_engine(connection_url)
    try:
        inspector = inspect(engine)
        return {
            "dialect": engine.dialect.name,
            "schema": schema,
            "tables": inspector.get_table_names(schema=schema),
            "views": inspector.get_view_names(schema=schema),
        }
    except Exception as exc:
        raise DatumError("The SQL tables could not be listed.", "sql_metadata_failed") from exc
    finally:
        engine.dispose()


def describe_table(
    connection_url: str, table: str, schema: str | None = None
) -> dict[str, Any]:
    if not _safe_identifier(table):
        raise DatumError("The table name is invalid.", "invalid_sql_identifier")
    engine = create_read_only_engine(connection_url)
    try:
        columns = inspect(engine).get_columns(table, schema=schema)
        return {
            "table": table,
            "schema": schema,
            "columns": [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": bool(column.get("nullable", True)),
                }
                for column in columns
            ],
        }
    except Exception as exc:
        raise DatumError("The SQL table could not be described.", "sql_metadata_failed") from exc
    finally:
        engine.dispose()


def query_dataframe(
    connection_url: str, query: str, limit: int = 1000
) -> dict[str, Any]:
    if not _is_read_only_query(query):
        raise DatumError(
            "Only one read-only SELECT or WITH query is allowed.",
            "unsafe_sql_query",
        )
    bounded_limit = max(1, min(limit, 10_000))
    engine = create_read_only_engine(connection_url)
    try:
        bounded_query = (
            f"SELECT * FROM ({query.rstrip(';')}) AS datum_limited "
            "LIMIT :datum_limit"
        )
        frame = _read_query(engine, bounded_query, bounded_limit)
        frame.columns = _unique_columns([str(column) for column in frame.columns])
        return {
            "dialect": engine.dialect.name,
            "row_count": len(frame),
            "columns": list(frame.columns),
            "rows": json.loads(frame.to_json(orient="records", date_format="iso")),
        }
    except Exception as exc:
        raise DatumError("The SQL query could not be executed.", "sql_query_failed") from exc
    finally:
        engine.dispose()


def query_frame(connection_url: str, query: str, limit: int = 1000) -> pd.DataFrame:
    if not _is_read_only_query(query):
        raise DatumError(
            "Only one read-only SELECT or WITH query is allowed.",
            "unsafe_sql_query",
        )
    bounded_limit = max(1, min(limit, 10_000))
    engine = create_read_only_engine(connection_url)
    try:
        bounded_query = (
            f"SELECT * FROM ({query.rstrip(';')}) AS datum_limited "
            "LIMIT :datum_limit"
        )
        return _read_query(engine, bounded_query, bounded_limit)
    except Exception as exc:
        if isinstance(exc, DatumError):
            raise
        raise DatumError("The SQL query could not be executed.", "sql_query_failed") from exc
    finally:
        engine.dispose()


def _is_read_only_query(query: str) -> bool:
    normalized = query.strip()
    return (
        bool(normalized)
        and normalized.upper().startswith(("SELECT", "WITH"))
        and ";" not in normalized.rstrip(";")
        and not FORBIDDEN_SQL.search(normalized)
    )


def _read_query(engine: Engine, query: str, limit: int) -> pd.DataFrame:
    with engine.connect() as connection:
        frame = pd.read_sql_query(
            text(query),
            connection,
            params={"datum_limit": limit},
        )
    frame.columns = _unique_columns([str(column) for column in frame.columns])
    return frame


def _safe_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for column in columns:
        count = counts.get(column, 0)
        result.append(column if count == 0 else f"{column}.{count}")
        counts[column] = count + 1
    return result
