import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL, created_at TEXT NOT NULL, metadata TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, mode TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                result TEXT, error TEXT
            );
            """
        )
        for statement in (
            "ALTER TABLE runs ADD COLUMN stage TEXT NOT NULL DEFAULT 'pending'",
            "ALTER TABLE runs ADD COLUMN stage_status TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE runs ADD COLUMN approval_status TEXT",
        ):
            try:
                self.connection.execute(statement)
            except sqlite3.OperationalError:
                pass
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add_dataset(self, dataset: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
            (
                dataset["id"],
                dataset["filename"],
                dataset["path"],
                dataset["size_bytes"],
                dataset["created_at"],
                json.dumps(dataset["metadata"]),
            ),
        )
        self.connection.commit()

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT id, filename, size_bytes, created_at, metadata "
            "FROM datasets ORDER BY created_at DESC"
        ).fetchall()
        datasets = []
        for row in rows:
            dataset = dict(row)
            dataset["metadata"] = json.loads(dataset["metadata"])
            datasets.append(dataset)
        return datasets

    def add_run(self, run: dict[str, Any]) -> None:
        self.connection.execute(
            (
                "INSERT INTO runs "
                "(id, dataset_id, mode, status, created_at, updated_at, result, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                run["id"],
                run["dataset_id"],
                run["mode"],
                run["status"],
                run["created_at"],
                run["updated_at"],
                None,
                None,
            ),
        )
        self.connection.commit()

    def update_run(
        self,
        run_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        stage: str | None = None,
        stage_status: dict[str, str] | None = None,
        approval_status: str | None = None,
    ) -> None:
        current = self.get_run(run_id) or {}
        stored_result = (
            json.dumps(result)
            if result is not None
            else json.dumps(current.get("result"))
            if current.get("result") is not None
            else None
        )
        self.connection.execute(
            (
                "UPDATE runs SET status = ?, updated_at = ?, result = ?, error = ?, "
                "stage = ?, stage_status = ?, approval_status = ? WHERE id = ?"
            ),
            (
                status,
                utc_now(),
                stored_result,
                error,
                stage or current.get("stage", "pending"),
                json.dumps(stage_status or current.get("stage_status", {})),
                approval_status if approval_status is not None else current.get("approval_status"),
                run_id,
            ),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["stage_status"] = json.loads(result["stage_status"] or "{}")
        return result
