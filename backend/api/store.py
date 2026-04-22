"""
Thin SQLite layer for benchmark runs.

One table: `runs`. One row per submission. JSON blob for `config` so the
schema doesn't have to track every UI tweak.

We intentionally do NOT use SQLAlchemy or any ORM — this is a local single-
file DB backing a short-lived demo app. A raw sqlite3 wrapper is cleaner and
easier for the course reviewer to read.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    config_json  TEXT NOT NULL,
    job_dir      TEXT NOT NULL,
    exit_code    INTEGER,
    error        TEXT
);
"""

VALID_STATUSES = {"queued", "running", "done", "user_error", "crashed", "timeout"}


@dataclass
class Run:
    id: str
    status: str
    created_at: str
    finished_at: str | None
    config: dict[str, Any]
    job_dir: str
    exit_code: int | None
    error: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        return cls(
            id=row["id"],
            status=row["status"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            config=json.loads(row["config_json"]),
            job_dir=row["job_dir"],
            exit_code=row["exit_code"],
            error=row["error"],
        )


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create(self, *, run_id: str, config: dict[str, Any], job_dir: Path) -> Run:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, status, created_at, config_json, job_dir) VALUES (?, 'queued', ?, ?, ?)",
                (run_id, now, json.dumps(config), str(job_dir)),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> Run | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.from_row(row) if row else None

    def set_status(self, run_id: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))

    def mark_finished(
        self,
        run_id: str,
        *,
        status: str,
        exit_code: int | None,
        error: str | None = None,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, exit_code = ?, error = ? WHERE id = ?",
                (status, now, exit_code, error, run_id),
            )

    def list_recent(self, limit: int = 50) -> list[Run]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [Run.from_row(r) for r in rows]
