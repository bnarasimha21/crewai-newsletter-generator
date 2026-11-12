"""SQLite run history: one row per `generate()` call, read by the Streamlit
history page and written by the pipeline and CLI."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_PRICING_PATH = Path(__file__).parent / "pricing.yaml"


@lru_cache(maxsize=1)
def _load_pricing() -> dict:
    """Load newsletter/pricing.yaml once (see that file for the source/date)."""
    return yaml.safe_load(_PRICING_PATH.read_text(encoding="utf-8"))


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    model: str | None = None,
) -> float | None:
    """Estimate USD cost from token counts using newsletter/pricing.yaml.

    Returns None if `model` (or the configured default_model) is not in the
    pricing table, rather than guessing a rate - an unpriced run should show
    up as "cost unknown", not a silently wrong number.
    """
    pricing = _load_pricing()
    model = model or pricing.get("default_model")
    rates = pricing.get("models", {}).get(model)
    if rates is None:
        return None
    return (prompt_tokens / 1_000_000) * rates["input"] + (
        completion_tokens / 1_000_000
    ) * rates["output"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost REAL,
    error TEXT
);
"""


@dataclass
class RunRecord:
    id: int
    topic: str
    started_at: str
    finished_at: str | None
    status: str
    run_dir: str
    tokens: int
    estimated_cost: float | None
    error: str | None


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # `timeout=30` (seconds a writer waits on a lock before raising
    # OperationalError, not the HTTP kind) plus WAL mode: the sqlite3
    # default of 5s was found, under an adversarial review of this
    # rewrite, to be tight enough that two generate() calls (or a run and
    # a Streamlit history-page read) landing within the same ~5s window
    # could raise "database is locked" and crash generate() with an
    # untyped exception, defeating the whole typed-error contract
    # (SearchFailed/RenderFailed/Timeout) the rewrite was built around.
    # WAL mode lets readers (list_runs) proceed without blocking on a
    # writer at all, which is the more common collision (Streamlit's
    # history table polling while a run is in flight).
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


@contextmanager
def _open(db_path: Path):
    conn = _connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def start_run(db_path: Path, *, topic: str, started_at: str, run_dir: str) -> int:
    """Insert a new 'running' row, returning its id."""
    with _open(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (topic, started_at, status, run_dir, tokens) "
            "VALUES (?, ?, 'running', ?, 0)",
            (topic, started_at, run_dir),
        )
        conn.commit()
        return int(cursor.lastrowid)


def finish_run(
    db_path: Path,
    run_id: int,
    *,
    finished_at: str,
    status: str,
    tokens: int = 0,
    estimated_cost: float | None = None,
    error: str | None = None,
) -> None:
    """Update a run row to its terminal status: 'success', 'failed' or 'timeout'."""
    with _open(db_path) as conn:
        conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, tokens = ?, "
            "estimated_cost = ?, error = ? WHERE id = ?",
            (finished_at, status, tokens, estimated_cost, error, run_id),
        )
        conn.commit()


def list_runs(db_path: Path, limit: int = 50) -> list[RunRecord]:
    if not db_path.exists():
        return []
    with _open(db_path) as conn:
        rows = conn.execute(
            "SELECT id, topic, started_at, finished_at, status, run_dir, tokens, "
            "estimated_cost, error FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [RunRecord(*row) for row in rows]
