"""SQLite persistence for completed agentic pipeline runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pipeline_state import PipelineRun


DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "pipeline_history.db"


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_history_db() -> None:
    """Create the pipeline history table if needed."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                rows INTEGER NOT NULL,
                columns INTEGER NOT NULL,
                summary TEXT,
                recommendations TEXT,
                agent_results TEXT,
                report_markdown TEXT
            )
            """
        )


def save_pipeline_run(run: PipelineRun) -> None:
    """Persist a completed run summary to SQLite."""
    init_history_db()
    agent_payload: dict[str, Any] = {
        name: {
            "status": result.status,
            "summary": result.summary,
            "findings": result.findings,
            "metrics": result.metrics,
            "duration_seconds": result.duration_seconds,
        }
        for name, result in run.agent_results.items()
    }
    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs (
                run_id, dataset_name, created_at, completed_at, status, rows, columns,
                summary, recommendations, agent_results, report_markdown
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.dataset_name,
                run.created_at,
                run.completed_at,
                run.current_stage.value,
                run.active_df.shape[0],
                run.active_df.shape[1],
                run.executive_summary,
                json.dumps(run.recommendations),
                json.dumps(agent_payload, default=str),
                run.report_markdown,
            ),
        )


def load_recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Load recent completed pipeline run summaries."""
    init_history_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT run_id, dataset_name, created_at, completed_at, status, rows, columns,
                   summary, recommendations
            FROM pipeline_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["recommendations"] = json.loads(item.get("recommendations") or "[]")
        except json.JSONDecodeError:
            item["recommendations"] = []
        results.append(item)
    return results
