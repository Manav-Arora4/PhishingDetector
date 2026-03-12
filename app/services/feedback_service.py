"""Feedback and prediction logging service backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class FeedbackService:
    """Persist model predictions and analyst feedback for retraining."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    text TEXT,
                    url TEXT,
                    decision TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    explanation_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    prediction_id INTEGER,
                    text TEXT,
                    url TEXT,
                    user_label INTEGER NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                );
                """
            )

    def log_prediction(self, *, text: str | None, url: str | None, decision: str, risk_score: float, explanation: dict) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (text, url, decision, risk_score, explanation_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (text, url, decision, risk_score, json.dumps(explanation, sort_keys=True)),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def record_feedback(
        self,
        *,
        user_label: int,
        prediction_id: int | None = None,
        text: str | None = None,
        url: str | None = None,
        notes: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback (prediction_id, text, url, user_label, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (prediction_id, text, url, user_label, notes),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_feedback_samples(self, limit: int = 100) -> list[tuple[str, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT COALESCE(feedback.text, predictions.text) AS text, feedback.user_label AS user_label
                FROM feedback
                LEFT JOIN predictions ON predictions.id = feedback.prediction_id
                WHERE COALESCE(feedback.text, predictions.text) IS NOT NULL
                ORDER BY feedback.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(str(row["text"]), int(row["user_label"])) for row in rows if row["text"]]
