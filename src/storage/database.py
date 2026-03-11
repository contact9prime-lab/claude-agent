"""Local SQLite storage for transcripts, tasks, hashtags, and sessions.

SQLite is the perfect fit here:
- Zero config, no server, single file
- Runs on desktop and mobile
- Fast enough for real-time inserts
- Full-text search for transcript queries
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.models import (
    Hashtag,
    Insight,
    Priority,
    Session,
    Task,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    title TEXT DEFAULT '',
    total_speech_seconds REAL DEFAULT 0,
    transcript TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    decisions TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    description TEXT NOT NULL,
    assignee TEXT DEFAULT '',
    priority TEXT DEFAULT 'medium',
    due_hint TEXT DEFAULT '',
    completed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS hashtags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    tag TEXT NOT NULL,
    context TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_completed ON tasks(completed);
CREATE INDEX IF NOT EXISTS idx_hashtags_tag ON hashtags(tag);
CREATE INDEX IF NOT EXISTS idx_hashtags_session ON hashtags(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

-- Full-text search on transcripts
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    session_id,
    content,
    tokenize='porter'
);
"""


class Database:
    """Local SQLite database for the desk voice agent."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the database connection and ensure schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        self._conn.executescript(SCHEMA)
        logger.info("Database connected: %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self) -> int:
        """Create a new active session. Returns the session ID."""
        cursor = self.conn.execute(
            "INSERT INTO sessions (started_at, status) VALUES (?, 'active')",
            (datetime.now().isoformat(),),
        )
        self.conn.commit()
        session_id = cursor.lastrowid
        logger.info("Created session: %d", session_id)
        return session_id

    def end_session(self, session_id: int) -> None:
        """Mark a session as ended."""
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, status = 'ended' WHERE id = ?",
            (datetime.now().isoformat(), session_id),
        )
        self.conn.commit()

    def update_session_summary(
        self,
        session_id: int,
        title: str,
        summary: str,
        decisions: list[str],
    ) -> None:
        """Update session with final summary (after Gemini summarization)."""
        self.conn.execute(
            """UPDATE sessions
            SET title = ?, summary = ?, decisions = ?, status = 'summarized'
            WHERE id = ?""",
            (title, summary, json.dumps(decisions), session_id),
        )
        self.conn.commit()

    def append_transcript(
        self, session_id: int, text: str, speech_seconds: float
    ) -> None:
        """Append transcribed text to a session's running transcript."""
        self.conn.execute(
            """UPDATE sessions
            SET transcript = transcript || ? || char(10),
                total_speech_seconds = total_speech_seconds + ?
            WHERE id = ?""",
            (text, speech_seconds, session_id),
        )
        # Update FTS index
        self.conn.execute(
            "INSERT INTO transcript_fts (session_id, content) VALUES (?, ?)",
            (str(session_id), text),
        )
        self.conn.commit()

    def get_session(self, session_id: int) -> Optional[dict]:
        """Get a session by ID."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_active_session(self) -> Optional[dict]:
        """Get the currently active session, if any."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions."""
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def add_task(self, task: Task, session_id: int) -> int:
        """Add a task linked to a session."""
        cursor = self.conn.execute(
            """INSERT INTO tasks (session_id, description, assignee, priority, due_hint, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                task.description,
                task.assignee,
                task.priority.value,
                task.due_hint,
                task.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def complete_task(self, task_id: int) -> None:
        """Mark a task as completed."""
        self.conn.execute(
            "UPDATE tasks SET completed = 1 WHERE id = ?", (task_id,)
        )
        self.conn.commit()

    def list_tasks(self, pending_only: bool = True, limit: int = 50) -> list[dict]:
        """List tasks, optionally filtering to pending only."""
        query = "SELECT t.*, s.title as session_title FROM tasks t LEFT JOIN sessions s ON t.session_id = s.id"
        if pending_only:
            query += " WHERE t.completed = 0"
        query += " ORDER BY t.id DESC LIMIT ?"
        rows = self.conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Hashtags
    # ------------------------------------------------------------------

    def add_hashtag(self, hashtag: Hashtag, session_id: int) -> int:
        """Add a hashtag linked to a session."""
        cursor = self.conn.execute(
            "INSERT INTO hashtags (session_id, tag, context) VALUES (?, ?, ?)",
            (session_id, hashtag.tag, hashtag.context),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_hashtags(self, limit: int = 50) -> list[dict]:
        """List recent hashtags with their session context."""
        rows = self.conn.execute(
            """SELECT h.*, s.title as session_title
            FROM hashtags h LEFT JOIN sessions s ON h.session_id = s.id
            ORDER BY h.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_by_hashtag(self, tag: str) -> list[dict]:
        """Find all sessions with a specific hashtag."""
        rows = self.conn.execute(
            """SELECT DISTINCT s.*
            FROM sessions s
            JOIN hashtags h ON s.id = h.session_id
            WHERE h.tag = ?
            ORDER BY s.id DESC""",
            (tag,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_transcripts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across all transcripts."""
        rows = self.conn.execute(
            """SELECT s.*
            FROM transcript_fts fts
            JOIN sessions s ON CAST(fts.session_id AS INTEGER) = s.id
            WHERE fts.content MATCH ?
            GROUP BY s.id
            ORDER BY s.id DESC
            LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Bulk insert from Insight
    # ------------------------------------------------------------------

    def save_insight(self, insight: Insight, session_id: int) -> None:
        """Save a complete Gemini Insight to the database."""
        # Append transcript
        if insight.transcript:
            self.append_transcript(
                session_id, insight.transcript, 0  # duration tracked separately
            )

        # Save tasks
        for task in insight.tasks:
            self.add_task(task, session_id)

        # Save hashtags
        for hashtag in insight.hashtags:
            self.add_hashtag(hashtag, session_id)

        logger.info(
            "Saved insight for session %d: %d tasks, %d hashtags",
            session_id,
            len(insight.tasks),
            len(insight.hashtags),
        )
