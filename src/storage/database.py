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
    reminder_at TEXT DEFAULT '',
    edited_at TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS hashtags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    tag TEXT NOT NULL,
    context TEXT DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    voice_embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    filename TEXT NOT NULL,
    duration_seconds REAL DEFAULT 0,
    transcript TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS voice_prints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL DEFAULT '',
    voice_embedding BLOB NOT NULL,
    sample_count INTEGER DEFAULT 1,
    audio_sample_file TEXT DEFAULT '',
    mapped_speaker_id INTEGER DEFAULT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (mapped_speaker_id) REFERENCES speakers(id)
);

CREATE TABLE IF NOT EXISTS recording_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL,
    voice_print_id INTEGER,
    speaker_label TEXT DEFAULT '',
    text TEXT DEFAULT '',
    start_seconds REAL DEFAULT 0,
    end_seconds REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id),
    FOREIGN KEY (voice_print_id) REFERENCES voice_prints(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_completed ON tasks(completed);
CREATE INDEX IF NOT EXISTS idx_hashtags_tag ON hashtags(tag);
CREATE INDEX IF NOT EXISTS idx_hashtags_session ON hashtags(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_recordings_session ON recordings(session_id);
CREATE INDEX IF NOT EXISTS idx_voice_prints_mapped ON voice_prints(mapped_speaker_id);
CREATE INDEX IF NOT EXISTS idx_recording_segments_recording ON recording_segments(recording_id);
CREATE INDEX IF NOT EXISTS idx_recording_segments_voice_print ON recording_segments(voice_print_id);

-- Full-text search on transcripts
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    session_id,
    content,
    tokenize='porter'
);
"""

MIGRATIONS = [
    # Add reminder_at and edited_at to tasks if missing
    ("ALTER TABLE tasks ADD COLUMN reminder_at TEXT DEFAULT ''", "tasks", "reminder_at"),
    ("ALTER TABLE tasks ADD COLUMN edited_at TEXT DEFAULT ''", "tasks", "edited_at"),
    # Add audio_sample_file to voice_prints for existing DBs
    ("ALTER TABLE voice_prints ADD COLUMN audio_sample_file TEXT DEFAULT ''", "voice_prints", "audio_sample_file"),
]


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
        self._run_migrations()
        logger.info("Database connected: %s", self.db_path)

    def _run_migrations(self) -> None:
        """Run schema migrations for existing databases."""
        for sql, table, column in MIGRATIONS:
            try:
                # Check if column exists
                cursor = self._conn.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                if column not in columns:
                    self._conn.execute(sql)
                    self._conn.commit()
                    logger.info("Migration: added %s.%s", table, column)
            except Exception as e:
                logger.debug("Migration skipped: %s", e)

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

    def update_task(
        self,
        task_id: int,
        description: Optional[str] = None,
        assignee: Optional[str] = None,
        priority: Optional[str] = None,
        due_hint: Optional[str] = None,
        reminder_at: Optional[str] = None,
    ) -> None:
        """Update task fields."""
        updates = []
        params = []
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if assignee is not None:
            updates.append("assignee = ?")
            params.append(assignee)
        if priority is not None:
            updates.append("priority = ?")
            params.append(priority)
        if due_hint is not None:
            updates.append("due_hint = ?")
            params.append(due_hint)
        if reminder_at is not None:
            updates.append("reminder_at = ?")
            params.append(reminder_at)
        if not updates:
            return
        updates.append("edited_at = ?")
        params.append(datetime.now().isoformat())
        params.append(task_id)
        self.conn.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()

    def get_task(self, task_id: int) -> Optional[dict]:
        """Get a single task by ID."""
        row = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def get_due_reminders(self) -> list[dict]:
        """Get tasks with reminders that are due now."""
        now = datetime.now().isoformat()
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE reminder_at != '' AND reminder_at <= ? AND completed = 0",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_reminder(self, task_id: int) -> None:
        """Clear a task's reminder after it fires."""
        self.conn.execute("UPDATE tasks SET reminder_at = '' WHERE id = ?", (task_id,))
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
    # Speakers
    # ------------------------------------------------------------------

    def add_speaker(self, name: str, embedding: bytes) -> int:
        """Add a speaker with their voice embedding."""
        cursor = self.conn.execute(
            "INSERT INTO speakers (name, voice_embedding, created_at) VALUES (?, ?, ?)",
            (name, embedding, datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_speakers(self) -> list[dict]:
        """List all enrolled speakers."""
        rows = self.conn.execute(
            "SELECT id, name, created_at FROM speakers ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_speaker_embeddings(self) -> list[tuple[int, str, bytes]]:
        """Get all speaker embeddings for matching."""
        rows = self.conn.execute(
            "SELECT id, name, voice_embedding FROM speakers"
        ).fetchall()
        return [(r["id"], r["name"], r["voice_embedding"]) for r in rows]

    def update_speaker_name(self, speaker_id: int, name: str) -> None:
        """Rename a speaker."""
        self.conn.execute("UPDATE speakers SET name = ? WHERE id = ?", (name, speaker_id))
        self.conn.commit()

    def delete_speaker(self, speaker_id: int) -> None:
        """Delete a speaker profile."""
        self.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Recordings
    # ------------------------------------------------------------------

    def add_recording(
        self,
        session_id: int,
        filename: str,
        duration_seconds: float,
        transcript: str = "",
        summary: str = "",
    ) -> int:
        """Add a recording entry for audio playback."""
        cursor = self.conn.execute(
            """INSERT INTO recordings (session_id, filename, duration_seconds, transcript, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, filename, duration_seconds, transcript, summary, datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def list_recordings(self, limit: int = 50) -> list[dict]:
        """List recent recordings."""
        rows = self.conn.execute(
            """SELECT r.*, s.title as session_title
            FROM recordings r LEFT JOIN sessions s ON r.session_id = s.id
            ORDER BY r.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recording(self, recording_id: int) -> Optional[dict]:
        """Get a single recording by ID."""
        row = self.conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Voice Prints
    # ------------------------------------------------------------------

    def add_voice_print(self, label: str, embedding: bytes, audio_sample_file: str = "") -> int:
        """Add a new auto-detected voice print."""
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO voice_prints (label, voice_embedding, sample_count, audio_sample_file, created_at, updated_at) VALUES (?, ?, 1, ?, ?, ?)",
            (label, embedding, audio_sample_file, now, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_voice_print_embedding(self, vp_id: int, embedding: bytes, sample_count: int) -> None:
        """Update a voice print's averaged embedding."""
        self.conn.execute(
            "UPDATE voice_prints SET voice_embedding = ?, sample_count = ?, updated_at = ? WHERE id = ?",
            (embedding, sample_count, datetime.now().isoformat(), vp_id),
        )
        self.conn.commit()

    def update_voice_print_audio(self, vp_id: int, audio_sample_file: str) -> None:
        """Update the audio sample file for a voice print."""
        self.conn.execute(
            "UPDATE voice_prints SET audio_sample_file = ?, updated_at = ? WHERE id = ?",
            (audio_sample_file, datetime.now().isoformat(), vp_id),
        )
        self.conn.commit()

    def list_voice_prints(self) -> list[dict]:
        """List all voice prints with their mapped speaker names."""
        rows = self.conn.execute(
            """SELECT vp.id, vp.label, vp.sample_count, vp.mapped_speaker_id,
                      vp.audio_sample_file, vp.created_at, vp.updated_at,
                      s.name as mapped_speaker_name
               FROM voice_prints vp
               LEFT JOIN speakers s ON vp.mapped_speaker_id = s.id
               ORDER BY vp.id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_voice_print_embeddings(self) -> list[tuple[int, str, bytes, int]]:
        """Get all voice print embeddings for matching. Returns (id, label, embedding, sample_count)."""
        rows = self.conn.execute(
            """SELECT vp.id, COALESCE(s.name, vp.label) as label, vp.voice_embedding, vp.sample_count
               FROM voice_prints vp
               LEFT JOIN speakers s ON vp.mapped_speaker_id = s.id"""
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def map_voice_print_to_speaker(self, vp_id: int, speaker_id: int) -> None:
        """Map a voice print to a known speaker."""
        self.conn.execute(
            "UPDATE voice_prints SET mapped_speaker_id = ?, updated_at = ? WHERE id = ?",
            (speaker_id, datetime.now().isoformat(), vp_id),
        )
        self.conn.commit()

    def rename_voice_print(self, vp_id: int, label: str) -> None:
        """Rename a voice print label."""
        self.conn.execute(
            "UPDATE voice_prints SET label = ?, updated_at = ? WHERE id = ?",
            (label, datetime.now().isoformat(), vp_id),
        )
        self.conn.commit()

    def merge_voice_prints(self, keep_id: int, merge_id: int) -> None:
        """Merge two voice prints — reassign segments and delete the merged one."""
        self.conn.execute(
            "UPDATE recording_segments SET voice_print_id = ? WHERE voice_print_id = ?",
            (keep_id, merge_id),
        )
        self.conn.execute("DELETE FROM voice_prints WHERE id = ?", (merge_id,))
        self.conn.commit()

    def delete_voice_print(self, vp_id: int) -> None:
        """Delete a voice print."""
        self.conn.execute(
            "UPDATE recording_segments SET voice_print_id = NULL WHERE voice_print_id = ?",
            (vp_id,),
        )
        self.conn.execute("DELETE FROM voice_prints WHERE id = ?", (vp_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Recording Segments
    # ------------------------------------------------------------------

    def add_recording_segment(
        self,
        recording_id: int,
        voice_print_id: Optional[int],
        speaker_label: str,
        text: str,
        start_seconds: float,
        end_seconds: float,
        confidence: float = 0.0,
    ) -> int:
        """Add a segment within a recording."""
        cursor = self.conn.execute(
            """INSERT INTO recording_segments
               (recording_id, voice_print_id, speaker_label, text, start_seconds, end_seconds, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (recording_id, voice_print_id, speaker_label, text,
             start_seconds, end_seconds, confidence, datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_recording_segments(self, recording_id: int) -> list[dict]:
        """Get all segments for a recording with voice print info."""
        rows = self.conn.execute(
            """SELECT rs.*, COALESCE(s.name, vp.label, rs.speaker_label) as resolved_speaker
               FROM recording_segments rs
               LEFT JOIN voice_prints vp ON rs.voice_print_id = vp.id
               LEFT JOIN speakers s ON vp.mapped_speaker_id = s.id
               WHERE rs.recording_id = ?
               ORDER BY rs.start_seconds""",
            (recording_id,),
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
            "Saved insight for session %s: %d tasks, %d hashtags",
            session_id,
            len(insight.tasks),
            len(insight.hashtags),
        )
