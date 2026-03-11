"""Task management screen for DeskVoice TUI.

Supports inline editing of task fields and setting reminders.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, ListView, ListItem, Select, Static

from src.storage.database import Database

logger = logging.getLogger(__name__)


class TaskDetailScreen(Screen):
    """Screen for viewing and editing a single task."""

    BINDINGS = [("escape", "pop_screen", "Back")]

    def __init__(self, db: Database, task_id: int, **kwargs):
        super().__init__(**kwargs)
        self._db = db
        self._task_id = task_id
        self._task = db.get_task(task_id) or {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="task-form"):
            yield Label(f"Task #{self._task_id}", classes="section-header")

            yield Label("Description:")
            yield Input(
                id="desc",
                value=self._task.get("description", ""),
            )
            yield Label("Assignee:")
            yield Input(id="assignee", value=self._task.get("assignee", ""))

            yield Label("Priority:")
            yield Select(
                [("high", "high"), ("medium", "medium"), ("low", "low")],
                id="priority",
                value=self._task.get("priority", "medium"),
            )

            yield Label("Due hint:")
            yield Input(id="due_hint", value=self._task.get("due_hint", ""))

            yield Label("Reminder (minutes from now, or empty to clear):")
            yield Input(id="reminder_mins", placeholder="e.g. 30")

            with Horizontal():
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Mark Done", id="done-btn", variant="success")
                yield Button("Cancel", id="cancel-btn")

            yield Static("", id="status")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save()
        elif event.button.id == "done-btn":
            self._db.complete_task(self._task_id)
            self.query_one("#status", Static).update("[green]Task completed![/green]")
        elif event.button.id == "cancel-btn":
            self.app.pop_screen()

    def _save(self) -> None:
        reminder_mins = self.query_one("#reminder_mins", Input).value.strip()
        reminder_at = ""
        if reminder_mins:
            try:
                mins = int(reminder_mins)
                reminder_at = (datetime.now() + timedelta(minutes=mins)).isoformat()
            except ValueError:
                pass

        self._db.update_task(
            self._task_id,
            description=self.query_one("#desc", Input).value,
            assignee=self.query_one("#assignee", Input).value,
            priority=self.query_one("#priority", Select).value,
            due_hint=self.query_one("#due_hint", Input).value,
            reminder_at=reminder_at if reminder_mins else None,
        )
        self.query_one("#status", Static).update("[green]Task saved![/green]")
