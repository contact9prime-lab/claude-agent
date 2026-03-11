"""DeskVoice Terminal UI — a rich terminal interface using Textual.

Provides real-time transcript display, task management, tags, and stats
directly in the terminal. Inspired by Claude Code's TUI design.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from src.config import AgentConfig
from src.storage.database import Database
from src.tui.screens.settings import SettingsScreen

logger = logging.getLogger(__name__)


class StatsBar(Static):
    """Top stats bar showing token usage and speech metrics."""

    chunks = reactive(0)
    speech_secs = reactive(0.0)
    tokens_in = reactive(0)
    tokens_out = reactive(0)
    tokens_total = reactive(0)

    def render(self) -> str:
        return (
            f"Chunks: {self.chunks}  |  "
            f"Speech: {self.speech_secs:.0f}s  |  "
            f"Tokens: {self.tokens_in:,} in / {self.tokens_out:,} out / {self.tokens_total:,} total"
        )


class TranscriptPanel(RichLog):
    """Scrollable panel showing live transcript entries."""

    def add_entry(self, text: str, summary: str = "", timestamp: str = "") -> None:
        ts = timestamp or datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{ts}[/dim]")
        self.write(text)
        if summary:
            self.write(f"[italic blue]{summary}[/italic blue]")
        self.write("")


class TaskPanel(ListView):
    """Panel showing extracted tasks."""

    def add_task(self, task: dict) -> None:
        priority = task.get("priority", "medium")
        color = {"high": "red", "medium": "yellow", "low": "dim"}.get(priority, "white")
        desc = task.get("description", "")
        assignee = task.get("assignee", "")
        due = task.get("due_hint", "")
        task_id = task.get("id", "?")

        parts = [f"[{color}][{priority.upper()}][/{color}] {desc}"]
        if assignee:
            parts.append(f" [yellow]-> {assignee}[/yellow]")
        if due:
            parts.append(f" [dim](due: {due})[/dim]")

        label = "".join(parts)
        self.append(ListItem(Label(f"#{task_id} {label}")))


class TagsPanel(Static):
    """Panel showing extracted hashtags."""

    tags: reactive[set] = reactive(set, init=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tags = set()

    def add_tag(self, tag: str) -> None:
        self.tags = self.tags | {tag}

    def watch_tags(self, tags: set) -> None:
        if tags:
            self.update(" ".join(f"[blue]#{t}[/blue]" for t in sorted(tags)))
        else:
            self.update("[dim]No tags yet[/dim]")


class DecisionsPanel(RichLog):
    """Panel showing decisions and questions."""

    def add_decision(self, text: str) -> None:
        self.write(f"[green]> {text}[/green]")

    def add_question(self, text: str) -> None:
        self.write(f"[yellow]? {text}[/yellow]")


class DeskVoiceTUI(App):
    """The main DeskVoice terminal UI application."""

    TITLE = "DeskVoice"
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-columns: 2fr 1fr;
        grid-rows: auto 1fr auto;
    }
    #stats-bar {
        column-span: 2;
        background: $surface;
        padding: 1;
        text-style: bold;
        color: $success;
    }
    #transcript-panel {
        border: solid $primary;
        height: 100%;
    }
    #sidebar {
        height: 100%;
    }
    #task-panel {
        border: solid $warning;
        height: 1fr;
    }
    #tags-panel {
        border: solid $primary;
        padding: 1;
        height: auto;
        min-height: 3;
    }
    #decisions-panel {
        border: solid $success;
        height: 1fr;
    }
    #bottom-bar {
        column-span: 2;
    }
    """

    SCREENS = {"settings": SettingsScreen}

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "push_screen('settings')", "Settings"),
        Binding("t", "focus_tasks", "Tasks"),
        Binding("r", "refresh_data", "Refresh"),
    ]

    def __init__(
        self,
        config: AgentConfig,
        agent=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.agent = agent
        self._db = Database(config.storage.db_path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield StatsBar(id="stats-bar")
        yield TranscriptPanel(id="transcript-panel", highlight=True, markup=True)
        with Vertical(id="sidebar"):
            yield TaskPanel(id="task-panel")
            yield TagsPanel(id="tags-panel")
            yield DecisionsPanel(id="decisions-panel", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._db.connect()
        self._load_existing_data()

        if self.agent:
            self._start_agent_thread()

        # Poll for updates every second
        self.set_interval(1.0, self._poll_updates)

    def _load_existing_data(self) -> None:
        """Load existing tasks and tags from the database."""
        task_panel = self.query_one("#task-panel", TaskPanel)
        for task in self._db.list_tasks(pending_only=True):
            task_panel.add_task(task)

        tags_panel = self.query_one("#tags-panel", TagsPanel)
        for h in self._db.list_hashtags(limit=50):
            tags_panel.add_tag(h["tag"])

    def _start_agent_thread(self) -> None:
        """Run the agent in a background thread."""
        def run():
            try:
                self.agent.start()
            except Exception as e:
                logger.error("Agent error: %s", e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _poll_updates(self) -> None:
        """Poll agent stats and update the UI."""
        if self.agent:
            stats = self.agent.stats
            bar = self.query_one("#stats-bar", StatsBar)
            bar.chunks = stats.get("chunks_processed", 0)
            bar.speech_secs = stats.get("speech_seconds", 0.0)
            bar.tokens_in = stats.get("total_input_tokens", 0)
            bar.tokens_out = stats.get("total_output_tokens", 0)
            bar.tokens_total = stats.get("total_tokens", 0)

    def push_insight(self, insight, duration: float = 0.0) -> None:
        """Called by the agent to push a new insight to the TUI."""
        self.call_from_thread(self._handle_insight, insight, duration)

    def _handle_insight(self, insight, duration: float) -> None:
        """Process an insight on the main thread."""
        transcript_panel = self.query_one("#transcript-panel", TranscriptPanel)
        if insight.transcript:
            transcript_panel.add_entry(
                insight.transcript,
                summary=insight.summary,
            )

        task_panel = self.query_one("#task-panel", TaskPanel)
        for task in insight.tasks:
            task_panel.add_task({
                "id": task.id,
                "description": task.description,
                "assignee": task.assignee,
                "priority": task.priority.value,
                "due_hint": task.due_hint,
            })

        tags_panel = self.query_one("#tags-panel", TagsPanel)
        for h in insight.hashtags:
            tags_panel.add_tag(h.tag)

        decisions_panel = self.query_one("#decisions-panel", DecisionsPanel)
        for d in insight.decisions:
            decisions_panel.add_decision(d)
        for q in insight.questions:
            decisions_panel.add_question(q)

    def action_focus_tasks(self) -> None:
        self.query_one("#task-panel", TaskPanel).focus()

    def action_refresh_data(self) -> None:
        self._load_existing_data()
