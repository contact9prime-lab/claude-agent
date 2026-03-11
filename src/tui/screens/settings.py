"""Settings screen for DeskVoice TUI.

Allows editing LLM provider, model, API key, audio device,
VAD thresholds, and session timeout. Persists to ~/.deskvoice/config.toml.
"""

from __future__ import annotations

import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

logger = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".deskvoice" / "config.toml"


def load_config_toml() -> dict:
    """Load config from TOML file."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(CONFIG_PATH.read_text())
    except ImportError:
        try:
            import toml
            return toml.load(CONFIG_PATH)
        except ImportError:
            return {}


def save_config_toml(data: dict) -> None:
    """Save config to TOML file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import toml
        CONFIG_PATH.write_text(toml.dumps(data))
        logger.info("Config saved to %s", CONFIG_PATH)
    except ImportError:
        # Fallback: write simple key=value format
        lines = []
        for section, values in data.items():
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        CONFIG_PATH.write_text("\n".join(lines))


class SettingsScreen(Screen):
    """Settings screen for editing agent configuration."""

    BINDINGS = [("escape", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="settings-form"):
            yield Label("LLM Settings", classes="section-header")
            yield Label("Provider:")
            yield Select(
                [("gemini", "gemini"), ("openai", "openai"), ("ollama", "ollama")],
                id="provider",
                value="gemini",
            )
            yield Label("Model:")
            yield Input(id="model", placeholder="gemini-2.5-flash")
            yield Label("API Key:")
            yield Input(id="api_key", placeholder="sk-...", password=True)
            yield Label("Base URL (for Ollama):")
            yield Input(id="base_url", placeholder="http://localhost:11434/v1")

            yield Label("Audio Settings", classes="section-header")
            yield Label("VAD Threshold (0.0-1.0):")
            yield Input(id="vad_threshold", placeholder="0.5")
            yield Label("Min Silence Duration (ms):")
            yield Input(id="min_silence_ms", placeholder="7000")
            yield Label("Session Timeout (seconds):")
            yield Input(id="session_timeout", placeholder="60")

            yield Button("Save", id="save-btn", variant="primary")
            yield Static("", id="save-status")
        yield Footer()

    def on_mount(self) -> None:
        """Load current config values."""
        config = load_config_toml()
        llm = config.get("llm", {})
        audio = config.get("audio", {})

        if llm.get("provider"):
            self.query_one("#provider", Select).value = llm["provider"]
        if llm.get("model"):
            self.query_one("#model", Input).value = llm["model"]
        if llm.get("api_key"):
            self.query_one("#api_key", Input).value = llm["api_key"]
        if llm.get("base_url"):
            self.query_one("#base_url", Input).value = llm["base_url"]
        if audio.get("vad_threshold"):
            self.query_one("#vad_threshold", Input).value = str(audio["vad_threshold"])
        if audio.get("min_silence_duration_ms"):
            self.query_one("#min_silence_ms", Input).value = str(audio["min_silence_duration_ms"])
        if audio.get("session_timeout"):
            self.query_one("#session_timeout", Input).value = str(audio["session_timeout"])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._save_settings()

    def _save_settings(self) -> None:
        """Save settings to config.toml."""
        config = {
            "llm": {
                "provider": self.query_one("#provider", Select).value,
                "model": self.query_one("#model", Input).value,
                "api_key": self.query_one("#api_key", Input).value,
                "base_url": self.query_one("#base_url", Input).value,
            },
            "audio": {},
        }

        vad = self.query_one("#vad_threshold", Input).value
        if vad:
            config["audio"]["vad_threshold"] = float(vad)
        silence = self.query_one("#min_silence_ms", Input).value
        if silence:
            config["audio"]["min_silence_duration_ms"] = int(silence)
        timeout = self.query_one("#session_timeout", Input).value
        if timeout:
            config["audio"]["session_timeout"] = int(timeout)

        save_config_toml(config)
        self.query_one("#save-status", Static).update("[green]Settings saved![/green]")
