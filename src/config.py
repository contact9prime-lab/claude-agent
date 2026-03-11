"""Configuration for the meeting transcript agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AudioConfig:
    """Audio capture settings."""

    sample_rate: int = 16000  # 16kHz — optimal for speech
    channels: int = 1  # Mono
    chunk_duration_sec: float = 0.5  # VAD processes 500ms frames
    dtype: str = "int16"

    # VAD settings
    vad_threshold: float = 0.5  # Silero VAD confidence threshold
    speech_pad_ms: int = 300  # Padding around speech segments
    min_speech_duration_ms: int = 500  # Ignore speech shorter than this
    min_silence_duration_ms: int = 1500  # Split on silence longer than this

    # Buffering — accumulate speech before sending to Gemini
    max_chunk_seconds: int = 120  # Send to Gemini every 2 min of speech max
    min_chunk_seconds: int = 10  # Don't send less than 10s of speech


@dataclass
class GeminiConfig:
    """Gemini API settings."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "gemini-2.5-flash"  # Cheapest, fastest
    max_audio_bytes: int = 20 * 1024 * 1024  # 20MB per request
    temperature: float = 0.1  # Low creativity for transcription


@dataclass
class StorageConfig:
    """Local storage settings."""

    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DB_PATH", str(Path.home() / ".deskvoice" / "deskvoice.db"))
        )
    )
    audio_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("AUDIO_DIR", str(Path.home() / ".deskvoice" / "audio"))
        )
    )

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Agent behavior
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    # Only classify audio types (speech/music/noise) — don't transcribe
    # unless speech is detected. Saves tokens.
    classify_before_transcribe: bool = True

    def validate(self) -> list[str]:
        errors = []
        if not self.gemini.api_key:
            errors.append("GEMINI_API_KEY is required")
        return errors
