"""Data models for the desk voice agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class AudioType(Enum):
    """Classification of audio content."""

    SPEECH = "speech"
    MUSIC = "music"
    NOISE = "noise"
    SILENCE = "silence"
    UNKNOWN = "unknown"


class Priority(Enum):
    """Priority level for extracted items."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AudioChunk:
    """A chunk of audio data ready for processing."""

    audio_data: bytes
    sample_rate: int
    duration_seconds: float
    timestamp_start: datetime
    timestamp_end: datetime
    audio_type: AudioType = AudioType.UNKNOWN
    source: str = "system_audio"  # system_audio, microphone, etc.


@dataclass
class Task:
    """An extracted task/action item."""

    id: Optional[int] = None
    description: str = ""
    assignee: str = ""  # Who should do it (if mentioned)
    priority: Priority = Priority.MEDIUM
    due_hint: str = ""  # e.g., "by Friday", "ASAP", "next week"
    completed: bool = False
    session_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Hashtag:
    """An extracted topic/hashtag."""

    tag: str = ""
    context: str = ""  # Surrounding text that generated this tag
    session_id: Optional[int] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with timing."""

    text: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    speaker: str = ""  # Speaker label if identified
    confidence: float = 0.0


@dataclass
class TokenUsage:
    """Token usage from a Gemini API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Insight:
    """Structured insight extracted by Gemini from a chunk."""

    transcript: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    summary: str = ""
    tasks: list[Task] = field(default_factory=list)
    hashtags: list[Hashtag] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)  # Open questions raised
    language: str = "en"
    audio_type: AudioType = AudioType.SPEECH
    token_usage: Optional[TokenUsage] = None


@dataclass
class Session:
    """A continuous conversation/call session.

    A session starts when speech is first detected and ends after
    a long silence gap (e.g., 60s of no speech).
    """

    id: Optional[int] = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    title: str = ""  # Auto-generated from content
    total_speech_seconds: float = 0.0
    transcript: str = ""
    summary: str = ""
    tasks: list[Task] = field(default_factory=list)
    hashtags: list[Hashtag] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    status: str = "active"  # active, ended, summarized
