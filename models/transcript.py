from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single segment of transcribed text."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")


class MeetingTranscript(BaseModel):
    """Full transcript record for a meeting."""

    meeting_id: str = Field(description="Google Meet meeting code or event ID")
    title: str = Field(default="", description="Meeting title from calendar event")
    organizer: str = Field(default="", description="Email of the meeting organizer")
    attendees: list[str] = Field(default_factory=list, description="List of attendee emails")
    meeting_start: Optional[datetime.datetime] = Field(default=None, description="Scheduled meeting start time")
    meeting_end: Optional[datetime.datetime] = Field(default=None, description="Scheduled meeting end time")
    audio_gcs_uri: str = Field(default="", description="GCS URI of the stored audio file")
    transcript_text: str = Field(default="", description="Full transcript as plain text")
    segments: list[TranscriptSegment] = Field(default_factory=list, description="Timestamped transcript segments")
    language: str = Field(default="en", description="Detected language of the transcript")
    duration_seconds: Optional[float] = Field(default=None, description="Duration of the recording")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    status: str = Field(default="pending", description="processing status: pending, processing, completed, failed")
    error_message: str = Field(default="", description="Error details if processing failed")

    def to_firestore_dict(self) -> dict:
        """Convert to a dict suitable for Firestore storage."""
        data = self.model_dump()
        # Convert datetime objects to ISO strings for Firestore
        for key in ("meeting_start", "meeting_end", "created_at"):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data

    @classmethod
    def from_firestore_dict(cls, data: dict) -> MeetingTranscript:
        """Create a MeetingTranscript from a Firestore document dict."""
        for key in ("meeting_start", "meeting_end", "created_at"):
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.datetime.fromisoformat(data[key])
        return cls(**data)
