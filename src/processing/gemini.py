"""Gemini Flash integration for audio transcription and insight extraction.

This module sends audio chunks to Gemini's multimodal API for:
1. Transcription with speaker diarization
2. Task/action item extraction
3. Hashtag/topic identification
4. Decision tracking
5. Open question detection

The key design principle is token efficiency:
- Only speech audio is sent (VAD pre-filtered)
- Audio is sent as WAV (Gemini handles it natively)
- A single prompt does transcription + insight extraction in one call
- Uses Gemini Flash (cheapest model) by default
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from src.config import GeminiConfig
from src.models.models import (
    AudioChunk,
    AudioType,
    Hashtag,
    Insight,
    Priority,
    Task,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

TRANSCRIBE_AND_EXTRACT_PROMPT = """\
You are an always-on desk assistant analyzing a live audio chunk from a conversation or call.

Analyze this audio and return a JSON object with these fields:

{
  "transcript": "Full verbatim transcript of the audio",
  "segments": [
    {"text": "segment text", "start_seconds": 0.0, "end_seconds": 2.5, "speaker": "Speaker 1"}
  ],
  "summary": "1-2 sentence summary of what was discussed",
  "tasks": [
    {"description": "what needs to be done", "assignee": "who (if mentioned)", "priority": "high|medium|low", "due_hint": "by when (if mentioned)"}
  ],
  "hashtags": [
    {"tag": "topic_name", "context": "brief context of why this tag"}
  ],
  "decisions": ["Any decisions that were made"],
  "questions": ["Any open questions that were raised but not answered"],
  "language": "detected language code (e.g., en, es, fr)",
  "audio_type": "speech|music|noise|unknown"
}

Rules:
- Be concise. This runs continuously so efficiency matters.
- Only extract tasks that are clearly actionable items someone committed to or was asked to do.
- Hashtags should be meaningful topics, not every noun. Think: what would you tag this conversation with?
- If the audio is music, noise, or unintelligible, set audio_type accordingly and leave other fields minimal.
- For speaker labels, use "Speaker 1", "Speaker 2" etc. if you can distinguish voices.
- Return ONLY valid JSON, no markdown formatting.
"""

CLASSIFY_PROMPT = """\
Listen to this short audio clip and classify it. Return ONLY a JSON object:
{"audio_type": "speech|music|noise|silence|unknown", "confidence": 0.0}
No other text.
"""


class GeminiProcessor:
    """Processes audio chunks through Gemini Flash."""

    def __init__(self, config: GeminiConfig) -> None:
        self.config = config
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    def classify_audio(self, chunk: AudioChunk) -> AudioType:
        """Quick classification of audio content type using Gemini.

        This is a lightweight call used when the local classifier is uncertain.
        Uses minimal tokens.
        """
        client = self._get_client()

        response = client.models.generate_content(
            model=self.config.model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(
                            data=chunk.audio_data,
                            mime_type="audio/wav",
                        ),
                        types.Part.from_text(text=CLASSIFY_PROMPT),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )

        try:
            result = json.loads(response.text.strip())
            audio_type = AudioType(result.get("audio_type", "unknown"))
            logger.info("Gemini classified audio as: %s", audio_type.value)
            return audio_type
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse Gemini classification: %s", response.text)
            return AudioType.UNKNOWN

    def transcribe_and_extract(self, chunk: AudioChunk) -> Insight:
        """Transcribe audio and extract insights in a single Gemini call.

        This is the main processing function — one API call does everything:
        transcription, task extraction, hashtags, decisions, questions.
        """
        client = self._get_client()

        logger.info(
            "Sending %.1fs audio to Gemini %s",
            chunk.duration_seconds,
            self.config.model,
        )

        response = client.models.generate_content(
            model=self.config.model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(
                            data=chunk.audio_data,
                            mime_type="audio/wav",
                        ),
                        types.Part.from_text(text=TRANSCRIBE_AND_EXTRACT_PROMPT),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=4096,
            ),
        )

        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> Insight:
        """Parse Gemini's JSON response into an Insight object."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini response as JSON: %s", text[:200])
            return Insight(
                transcript=text,
                summary="[Failed to parse structured response]",
            )

        segments = [
            TranscriptSegment(
                text=seg.get("text", ""),
                start_seconds=seg.get("start_seconds", 0.0),
                end_seconds=seg.get("end_seconds", 0.0),
                speaker=seg.get("speaker", ""),
            )
            for seg in data.get("segments", [])
        ]

        tasks = [
            Task(
                description=t.get("description", ""),
                assignee=t.get("assignee", ""),
                priority=Priority(t.get("priority", "medium")),
                due_hint=t.get("due_hint", ""),
            )
            for t in data.get("tasks", [])
        ]

        hashtags = [
            Hashtag(tag=h.get("tag", ""), context=h.get("context", ""))
            for h in data.get("hashtags", [])
        ]

        audio_type = AudioType.SPEECH
        try:
            audio_type = AudioType(data.get("audio_type", "speech"))
        except ValueError:
            pass

        return Insight(
            transcript=data.get("transcript", ""),
            segments=segments,
            summary=data.get("summary", ""),
            tasks=tasks,
            hashtags=hashtags,
            decisions=data.get("decisions", []),
            questions=data.get("questions", []),
            language=data.get("language", "en"),
            audio_type=audio_type,
        )
