"""Multi-provider LLM abstraction for audio transcription and insight extraction.

Supports: Gemini, OpenAI, Ollama, and any OpenAI-compatible API.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

from src.config import LLMConfig
from src.models.models import (
    AudioChunk,
    AudioType,
    Hashtag,
    Insight,
    Priority,
    Task,
    TokenUsage,
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


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def transcribe_and_extract(self, chunk: AudioChunk) -> Insight:
        """Send audio to the LLM and get back structured insights."""

    @staticmethod
    def _try_recover_json(text: str) -> Optional[dict]:
        """Try to recover truncated JSON by closing open structures."""
        # Count open braces/brackets
        opens = 0
        open_brackets = 0
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                opens += 1
            elif ch == "}":
                opens -= 1
            elif ch == "[":
                open_brackets += 1
            elif ch == "]":
                open_brackets -= 1

        if opens <= 0 and open_brackets <= 0:
            return None  # Not a truncation issue

        # Truncate to last complete value, then close structures
        # Find last complete key-value (ends with , or after a value)
        # Simple approach: strip trailing incomplete string/value, close brackets
        truncated = text.rstrip()
        # Remove trailing incomplete string
        if in_string:
            last_quote = truncated.rfind('"')
            if last_quote > 0:
                truncated = truncated[:last_quote + 1]
                in_string = False

        # Remove trailing comma or colon
        truncated = truncated.rstrip(",: \n\t")

        # Close open structures
        truncated += "]" * max(0, open_brackets) + "}" * max(0, opens)

        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            return None

    def _parse_response(self, text: str) -> Insight:
        """Parse JSON response into an Insight object."""
        cleaned = text.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.match(r"^```(?:json)?\s*\n(.*?)(?:\n```\s*)?$", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to recover truncated JSON by closing open braces/brackets
            data = self._try_recover_json(cleaned)
            if data is None:
                logger.error("Failed to parse LLM response as JSON: %s", text[:200])
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


class GeminiProvider(LLMProvider):
    """Google Gemini provider — native audio support."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client

    def transcribe_and_extract(self, chunk: AudioChunk) -> Insight:
        from google.genai import types

        client = self._get_client()
        logger.info("Sending %.1fs audio to Gemini %s", chunk.duration_seconds, self.config.model)

        response = client.models.generate_content(
            model=self.config.model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=chunk.audio_data, mime_type="audio/wav"),
                        types.Part.from_text(text=TRANSCRIBE_AND_EXTRACT_PROMPT),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )

        insight = self._parse_response(response.text)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            insight.token_usage = TokenUsage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage, "total_token_count", 0) or 0,
            )

        return insight


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (works with OpenAI, Azure, any compatible API)."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def transcribe_and_extract(self, chunk: AudioChunk) -> Insight:
        client = self._get_client()
        logger.info("Sending %.1fs audio to OpenAI %s", chunk.duration_seconds, self.config.model)

        audio_b64 = base64.b64encode(chunk.audio_data).decode("utf-8")

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": TRANSCRIBE_AND_EXTRACT_PROMPT},
                    ],
                }
            ],
            temperature=self.config.temperature,
            max_tokens=4096,
        )

        insight = self._parse_response(response.choices[0].message.content)

        if response.usage:
            insight.token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return insight


class OllamaProvider(LLMProvider):
    """Ollama provider — local models via OpenAI-compatible API."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            base_url = self.config.base_url or "http://localhost:11434/v1"
            self._client = openai.OpenAI(base_url=base_url, api_key="ollama")
        return self._client

    def transcribe_and_extract(self, chunk: AudioChunk) -> Insight:
        client = self._get_client()
        logger.info("Sending %.1fs audio to Ollama %s", chunk.duration_seconds, self.config.model)

        audio_b64 = base64.b64encode(chunk.audio_data).decode("utf-8")

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                        },
                        {"type": "text", "text": TRANSCRIBE_AND_EXTRACT_PROMPT},
                    ],
                }
            ],
            temperature=self.config.temperature,
        )

        insight = self._parse_response(response.choices[0].message.content)

        if response.usage:
            insight.token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return insight


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory to create the right provider based on config."""
    providers = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
    }
    cls = providers.get(config.provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {config.provider}. "
            f"Supported: {', '.join(providers.keys())}"
        )
    return cls(config)
