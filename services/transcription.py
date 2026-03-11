"""Transcription service using OpenAI Whisper API.

Handles audio transcription with support for:
- Multiple audio formats
- Large file chunking (Whisper API has a 25MB limit)
- Verbose output with timestamps for segment-level data
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
from pathlib import Path

from openai import OpenAI

from config import Config
from models.transcript import TranscriptSegment

logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    """Create an OpenAI client."""
    return OpenAI(api_key=Config.OPENAI_API_KEY)


def transcribe_audio(
    audio_data: bytes,
    filename: str = "recording.webm",
    language: str | None = None,
) -> dict:
    """Transcribe audio data using OpenAI Whisper API.

    Args:
        audio_data: Raw audio file bytes.
        filename: Original filename (used to determine format).
        language: Optional ISO-639-1 language code to guide transcription.

    Returns:
        Dict with keys:
        - text: Full transcript text
        - segments: List of TranscriptSegment dicts
        - language: Detected language
        - duration: Total duration in seconds
    """
    client = _get_client()

    file_size_mb = len(audio_data) / (1024 * 1024)
    logger.info("Transcribing audio: %s (%.1f MB)", filename, file_size_mb)

    if file_size_mb > Config.MAX_AUDIO_SIZE_MB:
        logger.info("File exceeds %d MB, chunking required", Config.MAX_AUDIO_SIZE_MB)
        return _transcribe_large_file(client, audio_data, filename, language)

    return _transcribe_single(client, audio_data, filename, language)


def _transcribe_single(
    client: OpenAI,
    audio_data: bytes,
    filename: str,
    language: str | None,
) -> dict:
    """Transcribe a single audio file within the size limit."""
    audio_file = io.BytesIO(audio_data)
    audio_file.name = filename

    kwargs = {
        "model": Config.WHISPER_MODEL,
        "file": audio_file,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if language:
        kwargs["language"] = language

    response = client.audio.transcriptions.create(**kwargs)

    segments = []
    for seg in getattr(response, "segments", []) or []:
        segments.append(TranscriptSegment(
            start=seg.get("start", 0.0) if isinstance(seg, dict) else seg.start,
            end=seg.get("end", 0.0) if isinstance(seg, dict) else seg.end,
            text=(seg.get("text", "") if isinstance(seg, dict) else seg.text).strip(),
        ))

    duration = getattr(response, "duration", None) or 0.0
    detected_language = getattr(response, "language", "en") or "en"

    logger.info(
        "Transcription complete: %d segments, %.1f seconds, language=%s",
        len(segments),
        duration,
        detected_language,
    )

    return {
        "text": response.text,
        "segments": [s.model_dump() for s in segments],
        "language": detected_language,
        "duration": duration,
    }


def _transcribe_large_file(
    client: OpenAI,
    audio_data: bytes,
    filename: str,
    language: str | None,
) -> dict:
    """Handle files larger than the Whisper API limit by splitting into chunks.

    Writes the audio to a temp file and splits it into chunks under the size limit.
    Each chunk is transcribed separately and results are merged.
    """
    chunk_size = Config.MAX_AUDIO_SIZE_MB * 1024 * 1024
    num_chunks = math.ceil(len(audio_data) / chunk_size)
    logger.info("Splitting into %d chunks", num_chunks)

    all_text_parts = []
    all_segments = []
    total_duration = 0.0
    detected_language = "en"

    suffix = Path(filename).suffix or ".webm"

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(audio_data))
        chunk = audio_data[start:end]

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(chunk)
            tmp.flush()
            tmp.seek(0)

            result = _transcribe_single(client, tmp.read(), f"chunk_{i}{suffix}", language)

        # Offset segment timestamps for chunks after the first
        time_offset = total_duration
        for seg in result["segments"]:
            seg["start"] += time_offset
            seg["end"] += time_offset
            all_segments.append(seg)

        all_text_parts.append(result["text"])
        total_duration += result["duration"]
        detected_language = result["language"]

    return {
        "text": " ".join(all_text_parts),
        "segments": all_segments,
        "language": detected_language,
        "duration": total_duration,
    }
