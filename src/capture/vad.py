"""Voice Activity Detection using Silero VAD.

Silero VAD is a lightweight (~1MB) model that runs on CPU with very low
latency. It processes audio frames and returns a probability that the
frame contains speech.

This module wraps Silero VAD into a streaming pipeline that:
1. Accepts raw audio frames from AudioStream
2. Detects speech regions with padding
3. Accumulates speech into chunks suitable for Gemini processing
4. Signals when a chunk is ready (enough speech) or a session ends (long silence)
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from src.config import AudioConfig
from src.models.models import AudioChunk, AudioType

logger = logging.getLogger(__name__)


class SileroVAD:
    """Silero VAD wrapper with streaming state management."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._model = None
        self._model_loaded = False

        # State for speech accumulation
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_time: Optional[datetime] = None
        self._speech_seconds: float = 0.0

        # Silence tracking for session boundaries
        self._silence_frames: int = 0
        self._min_silence_frames = int(
            config.min_silence_duration_ms / (config.chunk_duration_sec * 1000)
        )

        # Ring buffer for speech padding (keep last N frames before speech starts)
        pad_frames = max(1, int(config.speech_pad_ms / (config.chunk_duration_sec * 1000)))
        self._pre_speech_buffer: deque[np.ndarray] = deque(maxlen=pad_frames)

        self._in_speech = False

    def load_model(self) -> None:
        """Load the Silero VAD model. Call once at startup."""
        if self._model_loaded:
            return
        logger.info("Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model_loaded = True
        logger.info("Silero VAD model loaded")

    def process_frame(self, audio_frame: np.ndarray) -> Optional[AudioChunk]:
        """Process a single audio frame through VAD.

        Args:
            audio_frame: Raw audio data as numpy array (int16).

        Returns:
            An AudioChunk when enough speech has been accumulated,
            or None if still collecting.
        """
        if not self._model_loaded:
            self.load_model()

        # Convert int16 to float32 for Silero
        audio_float = audio_frame.astype(np.float32) / 32768.0
        if audio_float.ndim > 1:
            audio_float = audio_float[:, 0]  # Mono

        # Run VAD
        tensor = torch.from_numpy(audio_float)
        speech_prob = self._model(tensor, self.config.sample_rate).item()

        is_speech = speech_prob >= self.config.vad_threshold
        now = datetime.now()

        if is_speech:
            self._silence_frames = 0

            if not self._in_speech:
                # Speech just started — include pre-speech padding
                self._in_speech = True
                self._speech_start_time = self._speech_start_time or now
                self._speech_buffer.extend(self._pre_speech_buffer)
                self._pre_speech_buffer.clear()
                logger.debug("Speech started (prob=%.2f)", speech_prob)

            self._speech_buffer.append(audio_frame)
            self._speech_seconds += self.config.chunk_duration_sec

            # Check if we've accumulated enough speech for a chunk
            if self._speech_seconds >= self.config.max_chunk_seconds:
                return self._flush_chunk(now)

        else:
            self._pre_speech_buffer.append(audio_frame)

            if self._in_speech:
                self._silence_frames += 1
                # Add silence frames as padding after speech
                self._speech_buffer.append(audio_frame)

                if self._silence_frames >= self._min_silence_frames:
                    # Long silence — end of speech segment
                    self._in_speech = False
                    logger.debug(
                        "Speech ended after %.1fs silence",
                        self._silence_frames * self.config.chunk_duration_sec,
                    )

                    if self._speech_seconds >= (
                        self.config.min_chunk_seconds
                    ):
                        return self._flush_chunk(now)
                    else:
                        # Too short — discard
                        logger.debug(
                            "Discarding short speech (%.1fs)",
                            self._speech_seconds,
                        )
                        self._reset_buffer()

        return None

    def flush(self) -> Optional[AudioChunk]:
        """Force-flush any remaining speech buffer (e.g., on shutdown)."""
        if self._speech_buffer and self._speech_seconds >= self.config.min_chunk_seconds:
            return self._flush_chunk(datetime.now())
        self._reset_buffer()
        return None

    @property
    def is_in_speech(self) -> bool:
        return self._in_speech

    @property
    def buffered_seconds(self) -> float:
        return self._speech_seconds

    def _flush_chunk(self, end_time: datetime) -> AudioChunk:
        """Package the speech buffer into an AudioChunk."""
        audio_data = np.concatenate(self._speech_buffer)
        wav_bytes = self._to_wav_bytes(audio_data)
        duration = len(audio_data) / self.config.sample_rate

        chunk = AudioChunk(
            audio_data=wav_bytes,
            sample_rate=self.config.sample_rate,
            duration_seconds=duration,
            timestamp_start=self._speech_start_time or end_time,
            timestamp_end=end_time,
            audio_type=AudioType.SPEECH,
        )

        logger.info(
            "Speech chunk ready: %.1fs of audio",
            duration,
        )
        self._reset_buffer()
        return chunk

    def _reset_buffer(self) -> None:
        """Clear the speech accumulation state."""
        self._speech_buffer.clear()
        self._speech_start_time = None
        self._speech_seconds = 0.0
        self._silence_frames = 0

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        if audio.ndim > 1:
            audio = audio[:, 0]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(audio.astype(np.int16).tobytes())
        return buf.getvalue()
