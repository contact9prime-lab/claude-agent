"""Speaker identification using voice embeddings.

Uses resemblyzer for lightweight speaker embeddings.
Compares incoming audio against enrolled speaker profiles
to replace generic "Speaker 1" labels with real names.
"""

from __future__ import annotations

import io
import logging
import pickle
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded encoder
_encoder = None


def _get_encoder():
    """Lazy-load the resemblyzer voice encoder."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        logger.info("Speaker encoder loaded")
    return _encoder


def extract_embedding(audio_data: bytes, sample_rate: int = 16000) -> bytes:
    """Extract a voice embedding from WAV audio data.

    Returns the embedding as pickled bytes for database storage.
    """
    encoder = _get_encoder()
    from resemblyzer import preprocess_wav

    # Convert WAV bytes to float array
    buf = io.BytesIO(audio_data)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    wav = preprocess_wav(audio, source_sr=sr)
    embedding = encoder.embed_utterance(wav)
    return pickle.dumps(embedding)


def extract_embedding_from_numpy(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Extract embedding from a numpy float32 array."""
    encoder = _get_encoder()
    from resemblyzer import preprocess_wav

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0

    wav = preprocess_wav(audio, source_sr=sample_rate)
    embedding = encoder.embed_utterance(wav)
    return pickle.dumps(embedding)


def identify_speaker(
    audio_data: bytes,
    known_speakers: list[tuple[int, str, bytes]],
    threshold: float = 0.75,
) -> Optional[str]:
    """Identify the speaker in an audio chunk.

    Args:
        audio_data: WAV bytes of the audio chunk
        known_speakers: List of (id, name, pickled_embedding) from database
        threshold: Minimum cosine similarity for a match

    Returns:
        Speaker name if matched, None otherwise.
    """
    if not known_speakers:
        return None

    try:
        chunk_embedding = pickle.loads(extract_embedding(audio_data))
    except Exception as e:
        logger.debug("Failed to extract embedding: %s", e)
        return None

    best_name = None
    best_score = 0.0

    for _, name, emb_bytes in known_speakers:
        known_emb = pickle.loads(emb_bytes)
        score = float(np.dot(chunk_embedding, known_emb) / (
            np.linalg.norm(chunk_embedding) * np.linalg.norm(known_emb) + 1e-8
        ))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold and best_name:
        logger.debug("Speaker identified: %s (score=%.3f)", best_name, best_score)
        return best_name

    logger.debug("No speaker match (best=%.3f, threshold=%.3f)", best_score, threshold)
    return None


def record_enrollment_audio(duration_sec: float = 10.0, sample_rate: int = 16000) -> bytes:
    """Record audio from the default microphone for speaker enrollment.

    Returns WAV bytes.
    """
    import sounddevice as sd

    logger.info("Recording %ds of speech for enrollment...", duration_sec)
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    # Convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return buf.getvalue()
