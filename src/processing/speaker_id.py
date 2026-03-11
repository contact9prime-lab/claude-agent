"""Speaker identification and voice print detection using voice embeddings.

Uses resemblyzer for lightweight speaker embeddings.
Features:
- Extract voice embeddings from audio segments
- Auto-detect unique voices and create voice prints
- Match incoming audio against known voice prints
- Cluster multiple segments by speaker
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


def extract_raw_embedding(audio_data: bytes, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract a raw numpy embedding from WAV audio data.

    Returns the embedding as a numpy array, or None on failure.
    """
    try:
        encoder = _get_encoder()
        from resemblyzer import preprocess_wav

        buf = io.BytesIO(audio_data)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        wav = preprocess_wav(audio, source_sr=sr)
        if len(wav) < 1600:  # Too short for meaningful embedding
            return None
        return encoder.embed_utterance(wav)
    except Exception as e:
        logger.debug("Failed to extract raw embedding: %s", e)
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


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
        score = cosine_similarity(chunk_embedding, known_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold and best_name:
        logger.debug("Speaker identified: %s (score=%.3f)", best_name, best_score)
        return best_name

    logger.debug("No speaker match (best=%.3f, threshold=%.3f)", best_score, threshold)
    return None


def match_voice_print(
    embedding: np.ndarray,
    voice_prints: list[tuple[int, str, bytes, int]],
    threshold: float = 0.70,
) -> Optional[tuple[int, str, float]]:
    """Match an embedding against known voice prints.

    Args:
        embedding: Raw numpy embedding to match
        voice_prints: List of (id, label, pickled_embedding, sample_count) from database
        threshold: Minimum cosine similarity for a match

    Returns:
        (voice_print_id, label, score) if matched, None otherwise.
    """
    if not voice_prints:
        return None

    best_id = None
    best_label = None
    best_score = 0.0

    for vp_id, label, emb_bytes, _ in voice_prints:
        known_emb = pickle.loads(emb_bytes)
        score = cosine_similarity(embedding, known_emb)
        if score > best_score:
            best_score = score
            best_id = vp_id
            best_label = label

    if best_score >= threshold and best_id is not None:
        return (best_id, best_label, best_score)

    return None


def update_averaged_embedding(
    existing_pickled: bytes, new_embedding: np.ndarray, sample_count: int
) -> bytes:
    """Update a voice print embedding with a running average.

    Uses incremental mean: new_avg = old_avg + (new - old_avg) / (n + 1)
    """
    existing = pickle.loads(existing_pickled)
    updated = existing + (new_embedding - existing) / (sample_count + 1)
    # Re-normalize
    updated = updated / (np.linalg.norm(updated) + 1e-8)
    return pickle.dumps(updated)


def segment_audio_by_speaker(
    audio_data: bytes,
    sample_rate: int = 16000,
    segment_duration_sec: float = 2.0,
) -> list[dict]:
    """Segment audio into speaker-attributed chunks.

    Splits audio into fixed-duration windows, extracts embeddings,
    and clusters them to identify different speakers.

    Returns a list of segments: [{start, end, embedding, cluster}]
    """
    try:
        encoder = _get_encoder()
        from resemblyzer import preprocess_wav
    except ImportError:
        logger.warning("resemblyzer not available for speaker segmentation")
        return []

    buf = io.BytesIO(audio_data)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    wav = preprocess_wav(audio, source_sr=sr)
    total_duration = len(wav) / 16000  # resemblyzer uses 16kHz

    if total_duration < segment_duration_sec:
        # Too short to segment, treat as single speaker
        emb = encoder.embed_utterance(wav)
        return [{"start": 0.0, "end": total_duration, "embedding": emb, "cluster": 0}]

    # Extract embeddings for overlapping windows
    segment_samples = int(segment_duration_sec * 16000)
    hop_samples = segment_samples // 2  # 50% overlap
    segments = []

    for start_sample in range(0, len(wav) - segment_samples + 1, hop_samples):
        end_sample = start_sample + segment_samples
        window = wav[start_sample:end_sample]
        emb = encoder.embed_utterance(window)
        segments.append({
            "start": start_sample / 16000,
            "end": end_sample / 16000,
            "embedding": emb,
        })

    if not segments:
        return []

    # Cluster embeddings using agglomerative clustering
    embeddings = np.array([s["embedding"] for s in segments])
    labels = _cluster_embeddings(embeddings)

    for seg, label in zip(segments, labels):
        seg["cluster"] = int(label)

    return segments


def _cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.65) -> list[int]:
    """Cluster speaker embeddings using agglomerative clustering.

    Falls back to simple threshold-based clustering if sklearn is not available.
    """
    if len(embeddings) <= 1:
        return [0] * len(embeddings)

    try:
        from sklearn.cluster import AgglomerativeClustering

        # Use cosine distance for clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)
        return labels.tolist()
    except ImportError:
        # Fallback: simple greedy clustering
        labels = [-1] * len(embeddings)
        cluster_id = 0
        centroids = []

        for i, emb in enumerate(embeddings):
            matched = False
            for cid, centroid in enumerate(centroids):
                if cosine_similarity(emb, centroid) >= threshold:
                    labels[i] = cid
                    # Update centroid with running average
                    n = sum(1 for l in labels[:i] if l == cid)
                    centroids[cid] = centroid + (emb - centroid) / (n + 1)
                    matched = True
                    break

            if not matched:
                labels[i] = cluster_id
                centroids.append(emb.copy())
                cluster_id += 1

        return labels


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
