"""Audio content classifier.

Lightweight, local-only classification of audio content type.
Uses simple signal analysis heuristics before sending to Gemini,
so we never waste tokens on music or background noise.

Classification hierarchy:
1. Silence   — RMS below threshold
2. Music     — High spectral regularity, rhythmic patterns
3. Noise     — High energy but low speech features
4. Speech    — Passes all checks → send to Gemini
"""

from __future__ import annotations

import logging

import numpy as np

from src.models.models import AudioType

logger = logging.getLogger(__name__)

# Thresholds (tuned for 16kHz, 16-bit audio)
SILENCE_RMS_THRESHOLD = 200  # int16 scale
SPEECH_ZCR_LOW = 0.01  # Zero-crossing rate lower bound for speech
SPEECH_ZCR_HIGH = 0.25  # Upper bound (music/noise tends higher)
SPECTRAL_FLATNESS_THRESHOLD = 0.3  # Below this = tonal (music), above = noisy/speech


def classify_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> AudioType:
    """Classify audio content type using signal analysis.

    This is a fast, local-only check. Not perfect, but good enough to
    filter out obvious non-speech audio and save Gemini tokens.

    Args:
        audio_data: Raw audio as numpy array (int16 or float32).
        sample_rate: Sample rate in Hz.

    Returns:
        AudioType classification.
    """
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Convert to float for analysis
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_int = audio_data
    else:
        audio_float = audio_data.astype(np.float32)
        audio_int = (audio_data * 32768).astype(np.int16)

    # 1. Check for silence
    rms = np.sqrt(np.mean(audio_int.astype(np.float64) ** 2))
    if rms < SILENCE_RMS_THRESHOLD:
        return AudioType.SILENCE

    # 2. Zero-crossing rate — speech has moderate ZCR
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float))) > 0)
    zcr = zero_crossings / len(audio_float)

    # 3. Spectral flatness — speech has lower flatness than noise
    fft = np.fft.rfft(audio_float)
    magnitude = np.abs(fft) + 1e-10
    spectral_flatness = np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude)

    # 4. Energy variance — speech has high energy variance (pauses between words)
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)  # 10ms hop
    frames = [
        audio_float[i : i + frame_length]
        for i in range(0, len(audio_float) - frame_length, hop_length)
    ]
    if frames:
        frame_energies = [np.sum(f ** 2) for f in frames]
        energy_var = np.var(frame_energies) / (np.mean(frame_energies) + 1e-10)
    else:
        energy_var = 0.0

    # Classification logic
    if spectral_flatness > SPECTRAL_FLATNESS_THRESHOLD and zcr > SPEECH_ZCR_HIGH:
        result = AudioType.NOISE
    elif zcr < SPEECH_ZCR_LOW and energy_var < 0.1:
        result = AudioType.MUSIC
    elif SPEECH_ZCR_LOW <= zcr <= SPEECH_ZCR_HIGH and energy_var > 0.05:
        result = AudioType.SPEECH
    else:
        # When in doubt, let Gemini decide
        result = AudioType.UNKNOWN

    logger.debug(
        "Audio classified as %s (rms=%.0f, zcr=%.3f, flatness=%.3f, energy_var=%.3f)",
        result.value,
        rms,
        zcr,
        spectral_flatness,
        energy_var,
    )
    return result
