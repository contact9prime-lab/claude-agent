"""Audio notification for the desk voice agent.

Plays a short beep when a chunk is sent to the LLM, so the user
knows the agent is processing speech.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


def play_beep(freq: int = 880, duration_ms: int = 100, volume: float = 0.3) -> None:
    """Play a short beep sound asynchronously.

    Uses sounddevice to generate and play a sine wave tone.
    Runs in a background thread to avoid blocking the main loop.
    """
    def _play():
        try:
            import sounddevice as sd
            sample_rate = 44100
            t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000), endpoint=False)
            tone = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            sd.play(tone, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.debug("Could not play beep: %s", e)

    threading.Thread(target=_play, daemon=True).start()
