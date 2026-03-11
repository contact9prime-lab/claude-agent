"""System audio capture for macOS using sounddevice.

On macOS, capturing system audio requires a virtual audio device like
BlackHole (https://github.com/ExistentialAudio/BlackHole). You set up a
Multi-Output Device in Audio MIDI Setup that routes to both your speakers
and BlackHole. Then this module captures from the BlackHole input.

Alternatively, you can capture from the microphone directly (simpler setup,
works for speaker-phone calls and in-person conversations).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from src.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioStream:
    """Captures audio from a system input device in a background thread.

    Audio frames are pushed into an internal queue, which consumers
    (like the VAD pipeline) read from.
    """

    def __init__(self, config: AudioConfig, device: Optional[int] = None) -> None:
        self.config = config
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the audio stream and begin capturing."""
        with self._lock:
            if self._running:
                return

            blocksize = int(self.config.sample_rate * self.config.chunk_duration_sec)

            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                blocksize=blocksize,
                channels=self.config.channels,
                dtype=self.config.dtype,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True

            device_info = sd.query_devices(self.device or sd.default.device[0])
            logger.info(
                "Audio capture started: %s @ %dHz",
                device_info["name"],
                self.config.sample_rate,
            )

    def stop(self) -> None:
        """Stop capturing audio."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            logger.info("Audio capture stopped")

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read the next audio chunk from the queue.

        Returns None if no data is available within the timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            logger.warning("Audio callback status: %s", status)
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            # Drop oldest frame to prevent memory buildup
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(indata.copy())


def list_audio_devices() -> list[dict]:
    """List all available audio input devices.

    Returns a list of dicts with 'index', 'name', and 'max_input_channels'.
    """
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
    return inputs


def find_blackhole_device() -> Optional[int]:
    """Find the BlackHole virtual audio device for system audio capture.

    Returns the device index if found, None otherwise.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None
