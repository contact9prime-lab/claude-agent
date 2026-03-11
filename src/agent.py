"""DeskVoice Agent — the main orchestration loop.

This is the brain of the system. It:
1. Captures system audio via AudioStream
2. Runs it through Silero VAD to detect speech
3. Classifies audio type (speech vs music vs noise)
4. Sends speech chunks to Gemini Flash for transcription + insight extraction
5. Stores everything locally in SQLite
6. Manages session lifecycle (start on first speech, end on long silence)

The agent runs as a daemon and is designed to be as quiet as possible:
- VAD runs locally, no network calls for silence/noise
- Local classifier filters out music before hitting Gemini
- Only real speech costs tokens
"""

from __future__ import annotations

import io
import logging
import signal
import threading
import time
import wave
from datetime import datetime
from typing import Optional

import numpy as np

from src.capture.audio_stream import AudioStream
from src.capture.notifier import play_beep
from src.capture.vad import SileroVAD
from src.config import AgentConfig
from src.models.models import AudioChunk, AudioType
from src.processing.classifier import classify_audio
from src.processing.llm_provider import StreamCallback, create_provider
from src.storage.database import Database

logger = logging.getLogger(__name__)

# How long of silence before we end a session (seconds)
SESSION_END_SILENCE_SEC = 60


class DeskVoiceAgent:
    """The always-on desk voice agent."""

    def __init__(self, config: AgentConfig, device: Optional[int] = None) -> None:
        self.config = config
        self.config.storage.ensure_dirs()

        # Components
        self._audio = AudioStream(config.audio, device=device)
        self._vad = SileroVAD(config.audio)
        self._llm = create_provider(config.llm)
        self._db = Database(config.storage.db_path)

        # State
        self._running = False
        self._current_session_id: Optional[int] = None
        self._last_speech_time: Optional[datetime] = None
        self._tui = None  # Optional TUI reference for pushing updates
        self._stats = {
            "chunks_processed": 0,
            "speech_seconds": 0.0,
            "gemini_calls": 0,
            "tasks_found": 0,
            "sessions_completed": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the agent. Blocks until stop() is called."""
        logger.info("DeskVoice agent starting...")

        # Validate config
        errors = self.config.validate()
        if errors:
            for e in errors:
                logger.error("Config error: %s", e)
            raise RuntimeError(f"Configuration errors: {errors}")

        # Initialize components
        self._db.connect()
        self._vad.load_model()
        self._audio.start()
        self._running = True

        logger.info("DeskVoice agent running. Listening for speech...")

        # Start background reminder checker
        self._start_reminder_checker()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the agent gracefully."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down...")

        # Flush any remaining VAD buffer
        remaining = self._vad.flush()
        if remaining:
            self._process_chunk(remaining)

        # End current session
        if self._current_session_id:
            self._end_session()

        self._audio.stop()
        self._db.close()
        logger.info("DeskVoice agent stopped. Stats: %s", self._stats)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """Core audio processing loop."""
        while self._running:
            # Read audio frame from capture
            frame = self._audio.read(timeout=1.0)
            if frame is None:
                self._check_session_timeout()
                continue

            # Run through VAD
            chunk = self._vad.process_frame(frame)

            if chunk is not None:
                self._process_chunk(chunk)

            # Check if we should end the session due to long silence
            self._check_session_timeout()

    def _process_chunk(self, chunk: AudioChunk) -> None:
        """Process a speech chunk: classify, transcribe, extract, store."""
        # Ensure we have an active session
        if self._current_session_id is None:
            self._start_session()

        self._last_speech_time = datetime.now()

        # Step 1: Local classification (free, no network)
        if self.config.classify_before_transcribe:
            wav_audio = self._wav_to_numpy(chunk.audio_data)
            local_type = classify_audio(wav_audio, chunk.sample_rate)

            if local_type in (AudioType.MUSIC, AudioType.SILENCE, AudioType.NOISE):
                logger.debug("Skipping %s chunk (%.1fs)", local_type.value, chunk.duration_seconds)
                return

        # Step 2: Send to LLM for transcription + insight extraction
        if chunk.duration_seconds >= 60:
            play_beep()

        try:
            callback = self._make_stream_callback()
            insight = self._llm.stream_transcribe(chunk, callback=callback)
            self._stats["gemini_calls"] += 1
        except Exception as e:
            logger.error("LLM processing failed: %s", e)
            # Save raw audio for later reprocessing
            self._save_audio_chunk(chunk)
            return

        # Log token usage immediately
        if insight.token_usage:
            tu = insight.token_usage
            self._stats["total_input_tokens"] += tu.input_tokens
            self._stats["total_output_tokens"] += tu.output_tokens
            self._stats["total_tokens"] += tu.total_tokens
            logger.info(
                "[tokens] in=%d out=%d total=%d (cumulative: %d)",
                tu.input_tokens,
                tu.output_tokens,
                tu.total_tokens,
                self._stats["total_tokens"],
            )

        # Step 3: Check if Gemini thinks this isn't actually speech
        if insight.audio_type in (AudioType.MUSIC, AudioType.NOISE):
            logger.info("Gemini classified as %s, skipping storage", insight.audio_type.value)
            return

        # Step 3b: Try to identify speakers using enrolled profiles
        self._tag_speakers(insight, chunk)

        # Step 4: Store everything
        self._db.save_insight(insight, self._current_session_id)
        self._db.append_transcript(
            self._current_session_id,
            insight.transcript,
            chunk.duration_seconds,
        )

        self._stats["chunks_processed"] += 1
        self._stats["speech_seconds"] += chunk.duration_seconds
        self._stats["tasks_found"] += len(insight.tasks)

        # Log activity in real-time
        if insight.transcript:
            preview = insight.transcript[:120] + ("..." if len(insight.transcript) > 120 else "")
            logger.info("[transcript] %s", preview)
        if insight.tasks:
            for task in insight.tasks:
                priority_marker = {"high": "!!!", "medium": "!!", "low": "!"}.get(task.priority.value, "")
                assignee = f" -> {task.assignee}" if task.assignee else ""
                due = f" (due: {task.due_hint})" if task.due_hint else ""
                logger.info("[task] %s %s%s%s", priority_marker, task.description, assignee, due)
        if insight.decisions:
            for decision in insight.decisions:
                logger.info("[decision] %s", decision)
        if insight.questions:
            for question in insight.questions:
                logger.info("[question] %s", question)
        if insight.hashtags:
            tags = ", ".join(f"#{h.tag}" for h in insight.hashtags)
            logger.info("[tags] %s", tags)

        # Broadcast to web UI and TUI
        self._broadcast_insight(insight, chunk)

        # Push to TUI if attached
        if self._tui:
            try:
                self._tui.push_insight(insight, chunk.duration_seconds)
            except Exception:
                pass

    def _broadcast_insight(self, insight, chunk: AudioChunk) -> None:
        """Send insight data to connected web UI clients."""
        try:
            from src.web.app import broadcast_event, _agent_stats
            _agent_stats.clear()
            _agent_stats.update(self._stats)
            broadcast_event("insight", {
                "transcript": insight.transcript,
                "summary": insight.summary,
                "tasks": [
                    {"id": t.id, "description": t.description, "assignee": t.assignee,
                     "priority": t.priority.value, "due_hint": t.due_hint}
                    for t in insight.tasks
                ],
                "decisions": insight.decisions,
                "questions": insight.questions,
                "hashtags": [{"tag": h.tag, "context": h.context} for h in insight.hashtags],
                "token_usage": {
                    "input_tokens": insight.token_usage.input_tokens,
                    "output_tokens": insight.token_usage.output_tokens,
                    "total_tokens": insight.token_usage.total_tokens,
                } if insight.token_usage else None,
                "duration_seconds": chunk.duration_seconds,
            })
        except Exception:
            pass  # Web UI not running

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _start_session(self) -> None:
        """Start a new conversation session."""
        self._current_session_id = self._db.create_session()
        self._last_speech_time = datetime.now()
        logger.info("Session started: #%d", self._current_session_id)
        try:
            from src.web.app import broadcast_event
            broadcast_event("session_started", {"session_id": self._current_session_id})
        except Exception:
            pass

    def _end_session(self) -> None:
        """End the current session."""
        if self._current_session_id is None:
            return

        self._db.end_session(self._current_session_id)
        self._stats["sessions_completed"] += 1
        logger.info("Session ended: #%d", self._current_session_id)
        try:
            from src.web.app import broadcast_event
            broadcast_event("session_ended", {"session_id": self._current_session_id})
        except Exception:
            pass

        # Generate a summary for the session using Gemini
        self._summarize_session(self._current_session_id)

        self._current_session_id = None
        self._last_speech_time = None

    def _summarize_session(self, session_id: int) -> None:
        """Generate a final summary for a completed session."""
        session = self._db.get_session(session_id)
        if not session or not session.get("transcript"):
            return

        # We could call Gemini here for a final session summary,
        # but for now we'll use the accumulated chunk summaries.
        # This saves tokens — the per-chunk summaries are usually enough.
        logger.info(
            "Session #%d completed: %.0fs of speech",
            session_id,
            session.get("total_speech_seconds", 0),
        )

    def _check_session_timeout(self) -> None:
        """End session if there's been a long silence."""
        if self._current_session_id is None:
            return
        if self._last_speech_time is None:
            return

        elapsed = (datetime.now() - self._last_speech_time).total_seconds()
        if elapsed > SESSION_END_SILENCE_SEC:
            logger.info(
                "%.0fs of silence — ending session #%d",
                elapsed,
                self._current_session_id,
            )
            self._end_session()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _make_stream_callback(self) -> StreamCallback:
        """Create a streaming callback that broadcasts partial tokens."""
        agent = self

        class _Callback(StreamCallback):
            def on_token(self, token: str) -> None:
                try:
                    from src.web.app import broadcast_event
                    broadcast_event("stream_token", {"token": token})
                except Exception:
                    pass
                if agent._tui:
                    try:
                        agent._tui.call_from_thread(
                            agent._tui.query_one("#transcript-panel").write, token
                        )
                    except Exception:
                        pass

            def on_complete(self, full_text: str) -> None:
                try:
                    from src.web.app import broadcast_event
                    broadcast_event("stream_complete", {"text": full_text})
                except Exception:
                    pass

        return _Callback()

    # ------------------------------------------------------------------
    # Speaker identification
    # ------------------------------------------------------------------

    def _tag_speakers(self, insight, chunk: AudioChunk) -> None:
        """Replace generic speaker labels with real names if enrolled."""
        try:
            from src.processing.speaker_id import identify_speaker
            known = self._db.get_speaker_embeddings()
            if not known:
                return
            name = identify_speaker(chunk.audio_data, known)
            if name and insight.segments:
                for seg in insight.segments:
                    if seg.speaker and seg.speaker.startswith("Speaker"):
                        seg.speaker = name
                # Also fix the transcript text
                if insight.transcript:
                    for i in range(1, 10):
                        if f"Speaker {i}" in insight.transcript:
                            insight.transcript = insight.transcript.replace(
                                f"Speaker {i}", name, 1
                            )
                            break
        except ImportError:
            pass  # resemblyzer not installed
        except Exception as e:
            logger.debug("Speaker ID failed: %s", e)

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    def _start_reminder_checker(self) -> None:
        """Start a background thread that checks for due reminders."""
        def _check_loop():
            while self._running:
                try:
                    due = self._db.get_due_reminders()
                    for task in due:
                        logger.info(
                            "[reminder] Task #%d: %s",
                            task["id"],
                            task["description"],
                        )
                        play_beep(freq=660, duration_ms=200)
                        try:
                            from src.web.app import broadcast_event
                            broadcast_event("reminder", {
                                "task_id": task["id"],
                                "description": task["description"],
                            })
                        except Exception:
                            pass
                        self._db.clear_reminder(task["id"])
                except Exception as e:
                    logger.debug("Reminder check failed: %s", e)
                time.sleep(30)  # Check every 30 seconds

        thread = threading.Thread(target=_check_loop, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_audio_chunk(self, chunk: AudioChunk) -> None:
        """Save an audio chunk to disk for later reprocessing."""
        timestamp = chunk.timestamp_start.strftime("%Y%m%d_%H%M%S")
        path = self.config.storage.audio_dir / f"chunk_{timestamp}.wav"
        path.write_bytes(chunk.audio_data)
        logger.info("Saved audio chunk for later: %s", path)

    @staticmethod
    def _wav_to_numpy(wav_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes back to numpy array for classification."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16)

    @property
    def stats(self) -> dict:
        return self._stats.copy()
