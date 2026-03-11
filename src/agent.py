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
from pathlib import Path
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

# How long silence before we end a session (seconds)
SESSION_END_SILENCE_SEC = 60

logger = logging.getLogger(__name__)


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
        self._paused = False  # Recording pause state
        self._recording_start_time: Optional[datetime] = None  # When recording started
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

        self._recording_start_time = datetime.now()
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

    def pause(self) -> None:
        """Pause recording (stops processing audio, keeps agent alive)."""
        if not self._paused:
            self._paused = True
            logger.info("Recording paused")
            try:
                from src.web.app import broadcast_event
                broadcast_event("recording_state", {"state": "paused"})
            except Exception:
                pass

    def resume(self) -> None:
        """Resume recording after pause."""
        if self._paused:
            self._paused = False
            self._recording_start_time = datetime.now()
            logger.info("Recording resumed")
            try:
                from src.web.app import broadcast_event
                broadcast_event("recording_state", {"state": "recording"})
            except Exception:
                pass

    def stop_and_process(self) -> dict:
        """Stop recording and process all buffered audio.

        This is called when the user clicks 'stop recording' in the UI.
        It flushes the VAD buffer, processes any remaining audio,
        and runs voice print analysis on the entire recording session.

        Returns a summary dict of the processing results.
        """
        logger.info("Stop recording requested — processing buffered audio...")

        try:
            from src.web.app import broadcast_event
            broadcast_event("recording_state", {"state": "processing"})
        except Exception:
            pass

        # Capture session ID before any state changes
        session_id = self._current_session_id

        # Flush any remaining speech in the VAD buffer
        # _process_chunk is synchronous — it waits for the LLM response
        remaining = self._vad.flush()
        if remaining:
            self._process_chunk(remaining)

        result = {
            "session_id": session_id,
            "voice_prints_detected": 0,
            "segments_processed": 0,
            "speakers_identified": [],
        }

        # Run voice print analysis on recent recordings.
        # We analyze ALL recent recordings (not just by session_id) because
        # some recordings may have been saved without a session link.
        vp_result = self._analyze_voice_prints_on_recent_recordings(session_id)
        result.update(vp_result)

        # End the session after all processing is done
        if session_id and self._current_session_id == session_id:
            self._end_session()

        # Pause recording
        self._paused = True
        self._recording_start_time = None

        try:
            from src.web.app import broadcast_event
            broadcast_event("recording_state", {"state": "stopped"})
            broadcast_event("processing_complete", result)
        except Exception:
            pass

        logger.info("Stop processing complete: %s", result)
        return result

    def _analyze_voice_prints_on_recent_recordings(self, session_id: int = None) -> dict:
        """Analyze recent recordings for voice prints using LLM-detected speakers.

        The LLM (Gemini) already identifies different speakers in transcripts
        as "Speaker 1", "Speaker 2", etc. This method:
        1. Collects all unique speaker labels from transcript segments
        2. Creates voice prints for each unique speaker
        3. Stores speaker-attributed segments in the database
        4. Optionally uses resemblyzer embeddings if available

        Falls back gracefully if resemblyzer is not installed.
        """
        result = {
            "voice_prints_detected": 0,
            "segments_processed": 0,
            "speakers_identified": [],
        }

        # Get recent recordings — try session filter first, fall back to recent
        recordings = self._db.list_recordings(limit=100)
        if session_id:
            target_recordings = [r for r in recordings if r.get("session_id") == session_id]
        else:
            target_recordings = []

        # If no session-linked recordings found, use the most recent ones
        if not target_recordings:
            target_recordings = recordings[:20]
            logger.info("No session-linked recordings found, analyzing %d recent recordings", len(target_recordings))

        if not target_recordings:
            logger.info("No recordings found for voice print analysis")
            return result

        logger.info("Analyzing %d recordings for voice prints...", len(target_recordings))

        # Collect all unique speaker labels from stored insights/segments
        # and also try resemblyzer if available
        has_resemblyzer = False
        try:
            from src.processing.speaker_id import (
                segment_audio_by_speaker,
                match_voice_print,
                update_averaged_embedding,
            )
            has_resemblyzer = True
            logger.info("resemblyzer available — will use audio embeddings")
        except ImportError:
            logger.info("resemblyzer not available — using LLM speaker labels for voice prints")

        existing_prints = self._db.get_voice_print_embeddings()
        # Track speaker label -> voice_print_id mapping for this session
        label_to_vp: dict[str, int] = {}
        seen_speakers = set()

        for rec in target_recordings:
            try:
                rec_id = rec.get("id")
                transcript = rec.get("transcript", "")
                duration = rec.get("duration_seconds", 0)

                if not transcript:
                    logger.debug("Recording #%s has no transcript, skipping", rec_id)
                    continue

                logger.info("Processing recording #%s (%s, %.1fs)...",
                            rec_id, rec["filename"], duration)

                # Strategy 1: Use resemblyzer audio embeddings if available
                if has_resemblyzer:
                    audio_path = self.config.storage.audio_dir / rec["filename"]
                    if audio_path.exists():
                        audio_data = audio_path.read_bytes()
                        segments = segment_audio_by_speaker(audio_data)
                        if segments:
                            self._process_embedding_segments(
                                rec_id, segments, existing_prints, label_to_vp,
                                seen_speakers, result
                            )
                            continue  # Skip LLM-based analysis if embeddings worked

                # Strategy 2: Use LLM-detected speaker labels from transcript
                speakers_in_transcript = self._extract_speakers_from_transcript(transcript)

                if not speakers_in_transcript:
                    # If no explicit speaker labels, treat entire recording as one speaker
                    speakers_in_transcript = {"Speaker 1": [(0.0, duration, transcript)]}

                logger.info("Found %d speakers in transcript of recording #%s: %s",
                            len(speakers_in_transcript), rec_id,
                            list(speakers_in_transcript.keys()))

                for speaker_label, segments_data in speakers_in_transcript.items():
                    # Get or create a voice print for this speaker label
                    vp_id = label_to_vp.get(speaker_label)

                    if vp_id is None:
                        # Check if we already have a voice print with this label
                        for ep in existing_prints:
                            if ep[1] == speaker_label:
                                vp_id = ep[0]
                                break

                    if vp_id is None:
                        # Create a new voice print (with empty embedding for LLM-only mode)
                        import pickle
                        empty_embedding = pickle.dumps(b"llm_speaker_label")

                        # Extract audio sample for this speaker
                        audio_sample = self._save_voice_print_audio_sample(
                            0, speaker_label, rec["filename"], segments_data
                        )

                        vp_id = self._db.add_voice_print(
                            speaker_label, empty_embedding,
                            audio_sample_file=audio_sample or "",
                        )

                        # Update the sample filename now that we have the ID
                        if audio_sample and "voiceprint_0_" in audio_sample:
                            new_filename = audio_sample.replace("voiceprint_0_", f"voiceprint_{vp_id}_")
                            old_path = self.config.storage.audio_dir / audio_sample
                            new_path = self.config.storage.audio_dir / new_filename
                            if old_path.exists():
                                old_path.rename(new_path)
                                self._db.update_voice_print_audio(vp_id, new_filename)
                                audio_sample = new_filename

                        existing_prints.append((vp_id, speaker_label, empty_embedding, 1))
                        result["voice_prints_detected"] += 1
                        logger.info("Created voice print '%s' (id=%d, audio=%s)",
                                    speaker_label, vp_id, audio_sample or "none")
                    else:
                        # Existing voice print — save audio sample if it doesn't have one
                        existing_vp = next(
                            (v for v in self._db.list_voice_prints() if v["id"] == vp_id), None
                        )
                        if existing_vp and not existing_vp.get("audio_sample_file"):
                            audio_sample = self._save_voice_print_audio_sample(
                                vp_id, speaker_label, rec["filename"], segments_data
                            )
                            if audio_sample:
                                self._db.update_voice_print_audio(vp_id, audio_sample)

                    label_to_vp[speaker_label] = vp_id
                    seen_speakers.add((vp_id, speaker_label))

                    # Store segments
                    for start_sec, end_sec, text in segments_data:
                        self._db.add_recording_segment(
                            recording_id=rec_id,
                            voice_print_id=vp_id,
                            speaker_label=speaker_label,
                            text=text,
                            start_seconds=start_sec,
                            end_seconds=end_sec,
                            confidence=0.8,
                        )
                        result["segments_processed"] += 1

                # Broadcast progress
                try:
                    from src.web.app import broadcast_event
                    broadcast_event("voice_print_progress", {
                        "recording_id": rec_id,
                        "segments": result["segments_processed"],
                        "clusters": len(seen_speakers),
                    })
                except Exception:
                    pass

            except Exception as e:
                logger.error("Failed to analyze recording #%s: %s", rec.get("id"), e, exc_info=True)

        result["speakers_identified"] = [
            {"id": vp_id, "label": label} for vp_id, label in seen_speakers
        ]

        logger.info("Voice print analysis complete: %d prints, %d segments, %d speakers",
                     result["voice_prints_detected"], result["segments_processed"],
                     len(result["speakers_identified"]))

        return result

    def _extract_speakers_from_transcript(self, transcript: str) -> dict[str, list[tuple[float, float, str]]]:
        """Extract speaker-labeled sections from a transcript.

        Parses patterns like "Speaker 1: ...", "Speaker 2: ..." etc.
        Returns {speaker_label: [(start_sec, end_sec, text), ...]}
        """
        import re
        speakers: dict[str, list[tuple[float, float, str]]] = {}

        # Match patterns like "Speaker 1:", "Speaker A:", "[Speaker 1]", etc.
        pattern = r'(?:^|\n)\s*\[?\s*(Speaker\s+\d+|Speaker\s+[A-Z])\s*\]?\s*[:\-]\s*'
        parts = re.split(pattern, transcript, flags=re.IGNORECASE)

        if len(parts) <= 1:
            # No speaker labels found — try alternative patterns
            # Try "Person 1:", "Voice 1:", etc.
            pattern2 = r'(?:^|\n)\s*\[?\s*((?:Person|Voice|Participant)\s+\d+)\s*\]?\s*[:\-]\s*'
            parts = re.split(pattern2, transcript, flags=re.IGNORECASE)

        if len(parts) <= 1:
            # Still no labels found
            return {}

        # parts[0] is text before first label (usually empty)
        # parts[1] is first label, parts[2] is first text, etc.
        current_pos = 0.0
        # Rough estimate: distribute time proportionally based on text length
        total_text_len = sum(len(parts[i]) for i in range(2, len(parts), 2))

        for i in range(1, len(parts), 2):
            label = parts[i].strip()
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if not text:
                continue

            # Estimate time range proportionally
            text_fraction = len(text) / max(total_text_len, 1)
            duration_estimate = text_fraction * 120  # rough estimate
            start = current_pos
            end = current_pos + duration_estimate
            current_pos = end

            if label not in speakers:
                speakers[label] = []
            speakers[label].append((start, end, text))

        return speakers

    def _process_embedding_segments(
        self, rec_id, segments, existing_prints, label_to_vp,
        seen_speakers, result
    ):
        """Process resemblyzer-based segments with embeddings."""
        import pickle
        from src.processing.speaker_id import match_voice_print, update_averaged_embedding

        # Group segments by cluster
        clusters = {}
        for seg in segments:
            cid = seg["cluster"]
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(seg)

        logger.info("Found %d speaker segments, %d clusters in recording #%s",
                     len(segments), len(clusters), rec_id)

        for cluster_id, cluster_segments in clusters.items():
            avg_embedding = np.mean(
                [s["embedding"] for s in cluster_segments], axis=0
            )

            match = match_voice_print(avg_embedding, existing_prints)

            if match:
                vp_id, vp_label, score = match
                logger.info("Matched cluster %d to '%s' (score=%.3f)", cluster_id, vp_label, score)
                for existing in existing_prints:
                    if existing[0] == vp_id:
                        new_count = existing[3] + 1
                        new_emb = update_averaged_embedding(
                            existing[2], avg_embedding, existing[3]
                        )
                        self._db.update_voice_print_embedding(vp_id, new_emb, new_count)
                        break
            else:
                vp_label = f"Voice {len(existing_prints) + 1}"
                emb_bytes = pickle.dumps(avg_embedding)
                vp_id = self._db.add_voice_print(vp_label, emb_bytes)
                existing_prints.append((vp_id, vp_label, emb_bytes, 1))
                result["voice_prints_detected"] += 1
                logger.info("Created voice print '%s' (id=%d)", vp_label, vp_id)

            seen_speakers.add((vp_id, vp_label))

            for seg in cluster_segments:
                self._db.add_recording_segment(
                    recording_id=rec_id,
                    voice_print_id=vp_id,
                    speaker_label=vp_label,
                    text="",
                    start_seconds=seg["start"],
                    end_seconds=seg["end"],
                    confidence=0.0,
                )
                result["segments_processed"] += 1

    def _extract_audio_segment(self, wav_path: Path, start_sec: float, end_sec: float, out_filename: str) -> Optional[str]:
        """Extract a time segment from a WAV file and save it as a new file.

        Returns the filename if successful, None otherwise.
        """
        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                total_frames = wf.getnframes()

                start_frame = int(start_sec * sr)
                end_frame = min(int(end_sec * sr), total_frames)

                if start_frame >= total_frames or start_frame >= end_frame:
                    return None

                wf.setpos(start_frame)
                frames = wf.readframes(end_frame - start_frame)

            out_path = self.config.storage.audio_dir / out_filename
            with wave.open(str(out_path), "wb") as wf_out:
                wf_out.setnchannels(channels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(sr)
                wf_out.writeframes(frames)

            logger.debug("Extracted audio segment: %s (%.1f-%.1fs)", out_filename, start_sec, end_sec)
            return out_filename
        except Exception as e:
            logger.debug("Failed to extract audio segment: %s", e)
            return None

    def _save_voice_print_audio_sample(
        self, vp_id: int, speaker_label: str, rec_filename: str,
        segments_data: list[tuple[float, float, str]]
    ) -> Optional[str]:
        """Save a combined audio sample for a voice print from its segments.

        Takes the first few segments (up to ~15 seconds) of a speaker's audio
        and concatenates them into a single sample file.
        """
        audio_path = self.config.storage.audio_dir / rec_filename
        if not audio_path.exists():
            return None

        try:
            with wave.open(str(audio_path), "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                total_frames = wf.getnframes()
                all_audio = wf.readframes(total_frames)

            # Concatenate segments for this speaker (up to ~15 seconds)
            collected_frames = b""
            collected_duration = 0.0
            max_sample_duration = 15.0

            for start_sec, end_sec, text in segments_data:
                if collected_duration >= max_sample_duration:
                    break

                start_frame = int(start_sec * sr)
                end_frame = min(int(end_sec * sr), total_frames)
                bytes_per_frame = channels * sampwidth
                start_byte = start_frame * bytes_per_frame
                end_byte = end_frame * bytes_per_frame

                if start_byte < len(all_audio) and end_byte <= len(all_audio):
                    collected_frames += all_audio[start_byte:end_byte]
                    collected_duration += (end_sec - start_sec)

            if not collected_frames:
                return None

            # Save as WAV
            safe_label = speaker_label.replace(" ", "_").lower()
            sample_filename = f"voiceprint_{vp_id}_{safe_label}.wav"
            out_path = self.config.storage.audio_dir / sample_filename

            with wave.open(str(out_path), "wb") as wf_out:
                wf_out.setnchannels(channels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(sr)
                wf_out.writeframes(collected_frames)

            logger.info("Saved voice print audio sample: %s (%.1fs)", sample_filename, collected_duration)
            return sample_filename

        except Exception as e:
            logger.debug("Failed to save voice print audio sample: %s", e)
            return None

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_recording(self) -> bool:
        return self._running and not self._paused

    @property
    def recording_start_time(self) -> Optional[datetime]:
        return self._recording_start_time

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """Core audio processing loop."""
        while self._running:
            # If paused, just drain the audio buffer without processing
            if self._paused:
                self._audio.read(timeout=1.0)
                time.sleep(0.1)
                continue

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

        # Step 4: Save audio file for playback
        audio_filename = self._save_audio_chunk(chunk)

        # Step 5: Store everything
        self._db.save_insight(insight, self._current_session_id)
        self._db.append_transcript(
            self._current_session_id,
            insight.transcript,
            chunk.duration_seconds,
        )

        # Save recording entry for playback
        self._db.add_recording(
            session_id=self._current_session_id,
            filename=audio_filename,
            duration_seconds=chunk.duration_seconds,
            transcript=insight.transcript or "",
            summary=insight.summary or "",
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
                "segments": [
                    {"text": s.text, "start_seconds": s.start_seconds,
                     "end_seconds": s.end_seconds, "speaker": s.speaker}
                    for s in insight.segments
                ],
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
                "audio_file": getattr(chunk, '_saved_filename', None),
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
        """Replace generic speaker labels with real names using voice prints and enrolled speakers."""
        try:
            from src.processing.speaker_id import (
                extract_raw_embedding,
                match_voice_print,
                identify_speaker,
            )

            # First try voice prints (auto-detected voices)
            voice_prints = self._db.get_voice_print_embeddings()
            embedding = extract_raw_embedding(chunk.audio_data)

            if embedding is not None and voice_prints:
                match = match_voice_print(embedding, voice_prints)
                if match:
                    vp_id, vp_label, score = match
                    if insight.segments:
                        for seg in insight.segments:
                            if seg.speaker and seg.speaker.startswith("Speaker"):
                                seg.speaker = vp_label
                    if insight.transcript:
                        for i in range(1, 10):
                            if f"Speaker {i}" in insight.transcript:
                                insight.transcript = insight.transcript.replace(
                                    f"Speaker {i}", vp_label, 1
                                )
                                break
                    return

            # Fall back to enrolled speaker profiles
            known = self._db.get_speaker_embeddings()
            if not known:
                return
            name = identify_speaker(chunk.audio_data, known)
            if name and insight.segments:
                for seg in insight.segments:
                    if seg.speaker and seg.speaker.startswith("Speaker"):
                        seg.speaker = name
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

    def _save_audio_chunk(self, chunk: AudioChunk) -> str:
        """Save an audio chunk to disk. Returns the filename."""
        timestamp = chunk.timestamp_start.strftime("%Y%m%d_%H%M%S")
        filename = f"chunk_{timestamp}.wav"
        path = self.config.storage.audio_dir / filename
        path.write_bytes(chunk.audio_data)
        logger.info("Saved audio chunk: %s", path)
        return filename

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
