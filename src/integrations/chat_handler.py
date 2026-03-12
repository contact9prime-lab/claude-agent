"""Unified chat handler for all messaging integrations.

Processes commands and queries from Telegram, WhatsApp, Slack, etc.
One handler, many frontends — all chat apps share the same brain.

Supported interactions:
- Voice messages → transcribe + extract insights
- Text queries → search second brain ("what did John say about pricing?")
- Commands → /briefing, /tasks, /people, /record, /search
- Inline buttons → mark task done, view session, play audio
"""

from __future__ import annotations

import io
import logging
import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import AgentConfig
from src.models.models import AudioChunk, AudioType
from src.storage.database import Database

logger = logging.getLogger(__name__)


class ChatResponse:
    """Structured response to send back to any chat platform."""

    def __init__(
        self,
        text: str = "",
        audio_path: Optional[Path] = None,
        buttons: Optional[list[dict]] = None,
        parse_mode: str = "markdown",
    ):
        self.text = text
        self.audio_path = audio_path
        # buttons: [{"label": "Mark Done", "callback": "done:123"}, ...]
        self.buttons = buttons or []
        self.parse_mode = parse_mode


class ChatHandler:
    """Unified handler that processes all chat interactions.

    Works with any messaging platform — just pass text/voice and get back
    a ChatResponse to send to the user.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        config.storage.ensure_dirs()
        self.db = Database(config.storage.db_path)
        self.db.connect()
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from src.processing.llm_provider import create_provider
            self._llm = create_provider(self.config.llm)
        return self._llm

    def handle_text(self, text: str, user_id: str = "") -> ChatResponse:
        """Handle a text message — could be a command or a natural language query."""
        text = text.strip()

        # Commands
        if text.startswith("/"):
            return self._handle_command(text, user_id)

        # Natural language query — search the second brain
        return self._handle_query(text, user_id)

    def handle_voice(
        self, audio_bytes: bytes, duration_sec: float = 0, user_id: str = "",
        mime_type: str = "audio/ogg",
    ) -> ChatResponse:
        """Handle a voice message — transcribe and extract insights."""
        # Convert to WAV for processing
        wav_bytes = self._convert_to_wav(audio_bytes, mime_type)
        if wav_bytes is None:
            return ChatResponse(text="Could not process audio. Try again?")

        # Get duration
        if duration_sec == 0:
            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    duration_sec = wf.getnframes() / wf.getframerate()
            except Exception:
                duration_sec = 0

        # Save to storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{user_id}_{timestamp}.wav"
        audio_path = self.config.storage.audio_dir / filename
        audio_path.write_bytes(wav_bytes)

        # Create audio chunk for LLM processing
        chunk = AudioChunk(
            audio_data=wav_bytes,
            sample_rate=16000,
            duration_seconds=duration_sec,
            timestamp_start=datetime.now(),
            timestamp_end=datetime.now(),
            audio_type=AudioType.SPEECH,
            source="chat_voice",
        )

        # Transcribe via LLM
        try:
            llm = self._get_llm()
            insight = llm.stream_transcribe(chunk)
        except Exception as e:
            logger.error("Voice transcription failed: %s", e)
            return ChatResponse(text="Transcription failed. I'll save the audio for later.")

        # Save to database
        rec_id = self.db.add_recording(
            session_id=None,
            filename=filename,
            duration_seconds=duration_sec,
            transcript=insight.transcript or "",
            summary=insight.summary or "",
        )
        self.db.save_insight(insight, session_id=None)

        # Build response
        parts = []
        if insight.transcript:
            parts.append(f"*Transcript:*\n{insight.transcript}")
        if insight.summary:
            parts.append(f"\n*Summary:* {insight.summary}")

        if insight.tasks:
            task_lines = []
            for t in insight.tasks:
                prefix = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(t.priority.value, "⚪")
                line = f"{prefix} {t.description}"
                if t.assignee:
                    line += f" → {t.assignee}"
                if t.due_hint:
                    line += f" (due: {t.due_hint})"
                task_lines.append(line)
            parts.append("\n*Tasks:*\n" + "\n".join(task_lines))

        if insight.decisions:
            parts.append("\n*Decisions:*\n" + "\n".join(f"✅ {d}" for d in insight.decisions))

        if insight.hashtags:
            tags = " ".join(f"#{h.tag}" for h in insight.hashtags)
            parts.append(f"\n{tags}")

        # Buttons for task actions
        buttons = []
        for t in insight.tasks:
            if t.id:
                buttons.append({"label": f"✓ Done: {t.description[:30]}", "callback": f"done:{t.id}"})

        return ChatResponse(
            text="\n".join(parts) if parts else "I heard that but couldn't extract anything meaningful.",
            buttons=buttons,
        )

    def handle_callback(self, callback_data: str, user_id: str = "") -> ChatResponse:
        """Handle inline button callbacks like 'done:123'."""
        if callback_data.startswith("done:"):
            task_id = int(callback_data.split(":")[1])
            self.db.complete_task(task_id)
            return ChatResponse(text=f"✅ Task #{task_id} marked done!")

        if callback_data.startswith("session:"):
            session_id = int(callback_data.split(":")[1])
            return self._get_session_detail(session_id)

        if callback_data.startswith("person:"):
            person_id = int(callback_data.split(":")[1])
            return self._get_person_detail(person_id)

        return ChatResponse(text="Unknown action.")

    def _handle_command(self, text: str, user_id: str) -> ChatResponse:
        """Route slash commands."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().rstrip("@deskvoicebot")  # Strip bot mention
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/start": self._cmd_start,
            "/help": self._cmd_help,
            "/briefing": self._cmd_briefing,
            "/today": self._cmd_briefing,
            "/tasks": self._cmd_tasks,
            "/people": self._cmd_people,
            "/search": self._cmd_search,
            "/recent": self._cmd_recent,
            "/digest": self._cmd_digest,
            "/status": self._cmd_status,
        }

        handler = handlers.get(cmd)
        if handler:
            return handler(arg, user_id)

        # Unknown command — treat as search query
        return self._handle_query(text.lstrip("/"), user_id)

    def _handle_query(self, query: str, user_id: str) -> ChatResponse:
        """Handle natural language queries by searching the second brain."""
        if len(query) < 2:
            return ChatResponse(text="Ask me anything! Try: \"what did we discuss about pricing?\"")

        # Search transcripts
        results = self.db.search_transcripts(query, limit=5)

        # Search tasks
        all_tasks = self.db.list_tasks(pending_only=False)
        q_lower = query.lower()
        matching_tasks = [
            t for t in all_tasks
            if q_lower in (t.get("description", "") or "").lower()
            or q_lower in (t.get("assignee", "") or "").lower()
        ][:5]

        if not results and not matching_tasks:
            return ChatResponse(
                text=f"Nothing found for \"{query}\". Try recording more conversations to build your second brain!",
            )

        parts = [f"🔍 *Results for \"{query}\":*\n"]
        buttons = []

        if matching_tasks:
            parts.append("*Tasks:*")
            for t in matching_tasks:
                status = "✅" if t.get("completed") else "⬜"
                parts.append(f"  {status} {t['description']}")
                if not t.get("completed") and t.get("id"):
                    buttons.append({"label": f"✓ {t['description'][:25]}", "callback": f"done:{t['id']}"})

        if results:
            parts.append("\n*From conversations:*")
            for s in results:
                title = s.get("title") or f"Session #{s['id']}"
                date = s.get("started_at", "")[:10]
                transcript = s.get("transcript", "")
                # Extract context around the match
                idx = transcript.lower().find(q_lower)
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(transcript), idx + len(query) + 60)
                    context = ("..." if start > 0 else "") + transcript[start:end] + ("..." if end < len(transcript) else "")
                else:
                    context = transcript[:120] + "..." if len(transcript) > 120 else transcript
                parts.append(f"\n📅 *{title}* ({date})\n_{context}_")
                buttons.append({"label": f"📄 {title[:25]}", "callback": f"session:{s['id']}"})

        return ChatResponse(text="\n".join(parts), buttons=buttons[:6])

    # ---- Commands ----

    def _cmd_start(self, arg: str, user_id: str) -> ChatResponse:
        return ChatResponse(
            text=(
                "👋 *Welcome to DeskVoice — Your Second Brain*\n\n"
                "I capture and remember everything from your conversations.\n\n"
                "*Quick actions:*\n"
                "🎤 Send a voice message → I'll transcribe & extract tasks\n"
                "💬 Ask me anything → I'll search your memory\n\n"
                "*Commands:*\n"
                "/briefing — Today's summary\n"
                "/tasks — Pending action items\n"
                "/people — People you've talked to\n"
                "/search <query> — Search your brain\n"
                "/recent — Recent sessions\n"
                "/digest — End-of-day digest\n"
                "/help — Full help\n\n"
                "Or just type a question like:\n"
                "_\"What did we decide about the budget?\"_"
            ),
        )

    def _cmd_help(self, arg: str, user_id: str) -> ChatResponse:
        return self._cmd_start(arg, user_id)

    def _cmd_briefing(self, arg: str, user_id: str) -> ChatResponse:
        briefing = self.db.get_daily_briefing(arg or None)

        sessions = briefing.get("session_count", 0)
        speech_min = round(briefing.get("total_speech_seconds", 0) / 60, 1)
        pending = briefing.get("pending_tasks", 0)
        speakers = briefing.get("speakers", [])
        tags = briefing.get("tags", [])

        parts = [
            f"📋 *Daily Briefing — {briefing.get('date', 'Today')}*\n",
            f"📊 {sessions} sessions | {speech_min}m speech | {pending} pending tasks",
        ]

        if speakers:
            people_str = ", ".join(
                f"{s.get('speaker_name', 'Unknown')} ({round(s.get('total_seconds', 0) / 60)}m)"
                for s in speakers
            )
            parts.append(f"\n👥 *People:* {people_str}")

        if tags:
            tag_str = " ".join(f"#{t['tag']}" for t in tags[:10])
            parts.append(f"\n🏷️ {tag_str}")

        # Pending tasks
        tasks = briefing.get("tasks", [])
        pending_tasks = [t for t in tasks if not t.get("completed")]
        if pending_tasks:
            parts.append("\n📝 *Pending tasks:*")
            buttons = []
            for t in pending_tasks[:8]:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(t.get("priority", ""), "⚪")
                desc = t.get("description", "")
                parts.append(f"  {priority_icon} {desc}")
                if t.get("id"):
                    buttons.append({"label": f"✓ {desc[:25]}", "callback": f"done:{t['id']}"})
            return ChatResponse(text="\n".join(parts), buttons=buttons[:6])

        return ChatResponse(text="\n".join(parts))

    def _cmd_tasks(self, arg: str, user_id: str) -> ChatResponse:
        show_all = arg.strip().lower() in ("all", "--all")
        tasks = self.db.list_tasks(pending_only=not show_all)

        if not tasks:
            return ChatResponse(text="✅ No pending tasks! You're all caught up.")

        parts = [f"📝 *{'All' if show_all else 'Pending'} Tasks ({len(tasks)}):*\n"]
        buttons = []

        for t in tasks[:15]:
            status = "✅" if t.get("completed") else "⬜"
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(t.get("priority", ""), "⚪")
            desc = t.get("description", "")
            line = f"{status} {priority_icon} {desc}"
            if t.get("assignee"):
                line += f" → {t['assignee']}"
            if t.get("due_hint"):
                line += f" ⏰ {t['due_hint']}"
            parts.append(line)

            if not t.get("completed") and t.get("id"):
                buttons.append({"label": f"✓ {desc[:25]}", "callback": f"done:{t['id']}"})

        return ChatResponse(text="\n".join(parts), buttons=buttons[:8])

    def _cmd_people(self, arg: str, user_id: str) -> ChatResponse:
        vps = self.db.list_voice_prints()
        if not vps:
            return ChatResponse(text="No people detected yet. Record some conversations first!")

        parts = ["👥 *People You've Talked To:*\n"]
        buttons = []

        for vp in vps:
            name = vp.get("mapped_speaker_name") or vp.get("label", "Unknown")
            samples = vp.get("sample_count", 0)
            created = (vp.get("created_at", "") or "")[:10]
            parts.append(f"  🗣️ *{name}* — {samples} voice samples (since {created})")
            buttons.append({"label": f"📖 {name[:20]}", "callback": f"person:{vp['id']}"})

        return ChatResponse(text="\n".join(parts), buttons=buttons[:8])

    def _cmd_search(self, arg: str, user_id: str) -> ChatResponse:
        if not arg:
            return ChatResponse(text="Usage: `/search <query>`\n\nOr just type your question directly!")
        return self._handle_query(arg, user_id)

    def _cmd_recent(self, arg: str, user_id: str) -> ChatResponse:
        sessions = self.db.list_sessions(limit=10)
        if not sessions:
            return ChatResponse(text="No sessions yet. Start recording!")

        parts = ["📅 *Recent Sessions:*\n"]
        buttons = []

        for s in sessions:
            time_str = s.get("started_at", "")[:16].replace("T", " ")
            dur = round(s.get("total_speech_seconds", 0) / 60)
            title = s.get("title") or s.get("summary", "")[:40] or f"Session #{s['id']}"
            status = "🟢" if s.get("status") == "active" else "⚪"
            parts.append(f"  {status} *{title}* — {time_str} ({dur}m)")
            buttons.append({"label": f"📄 {title[:20]}", "callback": f"session:{s['id']}"})

        return ChatResponse(text="\n".join(parts), buttons=buttons[:8])

    def _cmd_digest(self, arg: str, user_id: str) -> ChatResponse:
        briefing = self.db.get_daily_briefing()
        all_pending = self.db.list_tasks(pending_only=True)

        parts = [
            "🌙 *End-of-Day Digest*\n",
            f"📊 {briefing.get('session_count', 0)} conversations today",
            f"⏱️ {round(briefing.get('total_speech_seconds', 0) / 60)}m total speech",
        ]

        # Overdue tasks (from previous days)
        overdue = []
        today = datetime.now().strftime("%Y-%m-%d")
        for t in all_pending:
            created = t.get("created_at", "")
            if created and not created.startswith(today):
                days = (datetime.now() - datetime.fromisoformat(created)).days
                overdue.append((t, days))

        if overdue:
            parts.append(f"\n⚠️ *{len(overdue)} overdue items:*")
            for t, days in overdue[:5]:
                parts.append(f"  🔴 {t['description']} ({days}d ago)")

        if all_pending:
            parts.append(f"\n📝 *{len(all_pending)} total pending tasks*")

        buttons = [{"label": "📋 Full Briefing", "callback": "cmd:briefing"}]
        if all_pending:
            buttons.append({"label": "📝 All Tasks", "callback": "cmd:tasks"})

        return ChatResponse(text="\n".join(parts), buttons=buttons)

    def _cmd_status(self, arg: str, user_id: str) -> ChatResponse:
        sessions = self.db.list_sessions(limit=1)
        recordings = self.db.list_recordings(limit=1)
        tasks = self.db.list_tasks(pending_only=True)
        vps = self.db.list_voice_prints()

        parts = [
            "🔧 *DeskVoice Status*\n",
            f"📊 Sessions: {len(self.db.list_sessions(limit=1000))}",
            f"🎤 Recordings: {len(self.db.list_recordings(limit=1000))}",
            f"📝 Pending tasks: {len(tasks)}",
            f"👥 Voice prints: {len(vps)}",
            f"🧠 LLM: {self.config.llm.provider} / {self.config.llm.model}",
            f"💾 DB: {self.config.storage.db_path}",
        ]

        return ChatResponse(text="\n".join(parts))

    # ---- Detail views ----

    def _get_session_detail(self, session_id: int) -> ChatResponse:
        try:
            detail = self.db.get_session_detail(session_id)
        except Exception:
            detail = self.db.get_session(session_id)

        if not detail:
            return ChatResponse(text="Session not found.")

        title = detail.get("title") or f"Session #{session_id}"
        parts = [f"📄 *{title}*\n"]

        time_str = (detail.get("started_at", "") or "")[:16].replace("T", " ")
        dur = round(detail.get("total_speech_seconds", 0) / 60)
        parts.append(f"📅 {time_str} | ⏱️ {dur}m")

        if detail.get("summary"):
            parts.append(f"\n*Summary:* {detail['summary']}")

        transcript = detail.get("transcript", "")
        if transcript:
            # Truncate for chat
            if len(transcript) > 1500:
                transcript = transcript[:1500] + "..."
            parts.append(f"\n*Transcript:*\n_{transcript}_")

        return ChatResponse(text="\n".join(parts))

    def _get_person_detail(self, person_id: int) -> ChatResponse:
        segments = self.db.conn.execute(
            """SELECT rs.text, r.created_at, r.summary
               FROM recording_segments rs
               LEFT JOIN recordings r ON rs.recording_id = r.id
               WHERE rs.voice_print_id = ?
               ORDER BY r.created_at DESC LIMIT 20""",
            (person_id,),
        ).fetchall()
        segments = [dict(s) for s in segments]

        vps = self.db.list_voice_prints()
        vp = next((v for v in vps if v["id"] == person_id), None)
        name = vp.get("mapped_speaker_name") or vp.get("label", "Unknown") if vp else "Unknown"

        if not segments:
            return ChatResponse(text=f"No conversation history found for {name}.")

        parts = [f"👤 *{name}* — {len(segments)} segments\n"]

        for seg in segments[:10]:
            if seg.get("text") and seg["text"].strip():
                date = (seg.get("created_at", "") or "")[:10]
                parts.append(f"📅 {date}: _{seg['text'][:200]}_")

        return ChatResponse(text="\n".join(parts))

    # ---- Audio conversion ----

    def _convert_to_wav(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> Optional[bytes]:
        """Convert any audio format to 16kHz mono WAV using ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix=self._ext_for_mime(mime_type), delete=False) as inp:
                inp.write(audio_bytes)
                inp_path = inp.name

            out_path = inp_path + ".wav"
            result = subprocess.run(
                ["ffmpeg", "-i", inp_path, "-ar", "16000", "-ac", "1", "-f", "wav", out_path, "-y"],
                capture_output=True, timeout=60,
            )

            if result.returncode == 0:
                wav_bytes = Path(out_path).read_bytes()
                Path(inp_path).unlink(missing_ok=True)
                Path(out_path).unlink(missing_ok=True)
                return wav_bytes
            else:
                logger.error("ffmpeg failed: %s", result.stderr.decode()[:200])
                Path(inp_path).unlink(missing_ok=True)
                return None

        except FileNotFoundError:
            logger.error("ffmpeg not found — install it for audio conversion")
            return None
        except Exception as e:
            logger.error("Audio conversion failed: %s", e)
            return None

    @staticmethod
    def _ext_for_mime(mime_type: str) -> str:
        mapping = {
            "audio/ogg": ".ogg",
            "audio/oga": ".ogg",
            "audio/opus": ".opus",
            "audio/mp4": ".m4a",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "audio/amr": ".amr",
        }
        return mapping.get(mime_type, ".ogg")
