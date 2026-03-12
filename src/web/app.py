"""FastAPI web UI for DeskVoice.

Provides a dashboard showing live transcripts, tasks, sessions, and token usage.
Uses WebSocket for real-time bidirectional updates and SSE as fallback.
Features voice print management and stop-recording processing flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import AgentConfig
from src.storage.database import Database

logger = logging.getLogger(__name__)

# Global event bus for SSE and WebSocket
_event_queues: list[asyncio.Queue] = []
_ws_clients: list[WebSocket] = []
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_event(event_type: str, data: dict) -> None:
    """Broadcast an event to all connected SSE and WebSocket clients.

    Thread-safe: can be called from any thread (agent loop, background threads, etc.).
    """
    msg = {"type": event_type, "data": data, "timestamp": datetime.now().isoformat()}

    # SSE clients — put_nowait is thread-safe on asyncio.Queue
    for q in _event_queues:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    # WebSocket clients — must schedule on the asyncio event loop
    if not _ws_clients:
        return

    msg_str = json.dumps(msg)

    async def _send_to_all():
        disconnected = []
        for ws in _ws_clients:
            try:
                await ws.send_text(msg_str)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            try:
                _ws_clients.remove(ws)
            except ValueError:
                pass

    loop = _event_loop
    if loop is not None and loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(_send_to_all(), loop)
        except Exception:
            pass


def create_app(config: AgentConfig) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="DeskVoice", docs_url=None, redoc_url=None)

    db = Database(config.storage.db_path)
    db.connect()

    @app.on_event("startup")
    async def _capture_event_loop():
        global _event_loop
        _event_loop = asyncio.get_running_loop()

    # --- WebSocket endpoint ---
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(_ws_clients))
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                # Handle client messages
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "request_state":
                    # Send current state to the new client
                    state = _build_current_state(db)
                    await websocket.send_text(json.dumps({
                        "type": "state_sync",
                        "data": state,
                        "timestamp": datetime.now().isoformat(),
                    }))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("WebSocket error: %s", e)
        finally:
            try:
                _ws_clients.remove(websocket)
            except ValueError:
                pass
            logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))

    def _build_current_state(db: Database) -> dict:
        """Build the current state for a newly connected client."""
        rec_state = "stopped"
        start_time = None
        if _agent_ref:
            rec_state = "paused" if _agent_ref.is_paused else "recording"
            start_time = _agent_ref.recording_start_time
        return {
            "recording_state": rec_state,
            "recording_start_time": start_time.isoformat() if start_time else None,
            "stats": _agent_stats.copy() if _agent_stats else {},
            "tasks": db.list_tasks(pending_only=True),
            "tags": db.list_hashtags(),
            "voice_prints": db.list_voice_prints(),
            "speakers": db.list_speakers(),
        }

    # --- SSE endpoint (fallback) ---
    @app.get("/api/events")
    async def events(request: Request):
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        _event_queues.append(queue)

        async def stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(msg)}\n\n"
                    except asyncio.TimeoutError:
                        yield f": keepalive\n\n"
            finally:
                _event_queues.remove(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # --- REST endpoints ---
    @app.get("/api/sessions")
    def get_sessions(limit: int = 20):
        return db.list_sessions(limit=limit)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: int):
        return db.get_session(session_id)

    @app.get("/api/tasks")
    def get_tasks(all: bool = False):
        return db.list_tasks(pending_only=not all)

    @app.post("/api/tasks/{task_id}/done")
    def complete_task(task_id: int):
        db.complete_task(task_id)
        broadcast_event("task_completed", {"task_id": task_id})
        return {"ok": True}

    @app.get("/api/tags")
    def get_tags():
        return db.list_hashtags()

    @app.get("/api/search")
    def search(q: str):
        return db.search_transcripts(q)

    @app.get("/api/stats")
    def get_stats():
        return _agent_stats.copy() if _agent_stats else {}

    # --- Speaker endpoints ---
    @app.get("/api/speakers")
    def get_speakers():
        return db.list_speakers()

    @app.post("/api/speakers")
    async def enroll_speaker(request: Request):
        body = await request.json()
        name = body.get("name", "")
        if not name:
            return {"error": "name is required"}
        return {"message": "Use CLI 'deskvoice enroll' for voice enrollment"}

    @app.put("/api/speakers/{speaker_id}")
    async def update_speaker(speaker_id: int, request: Request):
        body = await request.json()
        name = body.get("name", "")
        if name:
            db.update_speaker_name(speaker_id, name)
        return {"ok": True}

    # --- Recording control endpoints ---
    @app.post("/api/recording/pause")
    def pause_recording():
        if _agent_ref:
            _agent_ref.pause()
            return {"ok": True, "state": "paused"}
        return {"ok": False, "error": "Agent not available"}

    @app.post("/api/recording/resume")
    def resume_recording():
        if _agent_ref:
            _agent_ref.resume()
            return {"ok": True, "state": "recording"}
        return {"ok": False, "error": "Agent not available"}

    @app.post("/api/recording/stop")
    def stop_recording():
        """Stop recording and process all buffered audio in the backend.

        This triggers:
        1. Flush VAD buffer and process remaining speech
        2. Voice print analysis on the session's recordings
        3. Speaker segmentation and tagging
        """
        if not _agent_ref:
            return {"ok": False, "error": "Agent not available"}

        # Run processing in a background thread to not block the API
        result = {"ok": True, "state": "processing"}

        def _process():
            try:
                proc_result = _agent_ref.stop_and_process()
                broadcast_event("processing_complete", proc_result)
            except Exception as e:
                logger.error("Stop processing failed: %s", e)
                broadcast_event("processing_error", {"error": str(e)})

        thread = threading.Thread(target=_process, daemon=True)
        thread.start()
        return result

    @app.get("/api/recording/state")
    def recording_state():
        if _agent_ref:
            if _agent_ref.is_paused:
                state = "paused"
            elif _agent_ref.is_recording:
                state = "recording"
            else:
                state = "stopped"
            start_time = _agent_ref.recording_start_time
            return {
                "state": state,
                "start_time": start_time.isoformat() if start_time else None,
            }
        return {"state": "stopped", "start_time": None}

    # --- Recordings (playback) endpoints ---
    @app.get("/api/recordings")
    def get_recordings(limit: int = 50):
        return db.list_recordings(limit=limit)

    @app.get("/api/recordings/{recording_id}")
    def get_recording(recording_id: int):
        return db.get_recording(recording_id)

    @app.get("/api/recordings/{recording_id}/audio")
    def get_recording_audio(recording_id: int):
        rec = db.get_recording(recording_id)
        if not rec:
            return {"error": "Recording not found"}
        audio_path = config.storage.audio_dir / rec["filename"]
        if not audio_path.exists():
            return {"error": "Audio file not found"}
        return FileResponse(audio_path, media_type="audio/wav", filename=rec["filename"])

    @app.get("/api/recordings/{recording_id}/segments")
    def get_recording_segments(recording_id: int):
        """Get speaker-attributed segments for a recording."""
        return db.get_recording_segments(recording_id)

    # --- Voice Print endpoints ---
    @app.get("/api/voice-prints")
    def get_voice_prints():
        """List all detected voice prints with segment info."""
        vps = db.list_voice_prints()
        # Enrich with segment counts and text snippets
        for vp in vps:
            # Get segments associated with this voice print
            segments = db.conn.execute(
                """SELECT rs.text, rs.start_seconds, rs.end_seconds, rs.recording_id, r.filename
                   FROM recording_segments rs
                   LEFT JOIN recordings r ON rs.recording_id = r.id
                   WHERE rs.voice_print_id = ?
                   ORDER BY rs.created_at DESC LIMIT 10""",
                (vp["id"],),
            ).fetchall()
            vp["segments"] = [dict(s) for s in segments]
            vp["has_audio"] = bool(vp.get("audio_sample_file"))
        return vps

    @app.get("/api/voice-prints/{vp_id}/audio")
    def get_voice_print_audio(vp_id: int):
        """Serve the audio sample for a voice print."""
        vps = db.list_voice_prints()
        vp = next((v for v in vps if v["id"] == vp_id), None)
        if not vp or not vp.get("audio_sample_file"):
            return {"error": "No audio sample available for this voice print"}
        audio_path = config.storage.audio_dir / vp["audio_sample_file"]
        if not audio_path.exists():
            return {"error": "Audio file not found"}
        return FileResponse(audio_path, media_type="audio/wav", filename=vp["audio_sample_file"])

    @app.put("/api/voice-prints/{vp_id}")
    async def update_voice_print(vp_id: int, request: Request):
        """Rename a voice print label."""
        body = await request.json()
        label = body.get("label", "")
        if label:
            db.rename_voice_print(vp_id, label)
            broadcast_event("voice_print_updated", {"id": vp_id, "label": label})
        return {"ok": True}

    @app.post("/api/voice-prints/{vp_id}/map")
    async def map_voice_print(vp_id: int, request: Request):
        """Map a voice print to a known speaker."""
        body = await request.json()
        speaker_id = body.get("speaker_id")
        if speaker_id is None:
            return {"error": "speaker_id is required"}
        db.map_voice_print_to_speaker(vp_id, speaker_id)
        broadcast_event("voice_print_mapped", {
            "voice_print_id": vp_id,
            "speaker_id": speaker_id,
        })
        return {"ok": True}

    @app.post("/api/voice-prints/merge")
    async def merge_voice_prints(request: Request):
        """Merge two voice prints (e.g., same person detected twice)."""
        body = await request.json()
        keep_id = body.get("keep_id")
        merge_id = body.get("merge_id")
        if not keep_id or not merge_id:
            return {"error": "keep_id and merge_id are required"}
        db.merge_voice_prints(keep_id, merge_id)
        broadcast_event("voice_prints_merged", {
            "keep_id": keep_id,
            "merge_id": merge_id,
        })
        return {"ok": True}

    @app.delete("/api/voice-prints/{vp_id}")
    def delete_voice_print(vp_id: int):
        """Delete a voice print."""
        db.delete_voice_print(vp_id)
        broadcast_event("voice_print_deleted", {"id": vp_id})
        return {"ok": True}

    # --- Task management endpoints ---
    @app.put("/api/tasks/{task_id}")
    async def update_task(task_id: int, request: Request):
        body = await request.json()
        db.update_task(
            task_id,
            description=body.get("description"),
            assignee=body.get("assignee"),
            priority=body.get("priority"),
            due_hint=body.get("due_hint"),
            reminder_at=body.get("reminder_at"),
        )
        broadcast_event("task_updated", {"task_id": task_id, **body})
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/remind")
    async def set_reminder(task_id: int, request: Request):
        body = await request.json()
        reminder_at = body.get("reminder_at", "")
        db.update_task(task_id, reminder_at=reminder_at)
        return {"ok": True, "reminder_at": reminder_at}

    # --- Daily Briefing ---
    @app.get("/api/briefing")
    def get_briefing(date: str = None):
        """Get daily briefing with sessions, tasks, speakers, and stats."""
        return db.get_daily_briefing(date)

    # --- Session Detail ---
    @app.get("/api/sessions/{session_id}/detail")
    def get_session_detail(session_id: int):
        """Get detailed session with recordings, segments, participants."""
        return db.get_session_detail(session_id)

    # --- Meeting Mode ---
    @app.post("/api/meeting/start")
    async def start_meeting(request: Request):
        """Start a named meeting session with optional participant list."""
        body = await request.json()
        title = body.get("title", "Meeting")
        participants = body.get("participants", [])

        if _agent_ref:
            _agent_ref.resume()
            # Force start a new session with the given title
            if hasattr(_agent_ref, '_start_session'):
                _agent_ref._start_session()
            if hasattr(_agent_ref, '_current_session_id') and _agent_ref._current_session_id:
                db.conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, _agent_ref._current_session_id),
                )
                db.conn.commit()
                broadcast_event("session_started", {
                    "session_id": _agent_ref._current_session_id,
                    "title": title,
                    "participants": participants,
                })
                return {
                    "ok": True,
                    "session_id": _agent_ref._current_session_id,
                    "title": title,
                }
        return {"ok": False, "error": "Agent not available"}

    @app.post("/api/meeting/end")
    def end_meeting():
        """End the current meeting — stop recording and process."""
        if not _agent_ref:
            return {"ok": False, "error": "Agent not available"}
        import threading
        def _process():
            try:
                result = _agent_ref.stop_and_process()
                broadcast_event("processing_complete", result)
            except Exception as e:
                logger.error("Meeting end processing failed: %s", e)
                broadcast_event("processing_error", {"error": str(e)})
        thread = threading.Thread(target=_process, daemon=True)
        thread.start()
        return {"ok": True, "state": "processing"}

    # --- Smart Search ---
    @app.get("/api/search/smart")
    def smart_search(q: str):
        """Search across transcripts, tasks, and decisions with context."""
        results = {
            "transcripts": [],
            "tasks": [],
            "recordings": [],
        }

        # Search transcripts via FTS
        try:
            sessions = db.search_transcripts(q, limit=10)
            for s in sessions:
                # Highlight matching context
                transcript = s.get("transcript", "")
                q_lower = q.lower()
                idx = transcript.lower().find(q_lower)
                if idx >= 0:
                    start = max(0, idx - 80)
                    end = min(len(transcript), idx + len(q) + 80)
                    context = ("..." if start > 0 else "") + transcript[start:end] + ("..." if end < len(transcript) else "")
                else:
                    context = transcript[:200]
                s["context"] = context
            results["transcripts"] = sessions
        except Exception:
            pass

        # Search tasks
        try:
            all_tasks = db.list_tasks(pending_only=False)
            q_lower = q.lower()
            matching_tasks = [
                t for t in all_tasks
                if q_lower in (t.get("description", "") or "").lower()
                or q_lower in (t.get("assignee", "") or "").lower()
            ]
            results["tasks"] = matching_tasks[:10]
        except Exception:
            pass

        # Search recordings by transcript content
        try:
            recs = db.list_recordings(limit=200)
            q_lower = q.lower()
            matching_recs = []
            for r in recs:
                transcript = r.get("transcript", "") or ""
                if q_lower in transcript.lower():
                    idx = transcript.lower().find(q_lower)
                    start = max(0, idx - 60)
                    end = min(len(transcript), idx + len(q) + 60)
                    r["context"] = ("..." if start > 0 else "") + transcript[start:end] + ("..." if end < len(transcript) else "")
                    matching_recs.append(r)
            results["recordings"] = matching_recs[:10]
        except Exception:
            pass

        return results

    # --- HTML frontend ---
    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD_HTML

    return app


# Store agent stats and reference for the API
_agent_stats: dict = {}
_agent_ref = None  # Reference to the DeskVoiceAgent instance


def set_agent_ref(agent) -> None:
    """Set the agent reference for recording control."""
    global _agent_ref
    _agent_ref = agent


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0f0f0f">
<title>DeskVoice</title>
<style>
  :root {
    --bg: #0f0f0f; --surface: #1a1a2e; --surface2: #16213e;
    --accent: #0f3460; --text: #e0e0e0; --text-dim: #888;
    --green: #4ecca3; --yellow: #f0c929; --red: #e74c3c; --blue: #3498db;
    --orange: #e67e22; --purple: #9b59b6;
    --radius: 10px; --safe-bottom: env(safe-area-inset-bottom, 0px);
  }
  * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); overflow-x: hidden; -webkit-font-smoothing: antialiased; }
  .container { max-width: 1000px; margin: 0 auto; padding: 12px; padding-bottom: calc(80px + var(--safe-bottom)); }
  header { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--accent); margin-bottom: 12px; position: sticky; top: 0; background: var(--bg); z-index: 100; }
  header h1 { font-size: 1.1em; color: var(--green); font-weight: 700; }
  .header-right { display: flex; align-items: center; gap: 8px; }
  .status { font-size: 0.7em; color: var(--text-dim); }
  .status .live { color: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Recording controls - compact for mobile */
  .rec-controls { display: flex; align-items: center; gap: 6px; }
  .rec-btn { width: 36px; height: 36px; border-radius: 50%; border: 2px solid var(--accent); background: var(--surface); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
  .rec-btn:active { transform: scale(0.95); }
  .rec-btn.recording { border-color: var(--red); animation: rec-pulse 1.5s infinite; }
  .rec-btn.recording .rec-icon { background: var(--red); }
  .rec-btn.paused { border-color: var(--yellow); }
  .rec-btn.paused .rec-icon { background: var(--yellow); border-radius: 2px; width: 12px; height: 12px; }
  .rec-btn.stopped { border-color: var(--text-dim); }
  .rec-btn.stopped .rec-icon { background: var(--green); }
  .rec-btn.processing { border-color: var(--orange); animation: rec-pulse 1s infinite; }
  .rec-btn.processing .rec-icon { background: var(--orange); border-radius: 2px; width: 12px; height: 12px; }
  .rec-icon { width: 14px; height: 14px; border-radius: 50%; background: var(--green); transition: all 0.2s; }
  @keyframes rec-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.4); } 50% { box-shadow: 0 0 0 8px rgba(231, 76, 60, 0); } }
  .rec-timer { font-size: 1em; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }
  .rec-timer.active { color: var(--red); }
  .rec-timer.paused { color: var(--yellow); }
  .rec-timer.processing { color: var(--orange); }
  .rec-label { display: none; }
  .stop-btn { padding: 5px 10px; border-radius: var(--radius); border: 1px solid var(--red); background: transparent; color: var(--red); cursor: pointer; font-family: inherit; font-size: 0.7em; font-weight: 700; transition: all 0.2s; }
  .stop-btn:active { background: var(--red); color: white; }
  .stop-btn:disabled { opacity: 0.3; }

  /* Bottom nav bar - mobile-first */
  .bottom-nav { position: fixed; bottom: 0; left: 0; right: 0; background: var(--surface); border-top: 1px solid var(--accent); display: flex; justify-content: space-around; padding: 6px 0 calc(6px + var(--safe-bottom)); z-index: 200; }
  .nav-btn { display: flex; flex-direction: column; align-items: center; gap: 2px; background: none; border: none; color: var(--text-dim); cursor: pointer; padding: 4px 8px; font-family: inherit; font-size: 0.6em; transition: color 0.2s; -webkit-tap-highlight-color: transparent; }
  .nav-btn:active, .nav-btn.active { color: var(--green); }
  .nav-btn .nav-icon { font-size: 1.6em; }
  .nav-btn .nav-badge { background: var(--red); color: white; border-radius: 8px; padding: 0 5px; font-size: 0.8em; min-width: 14px; text-align: center; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* Meeting start button */
  .meeting-btn { display: flex; align-items: center; gap: 8px; width: 100%; padding: 14px 16px; border-radius: var(--radius); border: 1px dashed var(--green); background: transparent; color: var(--green); cursor: pointer; font-family: inherit; font-size: 0.9em; font-weight: 600; margin-bottom: 12px; transition: all 0.2s; }
  .meeting-btn:active { background: var(--green); color: black; }
  .meeting-btn .meeting-icon { font-size: 1.3em; }

  .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
  @media (min-width: 768px) { .grid { grid-template-columns: 1fr 340px; } }
  .card { background: var(--surface); border: 1px solid var(--accent); border-radius: var(--radius); padding: 12px; margin-bottom: 10px; }
  .card h2 { font-size: 0.8em; color: var(--green); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; display: flex; justify-content: space-between; align-items: center; }
  .card h2 .count { color: var(--text-dim); font-size: 0.9em; }
  .transcript-entry { padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .transcript-entry .time { color: var(--text-dim); font-size: 0.75em; }
  .transcript-entry .speaker-tag { display: inline-block; background: var(--accent); color: var(--blue); padding: 1px 6px; border-radius: 4px; font-size: 0.75em; margin-left: 6px; }
  .transcript-entry .text { margin-top: 4px; line-height: 1.5; }
  .transcript-entry .summary { color: var(--blue); font-size: 0.85em; margin-top: 4px; font-style: italic; }
  .task-item { display: flex; align-items: flex-start; gap: 8px; padding: 10px 0; border-bottom: 1px solid #ffffff10; }
  .task-item .priority { font-size: 0.65em; padding: 3px 8px; border-radius: 4px; font-weight: 700; flex-shrink: 0; text-transform: uppercase; }
  .task-item .priority.high { background: var(--red); color: white; }
  .task-item .priority.medium { background: var(--yellow); color: black; }
  .task-item .priority.low { background: var(--accent); color: var(--text); }
  .task-item .desc { flex: 1; font-size: 0.85em; line-height: 1.4; }
  .task-item .assignee { color: var(--yellow); font-size: 0.8em; }
  .task-item .due { color: var(--text-dim); font-size: 0.75em; }
  .task-item .task-actions { display: flex; gap: 4px; flex-shrink: 0; }
  .task-item button { background: var(--green); border: none; color: black; padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 0.7em; font-weight: 600; min-height: 28px; }
  .task-item button:active { opacity: 0.7; }
  .task-item .edit-btn { background: var(--accent); color: var(--text-dim); }
  .tag { display: inline-block; background: var(--accent); color: var(--blue); padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin: 3px; cursor: default; }
  .tag:hover { background: var(--surface2); }
  .token-bar { display: flex; gap: 16px; font-size: 0.85em; flex-wrap: wrap; }
  .token-bar span { color: var(--text-dim); }
  .token-bar .val { color: var(--green); font-weight: bold; }
  .decision { color: var(--green); padding: 4px 0; font-size: 0.9em; }
  .decision::before { content: "\\2713 "; }
  .question { color: var(--yellow); padding: 4px 0; font-size: 0.9em; }
  .question::before { content: "? "; }
  #transcript-feed { max-height: 500px; overflow-y: auto; }
  #tasks-list { max-height: 400px; overflow-y: auto; }
  .empty { color: var(--text-dim); font-style: italic; font-size: 0.85em; padding: 20px 0; text-align: center; }

  /* Live streaming indicator */
  .streaming-indicator { display: none; color: var(--orange); font-size: 0.75em; animation: pulse 1s infinite; margin-bottom: 4px; }
  .streaming-indicator.active { display: block; }

  /* Recordings list */
  .recording-item { display: flex; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .recording-item .rec-play-btn { width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--green); background: transparent; color: var(--green); cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .recording-item .rec-play-btn:active { background: var(--green); color: black; }
  .recording-item .rec-info { flex: 1; min-width: 0; font-size: 0.8em; }

  /* Audio player bar */
  .audio-player-bar { display: none; background: var(--surface2); border: 1px solid var(--accent); border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; align-items: center; gap: 12px; }
  .audio-player-bar.visible { display: flex; }
  .audio-player-bar .player-btn { width: 28px; height: 28px; border-radius: 50%; border: 1px solid var(--green); background: transparent; color: var(--green); cursor: pointer; display: flex; align-items: center; justify-content: center; }
  .audio-player-bar .player-btn:hover { background: var(--green); color: black; }
  .audio-player-bar .player-info { flex: 1; font-size: 0.85em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .audio-player-bar .player-time { font-size: 0.8em; color: var(--text-dim); font-variant-numeric: tabular-nums; }
  .audio-player-bar .player-progress { flex: 2; height: 4px; background: var(--accent); border-radius: 2px; cursor: pointer; position: relative; }
  .audio-player-bar .player-progress-fill { height: 100%; background: var(--green); border-radius: 2px; width: 0; transition: width 0.1s; }
  .audio-player-bar .player-close { background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 1.1em; padding: 4px; }
  .audio-player-bar .player-close:hover { color: var(--text); }

  /* Voice prints section */
  .vp-item { background: var(--surface2); border: 1px solid var(--accent); border-radius: 8px; padding: 14px; margin-bottom: 10px; }
  .vp-item-header { display: flex; align-items: center; gap: 10px; }
  .vp-item .vp-avatar { width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1em; font-weight: bold; flex-shrink: 0; }
  .vp-item .vp-info { flex: 1; }
  .vp-item .vp-label { font-size: 1em; font-weight: bold; cursor: pointer; }
  .vp-item .vp-label:hover { color: var(--green); text-decoration: underline; }
  .vp-item .vp-meta { font-size: 0.75em; color: var(--text-dim); margin-top: 2px; }
  .vp-item .vp-mapped { font-size: 0.75em; color: var(--green); margin-top: 2px; }
  .vp-item .vp-play-btn { width: 36px; height: 36px; border-radius: 50%; border: 2px solid var(--green); background: transparent; color: var(--green); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 1em; flex-shrink: 0; transition: all 0.2s; }
  .vp-item .vp-play-btn:hover { background: var(--green); color: black; }
  .vp-item .vp-play-btn.playing { border-color: var(--red); color: var(--red); }
  .vp-item .vp-play-btn.playing:hover { background: var(--red); color: white; }
  .vp-item .vp-play-btn.no-audio { border-color: var(--text-dim); color: var(--text-dim); opacity: 0.5; cursor: not-allowed; }
  .vp-item .vp-segments { margin-top: 8px; padding-top: 8px; border-top: 1px solid #ffffff10; }
  .vp-item .vp-segment-text { font-size: 0.8em; color: var(--text-dim); line-height: 1.4; margin-bottom: 4px; padding-left: 8px; border-left: 2px solid; }
  .vp-item .vp-actions { display: flex; gap: 6px; margin-top: 10px; }
  .vp-item .vp-actions button { background: var(--accent); border: none; color: var(--text-dim); padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 0.75em; font-family: inherit; transition: all 0.2s; }
  .vp-item .vp-actions button:hover { color: var(--text); background: var(--surface); }
  .vp-item .vp-actions .map-btn { color: var(--green); border: 1px solid var(--green); background: transparent; }
  .vp-item .vp-actions .map-btn:hover { background: var(--green); color: black; }
  .vp-item .vp-actions .delete-btn { color: var(--red); }
  .vp-item .vp-actions .delete-btn:hover { background: var(--red); color: white; }
  #voice-prints-list { max-height: none; overflow-y: auto; }

  /* Speakers / voiceprint section */
  .speaker-item { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .speaker-item .speaker-avatar { width: 32px; height: 32px; border-radius: 50%; background: var(--accent); color: var(--blue); display: flex; align-items: center; justify-content: center; font-size: 0.85em; font-weight: bold; flex-shrink: 0; }
  .speaker-item .speaker-info { flex: 1; }
  .speaker-item .speaker-name { font-size: 0.9em; }
  .speaker-item .speaker-date { font-size: 0.75em; color: var(--text-dim); }
  .speaker-item .speaker-actions { display: flex; gap: 4px; }
  .speaker-item .speaker-actions button { background: var(--accent); border: none; color: var(--text-dim); padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75em; font-family: inherit; }
  .speaker-item .speaker-actions button:hover { color: var(--text); background: var(--surface2); }
  #speakers-list { max-height: 300px; overflow-y: auto; }
  .speaker-enroll-hint { font-size: 0.8em; color: var(--text-dim); margin-top: 8px; padding: 8px; background: var(--surface2); border-radius: 6px; }

  /* Processing overlay */
  .processing-banner { display: none; background: var(--surface2); border: 1px solid var(--orange); border-radius: var(--radius); padding: 10px 14px; margin-bottom: 10px; align-items: center; gap: 10px; }
  .processing-banner.visible { display: flex; }
  .processing-banner .spinner { width: 18px; height: 18px; border: 2px solid var(--orange); border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; flex-shrink: 0; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .processing-banner .processing-text { flex: 1; color: var(--orange); font-size: 0.85em; }
  .processing-banner .processing-detail { font-size: 0.7em; color: var(--text-dim); }

  /* Briefing card */
  .briefing { background: linear-gradient(135deg, var(--surface2), var(--surface)); border: 1px solid var(--accent); border-radius: var(--radius); padding: 14px; margin-bottom: 12px; }
  .briefing-header { font-size: 0.9em; font-weight: 700; color: var(--green); margin-bottom: 10px; }
  .briefing-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }
  .briefing-stat { text-align: center; padding: 8px; background: var(--bg); border-radius: 8px; }
  .briefing-stat .stat-value { font-size: 1.4em; font-weight: 700; color: var(--green); }
  .briefing-stat .stat-label { font-size: 0.65em; color: var(--text-dim); text-transform: uppercase; margin-top: 2px; }
  .briefing-people { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .briefing-person { display: flex; align-items: center; gap: 6px; background: var(--accent); padding: 4px 10px; border-radius: 16px; font-size: 0.75em; }
  .briefing-person .person-dot { width: 8px; height: 8px; border-radius: 50%; }
  .briefing-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }

  /* Search bar */
  .search-bar { display: flex; gap: 8px; margin-bottom: 12px; }
  .search-input { flex: 1; padding: 10px 14px; border-radius: var(--radius); border: 1px solid var(--accent); background: var(--surface); color: var(--text); font-family: inherit; font-size: 0.85em; outline: none; }
  .search-input:focus { border-color: var(--green); }
  .search-input::placeholder { color: var(--text-dim); }
  .search-results { margin-top: 8px; }
  .search-result { padding: 10px; background: var(--surface); border: 1px solid var(--accent); border-radius: var(--radius); margin-bottom: 6px; }
  .search-result .result-type { font-size: 0.65em; color: var(--blue); text-transform: uppercase; font-weight: 700; }
  .search-result .result-context { font-size: 0.8em; color: var(--text-dim); margin-top: 4px; line-height: 1.4; }
  .search-result mark { background: var(--yellow); color: black; border-radius: 2px; padding: 0 2px; }

  /* Session timeline */
  .session-item { padding: 10px; background: var(--surface2); border: 1px solid var(--accent); border-radius: var(--radius); margin-bottom: 8px; cursor: pointer; transition: border-color 0.2s; }
  .session-item:active { border-color: var(--green); }
  .session-item .session-title { font-size: 0.85em; font-weight: 600; }
  .session-item .session-meta { font-size: 0.7em; color: var(--text-dim); margin-top: 4px; display: flex; gap: 12px; }
  .session-item .session-preview { font-size: 0.75em; color: var(--text-dim); margin-top: 6px; line-height: 1.3; }
  .session-detail-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: var(--bg); z-index: 300; overflow-y: auto; padding: 12px; }
  .session-detail-overlay.visible { display: block; }
  .session-detail-back { display: flex; align-items: center; gap: 8px; color: var(--green); background: none; border: none; cursor: pointer; font-family: inherit; font-size: 0.85em; padding: 8px 0; }
  .session-participants { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
  .session-segment { display: flex; gap: 8px; padding: 8px 0; border-bottom: 1px solid #ffffff08; }
  .session-segment .seg-speaker { font-size: 0.75em; font-weight: 600; min-width: 70px; flex-shrink: 0; }
  .session-segment .seg-text { font-size: 0.8em; color: var(--text-dim); line-height: 1.4; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>DeskVoice</h1>
    <div class="header-right">
      <div class="rec-controls">
        <button class="rec-btn recording" id="rec-btn" onclick="toggleRecording()">
          <div class="rec-icon"></div>
        </button>
        <div class="rec-timer active" id="rec-timer">00:00</div>
        <button class="stop-btn" id="stop-btn" onclick="stopRecording()">STOP</button>
      </div>
      <span id="connection" class="status live">&#9679;</span>
    </div>
  </header>

  <!-- Processing banner -->
  <div class="processing-banner" id="processing-banner">
    <div class="spinner"></div>
    <div>
      <div class="processing-text">Processing recording...</div>
      <div class="processing-detail" id="processing-detail">Analyzing voice prints and transcripts</div>
    </div>
  </div>

  <!-- Audio player bar -->
  <div class="audio-player-bar" id="audio-player-bar">
    <button class="player-btn" id="player-play-btn" onclick="togglePlayer()">&#9654;</button>
    <span class="player-info" id="player-info">-</span>
    <span class="player-time" id="player-current-time">0:00</span>
    <div class="player-progress" id="player-progress" onclick="seekPlayer(event)">
      <div class="player-progress-fill" id="player-progress-fill"></div>
    </div>
    <span class="player-time" id="player-total-time">0:00</span>
    <button class="player-close" onclick="closePlayer()">&#10005;</button>
  </div>

  <!-- Tabs are now bottom nav -->

  <!-- Home / Briefing tab -->
  <div class="tab-content active" id="tab-home">
    <!-- Meeting start button -->
    <button class="meeting-btn" onclick="startMeeting()">
      <span class="meeting-icon">&#127908;</span> Start a Meeting
    </button>

    <!-- Daily briefing card -->
    <div class="briefing" id="briefing-card">
      <div class="briefing-header">Today's Briefing</div>
      <div class="briefing-stats">
        <div class="briefing-stat"><div class="stat-value" id="b-sessions">0</div><div class="stat-label">Sessions</div></div>
        <div class="briefing-stat"><div class="stat-value" id="b-speech">0m</div><div class="stat-label">Speech</div></div>
        <div class="briefing-stat"><div class="stat-value" id="b-tasks">0</div><div class="stat-label">Tasks</div></div>
      </div>
      <div class="briefing-people" id="b-people"></div>
      <div class="briefing-tags" id="b-tags"></div>
    </div>

    <!-- Live transcript -->
    <div class="card">
      <h2>Live Transcript</h2>
      <div class="streaming-indicator" id="streaming-indicator">Transcribing...</div>
      <div id="transcript-feed"><div class="empty">Waiting for speech...</div></div>
    </div>

    <!-- Token bar (compact) -->
    <div class="token-bar" style="padding: 8px; font-size: 0.7em; color: var(--text-dim);">
      <span>In:<span class="val" id="tokens-in">0</span></span>
      <span>Out:<span class="val" id="tokens-out">0</span></span>
      <span>Chunks:<span class="val" id="chunks-count">0</span></span>
    </div>
  </div>

  <!-- Tasks tab -->
  <div class="tab-content" id="tab-tasks">
    <div class="card">
      <h2>Action Items <span class="count" id="tasks-count"></span></h2>
      <div id="tasks-list"><div class="empty">No tasks yet</div></div>
    </div>
    <div class="card">
      <h2>Decisions & Questions</h2>
      <div id="decisions-feed"><div class="empty">None yet</div></div>
    </div>
    <div class="card">
      <h2>Tags <span class="count" id="tags-count"></span></h2>
      <div id="tags-list"><div class="empty">No tags yet</div></div>
    </div>
  </div>

  <!-- Sessions tab -->
  <div class="tab-content" id="tab-sessions">
    <div class="card">
      <h2>Today's Sessions</h2>
      <div id="sessions-list"><div class="empty">No sessions yet</div></div>
    </div>
  </div>

  <!-- Search tab -->
  <div class="tab-content" id="tab-search">
    <div class="search-bar">
      <input class="search-input" id="search-input" type="search" placeholder="Search transcripts, tasks, recordings..." oninput="debounceSearch()">
    </div>
    <div class="search-results" id="search-results"></div>
  </div>

  <!-- People / Voice Prints tab -->
  <div class="tab-content" id="tab-people">
    <div class="card">
      <h2>Detected Voices <span class="count" id="vp-count"></span></h2>
      <div id="voice-prints-list"><div class="empty">No voice prints detected yet. Record and stop to analyze.</div></div>
    </div>
  </div>
</div>

<!-- Session detail overlay -->
<div class="session-detail-overlay" id="session-detail-overlay">
  <button class="session-detail-back" onclick="closeSessionDetail()">&#8592; Back</button>
  <div id="session-detail-content"></div>
</div>

<!-- Bottom navigation bar -->
<div class="bottom-nav">
  <button class="nav-btn active" onclick="switchTab('home')" id="nav-home">
    <span class="nav-icon">&#127968;</span>Home
  </button>
  <button class="nav-btn" onclick="switchTab('tasks')" id="nav-tasks">
    <span class="nav-icon">&#9745;</span>Tasks<span class="nav-badge" id="nav-tasks-badge" style="display:none">0</span>
  </button>
  <button class="nav-btn" onclick="switchTab('sessions')" id="nav-sessions">
    <span class="nav-icon">&#128488;</span>Sessions
  </button>
  <button class="nav-btn" onclick="switchTab('search')" id="nav-search">
    <span class="nav-icon">&#128269;</span>Search
  </button>
  <button class="nav-btn" onclick="switchTab('people')" id="nav-people">
    <span class="nav-icon">&#128101;</span>People
  </button>
</div>

<audio id="audio-el" preload="auto"></audio>

<script>
const state = {
  tokensIn: 0, tokensOut: 0, tokensTotal: 0,
  chunks: 0, speechSecs: 0, taskCount: 0,
  transcripts: [], tasks: [], tags: new Set(), decisions: [], questions: [],
  recState: 'recording',
  recStartTime: null,
  timerInterval: null,
  currentRecId: null,
  isPlaying: false,
  ws: null,
  wsReconnectDelay: 1000,
  voicePrints: [],
  speakers: [],
  searchTimer: null,
  vpPlayingId: null,
  processingTimeout: null,
};

function $(id) { return document.getElementById(id); }

// ---- WebSocket Connection ----
function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(protocol + '//' + location.host + '/ws');

  ws.onopen = function() {
    state.ws = ws;
    state.wsReconnectDelay = 1000;
    $('connection').textContent = '\\u25cf Connected (WS)';
    $('connection').style.color = 'var(--green)';
    $('connection').className = 'live';
    // Request initial state
    ws.send(JSON.stringify({type: 'request_state'}));
  };

  ws.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    handleEvent(msg);
  };

  ws.onclose = function() {
    state.ws = null;
    $('connection').textContent = '\\u25cf Reconnecting...';
    $('connection').style.color = 'var(--red)';
    $('connection').className = '';
    setTimeout(connectWS, state.wsReconnectDelay);
    state.wsReconnectDelay = Math.min(state.wsReconnectDelay * 2, 10000);
  };

  ws.onerror = function() {
    ws.close();
  };
}

function handleEvent(msg) {
  const d = msg.data;
  switch (msg.type) {
    case 'state_sync':
      // Full state sync for new connection
      if (d.recording_state) setRecState(d.recording_state);
      if (d.recording_start_time && d.recording_state === 'recording') {
        state.recStartTime = new Date(d.recording_start_time).getTime();
        startTimer();
      }
      if (d.stats && d.stats.total_input_tokens) {
        state.tokensIn = d.stats.total_input_tokens;
        state.tokensOut = d.stats.total_output_tokens;
        state.tokensTotal = d.stats.total_tokens;
        state.chunks = d.stats.chunks_processed || 0;
        state.speechSecs = d.stats.speech_seconds || 0;
        updateTokens();
      }
      (d.tasks || []).forEach(function(t) { addTask(t); });
      (d.tags || []).forEach(function(t) { addTag(t.tag); });
      if (d.voice_prints) { state.voicePrints = d.voice_prints; }
      if (d.speakers) { state.speakers = d.speakers; }
      break;

    case 'insight':
      if (d.transcript) {
        // Find speaker labels in segments
        var speakers = [];
        if (d.segments) {
          d.segments.forEach(function(seg) { if (seg.speaker) speakers.push(seg.speaker); });
        }
        addTranscript({
          transcript: d.transcript,
          summary: d.summary,
          timestamp: msg.timestamp,
          speakers: [...new Set(speakers)],
        });
      }
      (d.tasks || []).forEach(function(t) { addTask(t); });
      (d.decisions || []).forEach(function(dec) { addDecisionOrQuestion('decision', dec); });
      (d.questions || []).forEach(function(q) { addDecisionOrQuestion('question', q); });
      (d.hashtags || []).forEach(function(h) { addTag(h.tag || h); });
      if (d.token_usage) {
        state.tokensIn += d.token_usage.input_tokens || 0;
        state.tokensOut += d.token_usage.output_tokens || 0;
        state.tokensTotal += d.token_usage.total_tokens || 0;
      }
      state.chunks++;
      state.speechSecs += d.duration_seconds || 0;
      updateTokens();
      break;

    case 'stream_token':
      handleStreamToken(d.token || '');
      break;

    case 'stream_complete':
      handleStreamComplete();
      break;

    case 'session_started':
      break;

    case 'session_ended':
      break;

    case 'task_completed':
      var el = document.getElementById('task-' + d.task_id);
      if (el) el.style.opacity = '0.4';
      break;

    case 'recording_state':
      setRecState(d.state);
      break;

    case 'processing_complete':
      if (state.processingTimeout) { clearTimeout(state.processingTimeout); state.processingTimeout = null; }
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
      $('stop-btn').disabled = false;
      // Always reload voice prints after processing
      loadVoicePrints();
      loadRecordings();
      if (d.speakers_identified && d.speakers_identified.length > 0) {
        var names = d.speakers_identified.map(function(s) { return s.label; }).join(', ');
        $('processing-detail').textContent = 'Done! Detected: ' + names;
      }
      break;

    case 'processing_error':
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
      $('stop-btn').disabled = false;
      break;

    case 'voice_print_progress':
      $('processing-detail').textContent = 'Analyzed recording: ' + d.segments + ' segments, ' + d.clusters + ' speakers detected';
      break;

    case 'voice_print_updated':
    case 'voice_print_mapped':
    case 'voice_prints_merged':
    case 'voice_print_deleted':
      loadVoicePrints();
      break;

    case 'reminder':
      break;

    case 'pong':
      break;
  }
}

// ---- Navigation ----
function switchTab(name) {
  document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
  document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
  var tab = document.getElementById('tab-' + name);
  if (tab) tab.classList.add('active');
  var nav = document.getElementById('nav-' + name);
  if (nav) nav.classList.add('active');
  if (name === 'people') loadVoicePrints();
  if (name === 'sessions') loadSessions();
  if (name === 'search') { var inp = $('search-input'); if (inp) inp.focus(); }
}

// ---- Recording controls ----
function toggleRecording() {
  if (state.recState === 'recording') {
    fetch('/api/recording/pause', { method: 'POST' }).then(function(r) { return r.json(); }).then(function(d) {
      if (d.ok) setRecState('paused');
    });
  } else if (state.recState === 'paused' || state.recState === 'stopped') {
    fetch('/api/recording/resume', { method: 'POST' }).then(function(r) { return r.json(); }).then(function(d) {
      if (d.ok) setRecState('recording');
    });
  }
}

function stopRecording() {
  if (state.recState === 'processing') return;
  setRecState('processing');
  $('processing-banner').classList.add('visible');
  $('processing-detail').textContent = 'Analyzing voice prints and transcripts...';
  $('stop-btn').disabled = true;

  // Safety timeout: if no processing_complete event arrives in 120s, clear the banner
  if (state.processingTimeout) clearTimeout(state.processingTimeout);
  state.processingTimeout = setTimeout(function() {
    if (state.recState === 'processing') {
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
      $('stop-btn').disabled = false;
      // Still try to load voice prints — processing may have finished silently
      loadVoicePrints();
      loadRecordings();
    }
  }, 120000);

  fetch('/api/recording/stop', { method: 'POST' }).then(function(r) { return r.json(); }).then(function(d) {
    if (!d.ok) {
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
      $('stop-btn').disabled = false;
    }
    // Results will come via WebSocket/SSE 'processing_complete' event
  }).catch(function() {
    setRecState('stopped');
    $('processing-banner').classList.remove('visible');
    $('stop-btn').disabled = false;
  });
}

function setRecState(s) {
  state.recState = s;
  var btn = $('rec-btn');
  var timer = $('rec-timer');
  var label = $('rec-label');
  var stopBtn = $('stop-btn');
  btn.className = 'rec-btn ' + s;
  timer.className = 'rec-timer ' + (s === 'recording' ? 'active' : s);

  var labels = {recording: 'Recording', paused: 'Paused', stopped: 'Stopped', processing: 'Processing...'};
  label.textContent = labels[s] || s;
  stopBtn.disabled = (s === 'processing' || s === 'stopped');

  if (s === 'recording') {
    state.recStartTime = state.recStartTime || Date.now();
    startTimer();
  } else if (s === 'stopped') {
    stopTimer();
    timer.textContent = '00:00';
    state.recStartTime = null;
    stopBtn.disabled = true;
  } else {
    stopTimer();
  }
}

function startTimer() {
  if (state.timerInterval) clearInterval(state.timerInterval);
  state.timerInterval = setInterval(updateTimer, 1000);
  updateTimer();
}

function stopTimer() {
  // Keep timer display frozen at current value
}

function updateTimer() {
  if (!state.recStartTime || state.recState !== 'recording') return;
  var elapsed = Math.floor((Date.now() - state.recStartTime) / 1000);
  var hrs = Math.floor(elapsed / 3600);
  var mins = Math.floor((elapsed % 3600) / 60);
  var secs = elapsed % 60;
  var timer = $('rec-timer');
  if (hrs > 0) {
    timer.textContent = hrs + ':' + String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  } else {
    timer.textContent = String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }
}

// ---- Token stats ----
function updateTokens() {
  $('tokens-in').textContent = state.tokensIn.toLocaleString();
  $('tokens-out').textContent = state.tokensOut.toLocaleString();
  $('tokens-total').textContent = state.tokensTotal.toLocaleString();
  $('chunks-count').textContent = state.chunks;
  $('speech-secs').textContent = Math.round(state.speechSecs) + 's';
}

// ---- Transcript ----
function addTranscript(data) {
  var feed = $('transcript-feed');
  if (feed.querySelector('.empty')) feed.innerHTML = '';
  var entry = document.createElement('div');
  entry.className = 'transcript-entry';
  var time = new Date(data.timestamp).toLocaleTimeString();
  var speakerTags = '';
  if (data.speakers && data.speakers.length > 0) {
    speakerTags = data.speakers.map(function(s) {
      return '<span class="speaker-tag">' + escHtml(s) + '</span>';
    }).join('');
  }
  entry.innerHTML = '<div class="time">' + time + speakerTags + '</div>'
    + '<div class="text">' + escHtml(data.transcript) + '</div>'
    + (data.summary ? '<div class="summary">' + escHtml(data.summary) + '</div>' : '');
  feed.appendChild(entry);
  feed.scrollTop = feed.scrollHeight;
}

// ---- Tasks ----
function addTask(task) {
  var list = $('tasks-list');
  if (list.querySelector('.empty')) list.innerHTML = '';
  state.tasks.push(task);
  state.taskCount++;
  $('tasks-count').textContent = '(' + state.taskCount + ')';
  // Update badge
  if ($('nav-tasks-badge')) { $('nav-tasks-badge').style.display = 'inline'; $('nav-tasks-badge').textContent = state.taskCount; }

  var item = document.createElement('div');
  item.className = 'task-item';
  item.id = 'task-' + (task.id || Date.now());
  var p = task.priority || 'medium';
  item.innerHTML = '<span class="priority ' + p + '">' + p.toUpperCase() + '</span>'
    + '<div class="desc">' + escHtml(task.description)
    + (task.assignee ? '<div class="assignee">-> ' + escHtml(task.assignee) + '</div>' : '')
    + (task.due_hint ? '<div class="due">Due: ' + escHtml(task.due_hint) + '</div>' : '')
    + '</div>'
    + '<div class="task-actions">'
    + (task.id ? '<button class="edit-btn" onclick="editTask(' + task.id + ')">Edit</button>' : '')
    + (task.id ? '<button onclick="markDone(' + task.id + ')">Done</button>' : '')
    + '</div>';
  list.prepend(item);
}

function addDecisionOrQuestion(type, text) {
  var feed = $('decisions-feed');
  if (feed.querySelector('.empty')) feed.innerHTML = '';
  var el = document.createElement('div');
  el.className = type;
  el.textContent = text;
  feed.appendChild(el);
}

// ---- Tags ----
function addTag(tag) {
  if (state.tags.has(tag)) return;
  state.tags.add(tag);
  $('tags-count').textContent = '(' + state.tags.size + ')';
  var list = $('tags-list');
  if (list.querySelector('.empty')) list.innerHTML = '';
  var el = document.createElement('span');
  el.className = 'tag';
  el.textContent = '#' + tag;
  list.appendChild(el);
}

// ---- Streaming ----
function handleStreamToken(token) {
  var feed = $('transcript-feed');
  if (feed.querySelector('.empty')) feed.innerHTML = '';
  $('streaming-indicator').classList.add('active');
  var streaming = document.getElementById('streaming-entry');
  if (!streaming) {
    streaming = document.createElement('div');
    streaming.id = 'streaming-entry';
    streaming.className = 'transcript-entry';
    streaming.innerHTML = '<div class="time">' + new Date().toLocaleTimeString() + '</div><div class="text" id="streaming-text"></div>';
    feed.appendChild(streaming);
  }
  var textEl = document.getElementById('streaming-text');
  if (textEl) textEl.textContent += token;
  feed.scrollTop = feed.scrollHeight;
}

function handleStreamComplete() {
  $('streaming-indicator').classList.remove('active');
  var el = document.getElementById('streaming-entry');
  if (el) el.removeAttribute('id');
  var textEl = document.getElementById('streaming-text');
  if (textEl) textEl.removeAttribute('id');
}

function markDone(taskId) {
  fetch('/api/tasks/' + taskId + '/done', { method: 'POST' })
    .then(function() {
      var el = document.getElementById('task-' + taskId);
      if (el) el.style.opacity = '0.4';
    });
}

function escHtml(s) {
  var d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

// ---- Recordings ----
function loadRecordings() {
  // Recordings are now shown inside session detail view
  // This function refreshes the briefing and sessions list
  loadBriefing();
  if (document.getElementById('tab-sessions').classList.contains('active')) {
    loadSessions();
  }
}

function formatDuration(secs) {
  if (!secs) return '0s';
  var m = Math.floor(secs / 60);
  var s = Math.round(secs % 60);
  return m > 0 ? m + 'm ' + s + 's' : s + 's';
}

function playRecording(recId) {
  var audio = $('audio-el');
  var bar = $('audio-player-bar');

  if (state.currentRecId === recId && state.isPlaying) {
    audio.pause();
    state.isPlaying = false;
    updatePlayerUI();
    return;
  }

  state.currentRecId = recId;
  audio.src = '/api/recordings/' + recId + '/audio';
  audio.play();
  state.isPlaying = true;
  bar.classList.add('visible');
  $('player-info').textContent = 'Recording #' + recId;
  updatePlayerUI();
}

function togglePlayer() {
  var audio = $('audio-el');
  if (state.isPlaying) {
    audio.pause();
    state.isPlaying = false;
  } else {
    audio.play();
    state.isPlaying = true;
  }
  updatePlayerUI();
}

function updatePlayerUI() {
  var btn = $('player-play-btn');
  btn.innerHTML = state.isPlaying ? '&#9646;&#9646;' : '&#9654;';
  document.querySelectorAll('.rec-play-btn').forEach(function(b) {
    b.classList.remove('playing');
    b.innerHTML = '&#9654;';
  });
}

function seekPlayer(e) {
  var audio = $('audio-el');
  if (!audio.duration) return;
  var rect = $('player-progress').getBoundingClientRect();
  var pct = (e.clientX - rect.left) / rect.width;
  audio.currentTime = pct * audio.duration;
}

function closePlayer() {
  var audio = $('audio-el');
  audio.pause();
  audio.src = '';
  state.isPlaying = false;
  state.currentRecId = null;
  $('audio-player-bar').classList.remove('visible');
}

// Audio element events
var audioEl = $('audio-el');
audioEl.addEventListener('timeupdate', function() {
  var cur = audioEl.currentTime;
  var dur = audioEl.duration || 0;
  $('player-current-time').textContent = formatTime(cur);
  $('player-total-time').textContent = formatTime(dur);
  var pct = dur > 0 ? (cur / dur) * 100 : 0;
  $('player-progress-fill').style.width = pct + '%';
});
audioEl.addEventListener('ended', function() {
  state.isPlaying = false;
  updatePlayerUI();
});

function formatTime(secs) {
  if (!secs || isNaN(secs)) return '0:00';
  var m = Math.floor(secs / 60);
  var s = Math.floor(secs % 60);
  return m + ':' + String(s).padStart(2, '0');
}

// ---- Voice Prints ----
var vpColors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#f39c12', '#d35400'];
state.vpPlayingId = null;

function loadVoicePrints() {
  fetch('/api/voice-prints').then(function(r) { return r.json(); }).then(function(vps) {
    var list = $('voice-prints-list');
    list.innerHTML = '';
    state.voicePrints = vps;
    if (!vps || vps.length === 0) {
      list.innerHTML = '<div class="empty">No voice prints detected yet. Record audio and click STOP to analyze.</div>';
      $('vp-count').textContent = '';
      return;
    }
    $('vp-count').textContent = '(' + vps.length + ')';
    vps.forEach(function(vp, idx) {
      var item = document.createElement('div');
      item.className = 'vp-item';
      var color = vpColors[idx % vpColors.length];
      var initial = (vp.label || 'V')[0].toUpperCase();
      var mappedText = vp.mapped_speaker_name
        ? '<span style="color:var(--green)">\\u2714 Mapped to: ' + escHtml(vp.mapped_speaker_name) + '</span>'
        : '<span style="color:var(--yellow)">\\u26A0 Not identified — click "Identify" to name this person</span>';

      // Play button
      var hasAudio = vp.has_audio;
      var playBtnClass = hasAudio ? 'vp-play-btn' : 'vp-play-btn no-audio';
      var playBtn = '<button class="' + playBtnClass + '" id="vp-play-' + vp.id + '" '
        + (hasAudio ? 'onclick="playVoicePrint(' + vp.id + ')" title="Listen to this voice"' : 'title="No audio sample available"')
        + '>&#9654;</button>';

      // Segment text snippets
      var segmentsHtml = '';
      if (vp.segments && vp.segments.length > 0) {
        segmentsHtml = '<div class="vp-segments">';
        var shown = 0;
        vp.segments.forEach(function(seg) {
          if (shown >= 3) return;
          if (seg.text && seg.text.trim()) {
            segmentsHtml += '<div class="vp-segment-text" style="border-color:' + color + ';">'
              + '&ldquo;' + escHtml(seg.text.substring(0, 150)) + (seg.text.length > 150 ? '...' : '') + '&rdquo;</div>';
            shown++;
          }
        });
        if (shown === 0) {
          segmentsHtml = '';
        } else {
          segmentsHtml += '</div>';
        }
      }

      item.innerHTML = '<div class="vp-item-header">'
        + playBtn
        + '<div class="vp-avatar" style="background:' + color + ';color:white;">' + initial + '</div>'
        + '<div class="vp-info">'
        + '<div class="vp-label" onclick="renameVoicePrint(' + vp.id + ')" title="Click to rename">' + escHtml(vp.label || 'Voice ' + vp.id) + '</div>'
        + '<div class="vp-meta">Samples: ' + vp.sample_count + ' | Created: ' + new Date(vp.created_at).toLocaleDateString() + '</div>'
        + '<div class="vp-mapped">' + mappedText + '</div>'
        + '</div>'
        + '</div>'
        + segmentsHtml
        + '<div class="vp-actions">'
        + '<button class="map-btn" onclick="identifyVoicePrint(' + vp.id + ')">Identify Person</button>'
        + '<button onclick="renameVoicePrint(' + vp.id + ')">Rename</button>'
        + '<button class="delete-btn" onclick="deleteVoicePrint(' + vp.id + ')">Delete</button>'
        + '</div>';
      list.appendChild(item);
    });
  });
}

function playVoicePrint(vpId) {
  var audio = $('audio-el');
  var bar = $('audio-player-bar');
  var btn = document.getElementById('vp-play-' + vpId);

  // If already playing this voice print, stop
  if (state.vpPlayingId === vpId && state.isPlaying) {
    audio.pause();
    state.isPlaying = false;
    state.vpPlayingId = null;
    if (btn) { btn.classList.remove('playing'); btn.innerHTML = '&#9654;'; }
    bar.classList.remove('visible');
    return;
  }

  // Reset any other playing vp buttons
  document.querySelectorAll('.vp-play-btn.playing').forEach(function(b) {
    b.classList.remove('playing');
    b.innerHTML = '&#9654;';
  });

  state.vpPlayingId = vpId;
  state.currentRecId = null;
  audio.src = '/api/voice-prints/' + vpId + '/audio';
  audio.play();
  state.isPlaying = true;

  if (btn) { btn.classList.add('playing'); btn.innerHTML = '&#9632;'; }

  var vp = state.voicePrints.find(function(v) { return v.id === vpId; });
  bar.classList.add('visible');
  $('player-info').textContent = 'Voice: ' + (vp ? vp.label : 'Voice ' + vpId);

  audio.onended = function() {
    state.isPlaying = false;
    state.vpPlayingId = null;
    if (btn) { btn.classList.remove('playing'); btn.innerHTML = '&#9654;'; }
    bar.classList.remove('visible');
  };
}

function identifyVoicePrint(vpId) {
  var vp = state.voicePrints.find(function(v) { return v.id === vpId; });
  var currentLabel = vp ? vp.label : '';

  // First, offer to play the audio if available
  var msg = 'Who is this person?';
  if (vp && vp.has_audio) {
    msg = 'Listen to the voice sample, then enter the person\\'s name.\\n\\n(Click the play button next to the voice print to listen)\\n\\nWho is this person?';
  }

  // Check if we have enrolled speakers to map to
  fetch('/api/speakers').then(function(r) { return r.json(); }).then(function(speakers) {
    var name;
    if (speakers && speakers.length > 0) {
      var options = speakers.map(function(s) { return s.id + ': ' + s.name; }).join('\\n');
      var choice = prompt(msg + '\\n\\nExisting speakers:\\n' + options + '\\n\\nEnter speaker ID to map, or type a new name:');
      if (!choice) return;

      var speakerId = parseInt(choice);
      if (!isNaN(speakerId) && speakers.some(function(s) { return s.id === speakerId; })) {
        // Map to existing speaker
        fetch('/api/voice-prints/' + vpId + '/map', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speaker_id: speakerId }),
        }).then(function() { loadVoicePrints(); });
        return;
      }
      name = choice;
    } else {
      name = prompt(msg);
    }

    if (name && name.trim()) {
      // Rename the voice print to the person's name
      fetch('/api/voice-prints/' + vpId, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: name.trim() }),
      }).then(function() { loadVoicePrints(); });
    }
  });
}

function renameVoicePrint(id) {
  var vp = state.voicePrints.find(function(v) { return v.id === id; });
  var current = vp ? vp.label : '';
  var newLabel = prompt('Rename voice print:', current);
  if (newLabel && newLabel !== current) {
    fetch('/api/voice-prints/' + id, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: newLabel }),
    }).then(function() { loadVoicePrints(); });
  }
}

// mapVoicePrint replaced by identifyVoicePrint above

function deleteVoicePrint(id) {
  if (confirm('Delete this voice print?')) {
    fetch('/api/voice-prints/' + id, { method: 'DELETE' })
      .then(function() { loadVoicePrints(); });
  }
}

// ---- Meeting Mode ----
function startMeeting() {
  var title = prompt('Meeting name (e.g., "Weekly Standup", "1:1 with Alice"):');
  if (!title) return;
  fetch('/api/meeting/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title }),
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      setRecState('recording');
      state.recStartTime = Date.now();
      startTimer();
      switchTab('home');
    }
  });
}

// ---- Daily Briefing ----
function loadBriefing() {
  fetch('/api/briefing').then(function(r) { return r.json(); }).then(function(b) {
    $('b-sessions').textContent = b.session_count || 0;
    $('b-speech').textContent = Math.round((b.total_speech_seconds || 0) / 60) + 'm';
    $('b-tasks').textContent = b.pending_tasks || 0;

    // Update task badge
    if (b.pending_tasks > 0) {
      $('nav-tasks-badge').style.display = 'inline';
      $('nav-tasks-badge').textContent = b.pending_tasks;
    }

    // People
    var people = $('b-people');
    people.innerHTML = '';
    var colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c'];
    (b.speakers || []).forEach(function(s, i) {
      var el = document.createElement('div');
      el.className = 'briefing-person';
      var c = colors[i % colors.length];
      el.innerHTML = '<span class="person-dot" style="background:' + c + '"></span>'
        + escHtml(s.speaker_name || 'Unknown')
        + ' <span style="color:var(--text-dim)">' + Math.round(s.total_seconds / 60) + 'm</span>';
      people.appendChild(el);
    });

    // Tags
    var tags = $('b-tags');
    tags.innerHTML = '';
    (b.tags || []).slice(0, 10).forEach(function(t) {
      var el = document.createElement('span');
      el.className = 'tag';
      el.textContent = '#' + t.tag;
      el.style.fontSize = '0.7em';
      tags.appendChild(el);
    });
  });
}

// ---- Sessions ----
function loadSessions() {
  fetch('/api/sessions?limit=20').then(function(r) { return r.json(); }).then(function(sessions) {
    var list = $('sessions-list');
    list.innerHTML = '';
    if (!sessions || sessions.length === 0) {
      list.innerHTML = '<div class="empty">No sessions yet</div>';
      return;
    }
    sessions.forEach(function(s) {
      var item = document.createElement('div');
      item.className = 'session-item';
      item.onclick = function() { openSessionDetail(s.id); };
      var time = new Date(s.started_at).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
      var dur = Math.round((s.total_speech_seconds || 0) / 60);
      var preview = (s.transcript || '').substring(0, 100);
      item.innerHTML = '<div class="session-title">' + escHtml(s.title || 'Session #' + s.id) + '</div>'
        + '<div class="session-meta"><span>' + time + '</span><span>' + dur + ' min speech</span><span>' + escHtml(s.status || '') + '</span></div>'
        + (preview ? '<div class="session-preview">' + escHtml(preview) + '...</div>' : '');
      list.appendChild(item);
    });
  });
}

function openSessionDetail(sessionId) {
  $('session-detail-overlay').classList.add('visible');
  $('session-detail-content').innerHTML = '<div class="empty">Loading...</div>';
  fetch('/api/sessions/' + sessionId + '/detail').then(function(r) { return r.json(); }).then(function(s) {
    if (!s) { $('session-detail-content').innerHTML = '<div class="empty">Session not found</div>'; return; }
    var html = '<h2 style="color:var(--green);margin:8px 0;">' + escHtml(s.title || 'Session #' + s.id) + '</h2>';
    html += '<div style="font-size:0.75em;color:var(--text-dim);margin-bottom:12px;">'
      + new Date(s.started_at).toLocaleString() + ' | '
      + Math.round((s.total_speech_seconds || 0) / 60) + ' min speech</div>';

    // Participants
    var parts = s.participants || {};
    if (Object.keys(parts).length > 0) {
      html += '<div class="session-participants">';
      var colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22'];
      var ci = 0;
      for (var name in parts) {
        var p = parts[name];
        html += '<div class="briefing-person"><span class="person-dot" style="background:' + colors[ci % colors.length] + '"></span>'
          + escHtml(name) + ' (' + Math.round(p.total_seconds / 60) + 'm)</div>';
        ci++;
      }
      html += '</div>';
    }

    // Tasks
    if (s.tasks && s.tasks.length > 0) {
      html += '<div class="card" style="margin-top:12px"><h2>Tasks</h2>';
      s.tasks.forEach(function(t) {
        html += '<div class="task-item"><span class="priority ' + (t.priority || 'medium') + '">' + (t.priority || 'M').charAt(0).toUpperCase() + '</span>'
          + '<div class="desc">' + escHtml(t.description) + '</div></div>';
      });
      html += '</div>';
    }

    // Tags
    if (s.tags && s.tags.length > 0) {
      html += '<div style="margin:8px 0;">';
      s.tags.forEach(function(t) {
        html += '<span class="tag">#' + escHtml(t.tag) + '</span>';
      });
      html += '</div>';
    }

    // Summary
    if (s.summary) {
      html += '<div class="card"><h2>Summary</h2><div style="font-size:0.85em;line-height:1.5;color:var(--text-dim);">' + escHtml(s.summary) + '</div></div>';
    }

    // Segments (transcript with speaker attribution)
    var segs = s.segments || [];
    if (segs.length > 0) {
      html += '<div class="card"><h2>Transcript</h2>';
      var spkColors = {};
      var spkCI = 0;
      segs.forEach(function(seg) {
        var spk = seg.resolved_speaker || seg.speaker_label || 'Unknown';
        if (!spkColors[spk]) { spkColors[spk] = colors[spkCI % colors.length]; spkCI++; }
        html += '<div class="session-segment">'
          + '<div class="seg-speaker" style="color:' + spkColors[spk] + '">' + escHtml(spk) + '</div>'
          + '<div class="seg-text">' + escHtml(seg.text || '') + '</div>'
          + '</div>';
      });
      html += '</div>';
    } else if (s.transcript) {
      html += '<div class="card"><h2>Transcript</h2><div style="font-size:0.8em;line-height:1.5;white-space:pre-wrap;">' + escHtml(s.transcript) + '</div></div>';
    }

    // Recordings with playback
    if (s.recordings && s.recordings.length > 0) {
      html += '<div class="card"><h2>Recordings</h2>';
      s.recordings.forEach(function(rec) {
        html += '<div class="recording-item">'
          + '<button class="rec-play-btn" onclick="playRecording(' + rec.id + ')">&#9654;</button>'
          + '<div class="rec-info"><div class="rec-time">' + formatDuration(rec.duration_seconds) + '</div></div>'
          + '</div>';
      });
      html += '</div>';
    }

    $('session-detail-content').innerHTML = html;
  });
}

function closeSessionDetail() {
  $('session-detail-overlay').classList.remove('visible');
}

// ---- Smart Search ----
function debounceSearch() {
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(doSearch, 400);
}

function doSearch() {
  var q = $('search-input').value.trim();
  var results = $('search-results');
  if (q.length < 2) { results.innerHTML = ''; return; }

  fetch('/api/search/smart?q=' + encodeURIComponent(q)).then(function(r) { return r.json(); }).then(function(d) {
    results.innerHTML = '';
    var total = (d.transcripts || []).length + (d.tasks || []).length + (d.recordings || []).length;
    if (total === 0) {
      results.innerHTML = '<div class="empty">No results for "' + escHtml(q) + '"</div>';
      return;
    }

    // Tasks
    (d.tasks || []).forEach(function(t) {
      var el = document.createElement('div');
      el.className = 'search-result';
      el.innerHTML = '<div class="result-type">Task</div>'
        + '<div style="font-size:0.85em;margin-top:4px;">' + highlightText(t.description || '', q) + '</div>'
        + (t.assignee ? '<div style="font-size:0.75em;color:var(--yellow);">-> ' + escHtml(t.assignee) + '</div>' : '');
      results.appendChild(el);
    });

    // Transcripts
    (d.transcripts || []).forEach(function(s) {
      var el = document.createElement('div');
      el.className = 'search-result';
      el.style.cursor = 'pointer';
      el.onclick = function() { openSessionDetail(s.id); };
      el.innerHTML = '<div class="result-type">Session: ' + escHtml(s.title || 'Session #' + s.id) + '</div>'
        + '<div class="result-context">' + highlightText(s.context || '', q) + '</div>';
      results.appendChild(el);
    });

    // Recordings
    (d.recordings || []).forEach(function(r) {
      var el = document.createElement('div');
      el.className = 'search-result';
      el.innerHTML = '<div class="result-type">Recording: ' + escHtml(r.filename || '') + '</div>'
        + '<div class="result-context">' + highlightText(r.context || '', q) + '</div>';
      results.appendChild(el);
    });
  });
}

function highlightText(text, query) {
  if (!query) return escHtml(text);
  var safe = escHtml(text);
  var qEsc = escHtml(query);
  var regex = new RegExp('(' + qEsc.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
  return safe.replace(regex, '<mark>$1</mark>');
}

// ---- Task inline editing ----
function editTask(taskId) {
  var task = state.tasks.find(function(t) { return t.id === taskId; });
  if (!task) return;
  var desc = prompt('Edit task:', task.description);
  if (desc && desc !== task.description) {
    fetch('/api/tasks/' + taskId, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description: desc }),
    }).then(function() {
      task.description = desc;
      var el = document.querySelector('#task-' + taskId + ' .desc');
      if (el) el.firstChild.textContent = desc;
    });
  }
}

// ---- Initialize ----
connectWS();

// Load initial data
fetch('/api/tasks').then(function(r) { return r.json(); }).then(function(tasks) {
  tasks.forEach(function(t) { addTask(t); });
});
fetch('/api/tags').then(function(r) { return r.json(); }).then(function(tags) {
  tags.forEach(function(t) { addTag(t.tag); });
});
fetch('/api/stats').then(function(r) { return r.json(); }).then(function(s) {
  if (s.total_input_tokens) {
    state.tokensIn = s.total_input_tokens;
    state.tokensOut = s.total_output_tokens;
    state.tokensTotal = s.total_tokens;
    state.chunks = s.chunks_processed || 0;
    state.speechSecs = s.speech_seconds || 0;
    updateTokens();
  }
});
fetch('/api/recording/state').then(function(r) { return r.json(); }).then(function(d) {
  if (d.state) {
    setRecState(d.state);
    if (d.start_time && d.state === 'recording') {
      state.recStartTime = new Date(d.start_time).getTime();
      startTimer();
    }
  }
});

// Load daily briefing
loadBriefing();

// Keepalive ping
setInterval(function() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({type: 'ping'}));
  }
}, 30000);

// Refresh briefing every 2 minutes
setInterval(loadBriefing, 120000);
</script>
</body>
</html>
"""
