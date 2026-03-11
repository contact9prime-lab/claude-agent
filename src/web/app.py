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


def broadcast_event(event_type: str, data: dict) -> None:
    """Broadcast an event to all connected SSE and WebSocket clients."""
    msg = {"type": event_type, "data": data, "timestamp": datetime.now().isoformat()}

    # SSE clients
    for q in _event_queues:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    # WebSocket clients
    msg_str = json.dumps(msg)
    disconnected = []
    for ws in _ws_clients:
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.ensure_future, ws.send_text(msg_str)
            )
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        try:
            _ws_clients.remove(ws)
        except ValueError:
            pass


def create_app(config: AgentConfig) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="DeskVoice", docs_url=None, redoc_url=None)

    db = Database(config.storage.db_path)
    db.connect()

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
        """List all detected voice prints."""
        return db.list_voice_prints()

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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeskVoice</title>
<style>
  :root {
    --bg: #0f0f0f; --surface: #1a1a2e; --surface2: #16213e;
    --accent: #0f3460; --text: #e0e0e0; --text-dim: #888;
    --green: #4ecca3; --yellow: #f0c929; --red: #e74c3c; --blue: #3498db;
    --orange: #e67e22; --purple: #9b59b6;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text); }
  .container { max-width: 1400px; margin: 0 auto; padding: 16px; }
  header { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--accent); margin-bottom: 16px; }
  header h1 { font-size: 1.4em; color: var(--green); }
  .header-right { display: flex; align-items: center; gap: 16px; }
  .status { display: flex; gap: 16px; font-size: 0.85em; color: var(--text-dim); }
  .status .live { color: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Recording controls */
  .rec-controls { display: flex; align-items: center; gap: 12px; }
  .rec-btn { width: 44px; height: 44px; border-radius: 50%; border: 2px solid var(--accent); background: var(--surface); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
  .rec-btn:hover { border-color: var(--green); transform: scale(1.05); }
  .rec-btn.recording { border-color: var(--red); animation: rec-pulse 1.5s infinite; }
  .rec-btn.recording .rec-icon { background: var(--red); }
  .rec-btn.paused { border-color: var(--yellow); }
  .rec-btn.paused .rec-icon { background: var(--yellow); border-radius: 2px; width: 14px; height: 14px; }
  .rec-btn.stopped { border-color: var(--text-dim); }
  .rec-btn.stopped .rec-icon { background: var(--green); }
  .rec-btn.processing { border-color: var(--orange); animation: rec-pulse 1s infinite; }
  .rec-btn.processing .rec-icon { background: var(--orange); border-radius: 2px; width: 14px; height: 14px; }
  .rec-icon { width: 16px; height: 16px; border-radius: 50%; background: var(--green); transition: all 0.2s; }
  @keyframes rec-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.4); } 50% { box-shadow: 0 0 0 8px rgba(231, 76, 60, 0); } }
  .rec-timer { font-size: 1.3em; font-weight: bold; font-variant-numeric: tabular-nums; min-width: 80px; color: var(--text); }
  .rec-timer.active { color: var(--red); }
  .rec-timer.paused { color: var(--yellow); }
  .rec-timer.processing { color: var(--orange); }
  .rec-label { font-size: 0.75em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }
  .stop-btn { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--red); background: transparent; color: var(--red); cursor: pointer; font-family: inherit; font-size: 0.8em; font-weight: bold; transition: all 0.2s; }
  .stop-btn:hover { background: var(--red); color: white; }
  .stop-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Tabs */
  .tabs { display: flex; gap: 2px; margin-bottom: 16px; background: var(--surface); border-radius: 8px; padding: 3px; }
  .tab { padding: 8px 16px; border: none; background: transparent; color: var(--text-dim); cursor: pointer; border-radius: 6px; font-family: inherit; font-size: 0.85em; transition: all 0.2s; }
  .tab:hover { color: var(--text); background: var(--surface2); }
  .tab.active { color: var(--green); background: var(--accent); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .grid { display: grid; grid-template-columns: 1fr 380px; gap: 16px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--accent); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  .card h2 { font-size: 0.9em; color: var(--green); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; display: flex; justify-content: space-between; align-items: center; }
  .card h2 .count { color: var(--text-dim); font-size: 0.9em; }
  .transcript-entry { padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .transcript-entry .time { color: var(--text-dim); font-size: 0.75em; }
  .transcript-entry .speaker-tag { display: inline-block; background: var(--accent); color: var(--blue); padding: 1px 6px; border-radius: 4px; font-size: 0.75em; margin-left: 6px; }
  .transcript-entry .text { margin-top: 4px; line-height: 1.5; }
  .transcript-entry .summary { color: var(--blue); font-size: 0.85em; margin-top: 4px; font-style: italic; }
  .task-item { display: flex; align-items: flex-start; gap: 8px; padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .task-item .priority { font-size: 0.7em; padding: 2px 6px; border-radius: 4px; font-weight: bold; flex-shrink: 0; }
  .task-item .priority.high { background: var(--red); color: white; }
  .task-item .priority.medium { background: var(--yellow); color: black; }
  .task-item .priority.low { background: var(--accent); color: var(--text); }
  .task-item .desc { flex: 1; }
  .task-item .assignee { color: var(--yellow); font-size: 0.85em; }
  .task-item .due { color: var(--text-dim); font-size: 0.8em; }
  .task-item button { background: var(--green); border: none; color: black; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75em; flex-shrink: 0; }
  .task-item button:hover { opacity: 0.8; }
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
  .recording-item { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #ffffff10; }
  .recording-item .rec-play-btn { width: 32px; height: 32px; border-radius: 50%; border: 1px solid var(--green); background: transparent; color: var(--green); cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all 0.2s; }
  .recording-item .rec-play-btn:hover { background: var(--green); color: black; }
  .recording-item .rec-play-btn.playing { border-color: var(--red); color: var(--red); }
  .recording-item .rec-play-btn.playing:hover { background: var(--red); color: white; }
  .recording-item .rec-info { flex: 1; min-width: 0; }
  .recording-item .rec-info .rec-time { font-size: 0.75em; color: var(--text-dim); }
  .recording-item .rec-info .rec-transcript { font-size: 0.85em; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .recording-item .rec-info .rec-summary { font-size: 0.8em; color: var(--blue); font-style: italic; }
  .recording-item .rec-info .rec-speakers { font-size: 0.75em; color: var(--purple); margin-top: 2px; }
  .recording-item .rec-duration { color: var(--text-dim); font-size: 0.8em; flex-shrink: 0; }
  #recordings-list { max-height: 500px; overflow-y: auto; }

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
  .vp-item { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #ffffff10; }
  .vp-item .vp-avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8em; font-weight: bold; flex-shrink: 0; }
  .vp-item .vp-info { flex: 1; }
  .vp-item .vp-label { font-size: 0.9em; cursor: pointer; }
  .vp-item .vp-label:hover { color: var(--green); }
  .vp-item .vp-meta { font-size: 0.75em; color: var(--text-dim); }
  .vp-item .vp-mapped { font-size: 0.75em; color: var(--green); }
  .vp-item .vp-actions { display: flex; gap: 4px; }
  .vp-item .vp-actions button { background: var(--accent); border: none; color: var(--text-dim); padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75em; font-family: inherit; }
  .vp-item .vp-actions button:hover { color: var(--text); background: var(--surface2); }
  .vp-item .vp-actions .map-btn { color: var(--green); }
  .vp-item .vp-actions .delete-btn { color: var(--red); }
  #voice-prints-list { max-height: 400px; overflow-y: auto; }

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
  .processing-banner { display: none; background: var(--surface2); border: 1px solid var(--orange); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; align-items: center; gap: 12px; }
  .processing-banner.visible { display: flex; }
  .processing-banner .spinner { width: 20px; height: 20px; border: 2px solid var(--orange); border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .processing-banner .processing-text { flex: 1; color: var(--orange); font-size: 0.9em; }
  .processing-banner .processing-detail { font-size: 0.75em; color: var(--text-dim); }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>DeskVoice</h1>
    <div class="header-right">
      <div class="rec-controls">
        <div class="rec-label" id="rec-label">Recording</div>
        <button class="rec-btn recording" id="rec-btn" onclick="toggleRecording()" title="Pause/Resume recording">
          <div class="rec-icon"></div>
        </button>
        <div class="rec-timer active" id="rec-timer">00:00</div>
        <button class="stop-btn" id="stop-btn" onclick="stopRecording()" title="Stop and process">STOP</button>
      </div>
      <div class="status">
        <span id="connection" class="live">&#9679; Connected</span>
      </div>
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

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab('dashboard')">Dashboard</button>
    <button class="tab" onclick="switchTab('recordings')">Recordings</button>
    <button class="tab" onclick="switchTab('voiceprints')">Voice Prints</button>
    <button class="tab" onclick="switchTab('speakers')">Speakers</button>
  </div>

  <!-- Dashboard tab -->
  <div class="tab-content active" id="tab-dashboard">
    <div class="card">
      <h2>Token Usage</h2>
      <div class="token-bar">
        <span>Input: <span class="val" id="tokens-in">0</span></span>
        <span>Output: <span class="val" id="tokens-out">0</span></span>
        <span>Total: <span class="val" id="tokens-total">0</span></span>
        <span>Chunks: <span class="val" id="chunks-count">0</span></span>
        <span>Speech: <span class="val" id="speech-secs">0s</span></span>
      </div>
    </div>
    <div class="grid">
      <div>
        <div class="card">
          <h2>Live Transcript</h2>
          <div class="streaming-indicator" id="streaming-indicator">Transcribing...</div>
          <div id="transcript-feed"><div class="empty">Waiting for speech...</div></div>
        </div>
        <div class="card">
          <h2>Decisions & Questions</h2>
          <div id="decisions-feed"><div class="empty">None yet</div></div>
        </div>
      </div>
      <div>
        <div class="card">
          <h2>Tasks <span class="count" id="tasks-count"></span></h2>
          <div id="tasks-list"><div class="empty">No tasks yet</div></div>
        </div>
        <div class="card">
          <h2>Tags <span class="count" id="tags-count"></span></h2>
          <div id="tags-list"><div class="empty">No tags yet</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Recordings tab -->
  <div class="tab-content" id="tab-recordings">
    <div class="card">
      <h2>Recordings <span class="count" id="rec-count"></span></h2>
      <div id="recordings-list"><div class="empty">No recordings yet</div></div>
    </div>
  </div>

  <!-- Voice Prints tab -->
  <div class="tab-content" id="tab-voiceprints">
    <div class="card">
      <h2>Detected Voice Prints <span class="count" id="vp-count"></span></h2>
      <div id="voice-prints-list"><div class="empty">No voice prints detected yet. Record audio and stop to analyze.</div></div>
    </div>
  </div>

  <!-- Speakers tab -->
  <div class="tab-content" id="tab-speakers">
    <div class="card">
      <h2>Enrolled Speakers</h2>
      <div id="speakers-list"><div class="empty">No speakers enrolled</div></div>
      <div class="speaker-enroll-hint">
        To enroll a new speaker, use the CLI: <code>deskvoice enroll &lt;name&gt;</code><br>
        Or map a detected voice print to a speaker name in the Voice Prints tab.
      </div>
    </div>
  </div>
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
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
      if (d.voice_prints_detected > 0 || d.segments_processed > 0) {
        loadVoicePrints();
      }
      break;

    case 'processing_error':
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
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

// ---- Tabs ----
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
  document.querySelector('.tab-content#tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'recordings') loadRecordings();
  if (name === 'voiceprints') loadVoicePrints();
  if (name === 'speakers') loadSpeakers();
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

  fetch('/api/recording/stop', { method: 'POST' }).then(function(r) { return r.json(); }).then(function(d) {
    if (!d.ok) {
      setRecState('stopped');
      $('processing-banner').classList.remove('visible');
    }
    // Results will come via WebSocket/SSE 'processing_complete' event
  }).catch(function() {
    setRecState('stopped');
    $('processing-banner').classList.remove('visible');
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
  state.taskCount++;
  $('tasks-count').textContent = '(' + state.taskCount + ')';

  var item = document.createElement('div');
  item.className = 'task-item';
  item.id = 'task-' + (task.id || Date.now());
  var p = task.priority || 'medium';
  item.innerHTML = '<span class="priority ' + p + '">' + p.toUpperCase() + '</span>'
    + '<div class="desc">' + escHtml(task.description)
    + (task.assignee ? '<div class="assignee">-> ' + escHtml(task.assignee) + '</div>' : '')
    + (task.due_hint ? '<div class="due">Due: ' + escHtml(task.due_hint) + '</div>' : '')
    + '</div>'
    + (task.id ? '<button onclick="markDone(' + task.id + ')">Done</button>' : '');
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
  fetch('/api/recordings').then(function(r) { return r.json(); }).then(function(recs) {
    var list = $('recordings-list');
    list.innerHTML = '';
    if (!recs || recs.length === 0) {
      list.innerHTML = '<div class="empty">No recordings yet</div>';
      $('rec-count').textContent = '';
      return;
    }
    $('rec-count').textContent = '(' + recs.length + ')';
    recs.forEach(function(rec) {
      var item = document.createElement('div');
      item.className = 'recording-item';
      var time = new Date(rec.created_at).toLocaleString();
      var dur = formatDuration(rec.duration_seconds);
      var transcript = rec.transcript ? rec.transcript.substring(0, 120) : 'No transcript';
      var isPlaying = state.currentRecId === rec.id && state.isPlaying;
      item.innerHTML = '<button class="rec-play-btn' + (isPlaying ? ' playing' : '') + '" onclick="playRecording(' + rec.id + ')" title="Play">'
        + (isPlaying ? '&#9632;' : '&#9654;') + '</button>'
        + '<div class="rec-info">'
        + '<div class="rec-time">' + time + '</div>'
        + '<div class="rec-transcript">' + escHtml(transcript) + '</div>'
        + (rec.summary ? '<div class="rec-summary">' + escHtml(rec.summary) + '</div>' : '')
        + '</div>'
        + '<span class="rec-duration">' + dur + '</span>';
      list.appendChild(item);
    });
  });
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
        ? 'Mapped to: ' + escHtml(vp.mapped_speaker_name)
        : 'Not mapped to a person';
      item.innerHTML = '<div class="vp-avatar" style="background:' + color + ';color:white;">' + initial + '</div>'
        + '<div class="vp-info">'
        + '<div class="vp-label" onclick="renameVoicePrint(' + vp.id + ')" title="Click to rename">' + escHtml(vp.label || 'Voice ' + vp.id) + '</div>'
        + '<div class="vp-meta">Samples: ' + vp.sample_count + ' | Created: ' + new Date(vp.created_at).toLocaleDateString() + '</div>'
        + '<div class="vp-mapped">' + mappedText + '</div>'
        + '</div>'
        + '<div class="vp-actions">'
        + '<button class="map-btn" onclick="mapVoicePrint(' + vp.id + ')">Map to Person</button>'
        + '<button onclick="renameVoicePrint(' + vp.id + ')">Rename</button>'
        + '<button class="delete-btn" onclick="deleteVoicePrint(' + vp.id + ')">Delete</button>'
        + '</div>';
      list.appendChild(item);
    });
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

function mapVoicePrint(vpId) {
  // Load speakers for mapping
  fetch('/api/speakers').then(function(r) { return r.json(); }).then(function(speakers) {
    if (!speakers || speakers.length === 0) {
      var name = prompt('No enrolled speakers. Enter a name to create a new speaker mapping:');
      if (name) {
        // Just rename the voice print to the person's name
        fetch('/api/voice-prints/' + vpId, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label: name }),
        }).then(function() { loadVoicePrints(); });
      }
      return;
    }
    var options = speakers.map(function(s) { return s.id + ': ' + s.name; }).join('\\n');
    var choice = prompt('Map to which speaker?\\n' + options + '\\n\\nEnter speaker ID:');
    if (choice) {
      var speakerId = parseInt(choice);
      if (!isNaN(speakerId)) {
        fetch('/api/voice-prints/' + vpId + '/map', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speaker_id: speakerId }),
        }).then(function() { loadVoicePrints(); });
      }
    }
  });
}

function deleteVoicePrint(id) {
  if (confirm('Delete this voice print?')) {
    fetch('/api/voice-prints/' + id, { method: 'DELETE' })
      .then(function() { loadVoicePrints(); });
  }
}

// ---- Speakers ----
function loadSpeakers() {
  fetch('/api/speakers').then(function(r) { return r.json(); }).then(function(speakers) {
    var list = $('speakers-list');
    list.innerHTML = '';
    state.speakers = speakers;
    if (!speakers || speakers.length === 0) {
      list.innerHTML = '<div class="empty">No speakers enrolled</div>';
      return;
    }
    speakers.forEach(function(sp) {
      var item = document.createElement('div');
      item.className = 'speaker-item';
      var initials = sp.name.split(' ').map(function(w) { return w[0]; }).join('').toUpperCase().substring(0, 2);
      item.innerHTML = '<div class="speaker-avatar">' + escHtml(initials) + '</div>'
        + '<div class="speaker-info">'
        + '<div class="speaker-name" id="speaker-name-' + sp.id + '">' + escHtml(sp.name) + '</div>'
        + '<div class="speaker-date">Enrolled: ' + new Date(sp.created_at).toLocaleDateString() + '</div>'
        + '</div>'
        + '<div class="speaker-actions">'
        + '<button onclick="renameSpeaker(' + sp.id + ')">Rename</button>'
        + '</div>';
      list.appendChild(item);
    });
  });
}

function renameSpeaker(id) {
  var nameEl = document.getElementById('speaker-name-' + id);
  if (!nameEl) return;
  var current = nameEl.textContent;
  var newName = prompt('Rename speaker:', current);
  if (newName && newName !== current) {
    fetch('/api/speakers/' + id, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName }),
    }).then(function(r) { return r.json(); }).then(function() {
      nameEl.textContent = newName;
    });
  }
}

// ---- Initialize ----
// Try WebSocket first, fall back to SSE
connectWS();

// Also set up SSE as fallback for older browsers
function connectSSE() {
  var es = new EventSource('/api/events');
  es.onmessage = function(e) {
    var msg = JSON.parse(e.data);
    handleEvent(msg);
  };
  es.onerror = function() {
    es.close();
    setTimeout(connectSSE, 3000);
  };
}

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

// Init recording state
fetch('/api/recording/state').then(function(r) { return r.json(); }).then(function(d) {
  if (d.state) {
    setRecState(d.state);
    if (d.start_time && d.state === 'recording') {
      state.recStartTime = new Date(d.start_time).getTime();
      startTimer();
    }
  }
});

// WebSocket keepalive ping
setInterval(function() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({type: 'ping'}));
  }
}, 30000);
</script>
</body>
</html>
"""
