"""FastAPI web UI for DeskVoice.

Provides a dashboard showing live transcripts, tasks, sessions, and token usage.
Uses Server-Sent Events (SSE) for real-time updates from the agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import AgentConfig
from src.storage.database import Database

logger = logging.getLogger(__name__)

# Global event bus for SSE
_event_queues: list[asyncio.Queue] = []


def broadcast_event(event_type: str, data: dict) -> None:
    """Broadcast an event to all connected SSE clients."""
    msg = {"type": event_type, "data": data, "timestamp": datetime.now().isoformat()}
    for q in _event_queues:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


def create_app(config: AgentConfig) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="DeskVoice", docs_url=None, redoc_url=None)

    db = Database(config.storage.db_path)
    db.connect()

    # --- SSE endpoint ---
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
        """Get agent stats if available."""
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
        # Enrollment via API requires pre-recorded audio
        # For now return the speaker list
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

    @app.get("/api/recording/state")
    def recording_state():
        if _agent_ref:
            state = "paused" if _agent_ref.is_paused else "recording"
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
    --orange: #e67e22;
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
  .rec-icon { width: 16px; height: 16px; border-radius: 50%; background: var(--green); transition: all 0.2s; }
  @keyframes rec-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.4); } 50% { box-shadow: 0 0 0 8px rgba(231, 76, 60, 0); } }
  .rec-timer { font-size: 1.3em; font-weight: bold; font-variant-numeric: tabular-nums; min-width: 80px; color: var(--text); }
  .rec-timer.active { color: var(--red); }
  .rec-timer.paused { color: var(--yellow); }
  .rec-label { font-size: 0.75em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }

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
      </div>
      <div class="status">
        <span id="connection" class="live">&#9679; Connected</span>
      </div>
    </div>
  </header>

  <!-- Audio player bar (shown when playing a recording) -->
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

  <!-- Speakers tab -->
  <div class="tab-content" id="tab-speakers">
    <div class="card">
      <h2>Enrolled Speakers</h2>
      <div id="speakers-list"><div class="empty">No speakers enrolled</div></div>
      <div class="speaker-enroll-hint">
        To enroll a new speaker, use the CLI: <code>deskvoice enroll &lt;name&gt;</code><br>
        Or rename an existing speaker by clicking the edit button.
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
  recState: 'recording', // recording, paused
  recStartTime: null,
  timerInterval: null,
  // Player state
  currentRecId: null,
  isPlaying: false,
};

function $(id) { return document.getElementById(id); }

// ---- Tabs ----
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('.tab-content#tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'recordings') loadRecordings();
  if (name === 'speakers') loadSpeakers();
}

// ---- Recording controls ----
function toggleRecording() {
  if (state.recState === 'recording') {
    fetch('/api/recording/pause', { method: 'POST' }).then(r => r.json()).then(d => {
      if (d.ok) setRecState('paused');
    });
  } else {
    fetch('/api/recording/resume', { method: 'POST' }).then(r => r.json()).then(d => {
      if (d.ok) setRecState('recording');
    });
  }
}

function setRecState(s) {
  state.recState = s;
  const btn = $('rec-btn');
  const timer = $('rec-timer');
  const label = $('rec-label');
  btn.className = 'rec-btn ' + s;
  timer.className = 'rec-timer ' + (s === 'recording' ? 'active' : s);
  label.textContent = s === 'recording' ? 'Recording' : 'Paused';
  if (s === 'recording') {
    state.recStartTime = Date.now();
    startTimer();
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
  const elapsed = Math.floor((Date.now() - state.recStartTime) / 1000);
  const hrs = Math.floor(elapsed / 3600);
  const mins = Math.floor((elapsed % 3600) / 60);
  const secs = elapsed % 60;
  const timer = $('rec-timer');
  if (hrs > 0) {
    timer.textContent = hrs + ':' + String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  } else {
    timer.textContent = String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }
}

// Init recording state
fetch('/api/recording/state').then(r => r.json()).then(d => {
  if (d.state) {
    setRecState(d.state);
    if (d.start_time && d.state === 'recording') {
      state.recStartTime = new Date(d.start_time).getTime();
      startTimer();
    }
  }
});

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
  const feed = $('transcript-feed');
  if (feed.querySelector('.empty')) feed.innerHTML = '';
  const entry = document.createElement('div');
  entry.className = 'transcript-entry';
  const time = new Date(data.timestamp).toLocaleTimeString();
  entry.innerHTML = '<div class="time">' + time + '</div>'
    + '<div class="text">' + escHtml(data.transcript) + '</div>'
    + (data.summary ? '<div class="summary">' + escHtml(data.summary) + '</div>' : '');
  feed.appendChild(entry);
  feed.scrollTop = feed.scrollHeight;
}

// ---- Tasks ----
function addTask(task) {
  const list = $('tasks-list');
  if (list.querySelector('.empty')) list.innerHTML = '';
  state.taskCount++;
  $('tasks-count').textContent = '(' + state.taskCount + ')';

  const item = document.createElement('div');
  item.className = 'task-item';
  item.id = 'task-' + (task.id || Date.now());
  const p = task.priority || 'medium';
  item.innerHTML = '<span class="priority ' + p + '">' + p.toUpperCase() + '</span>'
    + '<div class="desc">' + escHtml(task.description)
    + (task.assignee ? '<div class="assignee">-> ' + escHtml(task.assignee) + '</div>' : '')
    + (task.due_hint ? '<div class="due">Due: ' + escHtml(task.due_hint) + '</div>' : '')
    + '</div>'
    + (task.id ? '<button onclick="markDone(' + task.id + ')">Done</button>' : '');
  list.prepend(item);
}

function addDecisionOrQuestion(type, text) {
  const feed = $('decisions-feed');
  if (feed.querySelector('.empty')) feed.innerHTML = '';
  const el = document.createElement('div');
  el.className = type;
  el.textContent = text;
  feed.appendChild(el);
}

// ---- Tags ----
function addTag(tag) {
  if (state.tags.has(tag)) return;
  state.tags.add(tag);
  $('tags-count').textContent = '(' + state.tags.size + ')';
  const list = $('tags-list');
  if (list.querySelector('.empty')) list.innerHTML = '';
  const el = document.createElement('span');
  el.className = 'tag';
  el.textContent = '#' + tag;
  list.appendChild(el);
}

// ---- Streaming ----
function handleStreamToken(token) {
  const feed = $('transcript-feed');
  if (feed.querySelector('.empty')) feed.innerHTML = '';
  let streaming = document.getElementById('streaming-entry');
  if (!streaming) {
    streaming = document.createElement('div');
    streaming.id = 'streaming-entry';
    streaming.className = 'transcript-entry';
    streaming.innerHTML = '<div class="time">' + new Date().toLocaleTimeString() + '</div><div class="text" id="streaming-text"></div>';
    feed.appendChild(streaming);
  }
  const textEl = document.getElementById('streaming-text');
  if (textEl) textEl.textContent += token;
  feed.scrollTop = feed.scrollHeight;
}

function handleStreamComplete() {
  const el = document.getElementById('streaming-entry');
  if (el) el.removeAttribute('id');
  const textEl = document.getElementById('streaming-text');
  if (textEl) textEl.removeAttribute('id');
}

function markDone(taskId) {
  fetch('/api/tasks/' + taskId + '/done', { method: 'POST' })
    .then(() => {
      const el = document.getElementById('task-' + taskId);
      if (el) el.style.opacity = '0.4';
    });
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

// ---- Recordings ----
function loadRecordings() {
  fetch('/api/recordings').then(r => r.json()).then(recs => {
    const list = $('recordings-list');
    list.innerHTML = '';
    if (!recs || recs.length === 0) {
      list.innerHTML = '<div class="empty">No recordings yet</div>';
      $('rec-count').textContent = '';
      return;
    }
    $('rec-count').textContent = '(' + recs.length + ')';
    recs.forEach(rec => {
      const item = document.createElement('div');
      item.className = 'recording-item';
      const time = new Date(rec.created_at).toLocaleString();
      const dur = formatDuration(rec.duration_seconds);
      const transcript = rec.transcript ? rec.transcript.substring(0, 120) : 'No transcript';
      const isPlaying = state.currentRecId === rec.id && state.isPlaying;
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
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return m > 0 ? m + 'm ' + s + 's' : s + 's';
}

function playRecording(recId) {
  const audio = $('audio-el');
  const bar = $('audio-player-bar');

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
  const audio = $('audio-el');
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
  const btn = $('player-play-btn');
  btn.innerHTML = state.isPlaying ? '&#9646;&#9646;' : '&#9654;';
  // Update recording list play buttons too
  document.querySelectorAll('.rec-play-btn').forEach(b => {
    b.classList.remove('playing');
    b.innerHTML = '&#9654;';
  });
  if (state.isPlaying && state.currentRecId) {
    // Highlight the playing recording button
    loadRecordings();
  }
}

function seekPlayer(e) {
  const audio = $('audio-el');
  if (!audio.duration) return;
  const rect = $('player-progress').getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audio.currentTime = pct * audio.duration;
}

function closePlayer() {
  const audio = $('audio-el');
  audio.pause();
  audio.src = '';
  state.isPlaying = false;
  state.currentRecId = null;
  $('audio-player-bar').classList.remove('visible');
}

// Audio element events
const audioEl = $('audio-el');
audioEl.addEventListener('timeupdate', function() {
  const cur = audioEl.currentTime;
  const dur = audioEl.duration || 0;
  $('player-current-time').textContent = formatTime(cur);
  $('player-total-time').textContent = formatTime(dur);
  const pct = dur > 0 ? (cur / dur) * 100 : 0;
  $('player-progress-fill').style.width = pct + '%';
});
audioEl.addEventListener('ended', function() {
  state.isPlaying = false;
  updatePlayerUI();
});

function formatTime(secs) {
  if (!secs || isNaN(secs)) return '0:00';
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m + ':' + String(s).padStart(2, '0');
}

// ---- Speakers ----
function loadSpeakers() {
  fetch('/api/speakers').then(r => r.json()).then(speakers => {
    const list = $('speakers-list');
    list.innerHTML = '';
    if (!speakers || speakers.length === 0) {
      list.innerHTML = '<div class="empty">No speakers enrolled</div>';
      return;
    }
    speakers.forEach(sp => {
      const item = document.createElement('div');
      item.className = 'speaker-item';
      const initials = sp.name.split(' ').map(w => w[0]).join('').toUpperCase().substring(0, 2);
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
  const nameEl = document.getElementById('speaker-name-' + id);
  if (!nameEl) return;
  const current = nameEl.textContent;
  const newName = prompt('Rename speaker:', current);
  if (newName && newName !== current) {
    fetch('/api/speakers/' + id, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName }),
    }).then(r => r.json()).then(() => {
      nameEl.textContent = newName;
    });
  }
}

// ---- Load initial data ----
fetch('/api/tasks').then(r => r.json()).then(tasks => {
  tasks.forEach(t => addTask(t));
});
fetch('/api/tags').then(r => r.json()).then(tags => {
  tags.forEach(t => addTag(t.tag));
});
fetch('/api/stats').then(r => r.json()).then(s => {
  if (s.total_input_tokens) {
    state.tokensIn = s.total_input_tokens;
    state.tokensOut = s.total_output_tokens;
    state.tokensTotal = s.total_tokens;
    state.chunks = s.chunks_processed || 0;
    state.speechSecs = s.speech_seconds || 0;
    updateTokens();
  }
});

// ---- SSE connection ----
function connectSSE() {
  const es = new EventSource('/api/events');
  es.onmessage = function(e) {
    const msg = JSON.parse(e.data);
    const d = msg.data;
    switch (msg.type) {
      case 'insight':
        if (d.transcript) addTranscript({transcript: d.transcript, summary: d.summary, timestamp: msg.timestamp});
        (d.tasks || []).forEach(t => addTask(t));
        (d.decisions || []).forEach(dec => addDecisionOrQuestion('decision', dec));
        (d.questions || []).forEach(q => addDecisionOrQuestion('question', q));
        (d.hashtags || []).forEach(h => addTag(h.tag || h));
        if (d.token_usage) {
          state.tokensIn += d.token_usage.input_tokens || 0;
          state.tokensOut += d.token_usage.output_tokens || 0;
          state.tokensTotal += d.token_usage.total_tokens || 0;
        }
        state.chunks++;
        state.speechSecs += d.duration_seconds || 0;
        updateTokens();
        break;
      case 'session_started':
        break;
      case 'session_ended':
        break;
      case 'stream_token':
        handleStreamToken(d.token || '');
        break;
      case 'stream_complete':
        handleStreamComplete();
        break;
      case 'task_completed':
        var el = document.getElementById('task-' + d.task_id);
        if (el) el.style.opacity = '0.4';
        break;
      case 'recording_state':
        setRecState(d.state);
        break;
    }
  };
  es.onerror = function() {
    $('connection').textContent = '\\u25cf Reconnecting...';
    $('connection').style.color = 'var(--red)';
    es.close();
    setTimeout(connectSSE, 3000);
  };
  es.onopen = function() {
    $('connection').textContent = '\\u25cf Connected';
    $('connection').style.color = 'var(--green)';
  };
}
connectSSE();
</script>
</body>
</html>
"""
