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
from fastapi.responses import HTMLResponse, StreamingResponse
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
        from src.web import _agent_stats
        return _agent_stats.copy() if _agent_stats else {}

    # --- HTML frontend ---
    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD_HTML

    return app


# Store agent stats for the API
_agent_stats: dict = {}


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
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text); }
  .container { max-width: 1200px; margin: 0 auto; padding: 16px; }
  header { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--accent); margin-bottom: 16px; }
  header h1 { font-size: 1.4em; color: var(--green); }
  .status { display: flex; gap: 16px; font-size: 0.85em; color: var(--text-dim); }
  .status .live { color: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .grid { display: grid; grid-template-columns: 1fr 350px; gap: 16px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--surface); border: 1px solid var(--accent); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  .card h2 { font-size: 0.9em; color: var(--green); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .transcript-entry { padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .transcript-entry .time { color: var(--text-dim); font-size: 0.75em; }
  .transcript-entry .text { margin-top: 4px; line-height: 1.5; }
  .transcript-entry .summary { color: var(--blue); font-size: 0.85em; margin-top: 4px; font-style: italic; }
  .task-item { display: flex; align-items: flex-start; gap: 8px; padding: 8px 0; border-bottom: 1px solid #ffffff10; }
  .task-item .priority { font-size: 0.7em; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
  .task-item .priority.high { background: var(--red); color: white; }
  .task-item .priority.medium { background: var(--yellow); color: black; }
  .task-item .priority.low { background: var(--accent); color: var(--text); }
  .task-item .desc { flex: 1; }
  .task-item .assignee { color: var(--yellow); font-size: 0.85em; }
  .task-item .due { color: var(--text-dim); font-size: 0.8em; }
  .task-item button { background: var(--green); border: none; color: black; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75em; }
  .tag { display: inline-block; background: var(--accent); color: var(--blue); padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin: 2px; }
  .token-bar { display: flex; gap: 16px; font-size: 0.85em; }
  .token-bar span { color: var(--text-dim); }
  .token-bar .val { color: var(--green); font-weight: bold; }
  .decision { color: var(--green); padding: 4px 0; font-size: 0.9em; }
  .decision::before { content: "\\2713 "; }
  .question { color: var(--yellow); padding: 4px 0; font-size: 0.9em; }
  .question::before { content: "? "; }
  #transcript-feed { max-height: 500px; overflow-y: auto; }
  #tasks-list { max-height: 400px; overflow-y: auto; }
  .empty { color: var(--text-dim); font-style: italic; font-size: 0.85em; padding: 20px 0; text-align: center; }
  .session-info { font-size: 0.8em; color: var(--text-dim); margin-bottom: 8px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>DeskVoice</h1>
    <div class="status">
      <span id="connection" class="live">&#9679; Connected</span>
      <span id="provider-info"></span>
    </div>
  </header>

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
        <h2>Tasks</h2>
        <div id="tasks-list"><div class="empty">No tasks yet</div></div>
      </div>
      <div class="card">
        <h2>Tags</h2>
        <div id="tags-list"><div class="empty">No tags yet</div></div>
      </div>
    </div>
  </div>
</div>

<script>
const state = {
  tokensIn: 0, tokensOut: 0, tokensTotal: 0,
  chunks: 0, speechSecs: 0,
  transcripts: [], tasks: [], tags: new Set(), decisions: [], questions: []
};

function $(id) { return document.getElementById(id); }

function updateTokens() {
  $('tokens-in').textContent = state.tokensIn.toLocaleString();
  $('tokens-out').textContent = state.tokensOut.toLocaleString();
  $('tokens-total').textContent = state.tokensTotal.toLocaleString();
  $('chunks-count').textContent = state.chunks;
  $('speech-secs').textContent = Math.round(state.speechSecs) + 's';
}

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

function addTask(task) {
  const list = $('tasks-list');
  if (list.querySelector('.empty')) list.innerHTML = '';

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

function addTag(tag) {
  if (state.tags.has(tag)) return;
  state.tags.add(tag);
  const list = $('tags-list');
  if (list.querySelector('.empty')) list.innerHTML = '';
  const el = document.createElement('span');
  el.className = 'tag';
  el.textContent = '#' + tag;
  list.appendChild(el);
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

// Load existing data
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

// SSE connection
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
      case 'task_completed':
        const el = document.getElementById('task-' + d.task_id);
        if (el) el.style.opacity = '0.4';
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
