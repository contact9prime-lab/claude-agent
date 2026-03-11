# DeskVoice Feature Plan

## Features (in implementation order)

### Phase 1: Core Fixes & Quick Wins

**1. Skip LLM call when no speech detected**
- In `agent.py:_process_chunk()`, already have classifier — ensure we bail early if `audio_type != SPEECH`
- Also skip if VAD returns a chunk but classifier says noise/music
- Files: `src/agent.py`

**2. Beep notification when sending to LLM**
- Play a short beep sound when a chunk is sent to LLM after 1+ min of recording
- Use `sounddevice` to play a generated sine wave tone (no external file needed)
- Add `src/capture/notifier.py` with `play_beep(freq=880, duration_ms=100)`
- Call from `agent.py` right before `_llm.transcribe_and_extract()`
- Only beep if chunk duration >= 60s
- Files: new `src/capture/notifier.py`, modify `src/agent.py`

### Phase 2: Speaker Recognition & Tagging

**3. Speaker voice profiles**
- Add a `speakers` table: `id, name, voice_embedding BLOB, created_at`
- Use `resemblyzer` or `speechbrain` for speaker embeddings (lightweight)
- On each chunk, extract speaker embeddings, compare against known profiles
- If match > threshold, tag segments with real name instead of "Speaker 1"
- Add CLI: `deskvoice speakers` (list), `deskvoice enroll <name>` (record 10s voice sample)
- Add API: `POST /api/speakers` (enroll), `GET /api/speakers` (list), `PUT /api/speakers/:id` (rename)
- Files: new `src/processing/speaker_id.py`, modify `src/storage/database.py`, `src/models/models.py`, `src/agent.py`, `src/cli.py`

### Phase 3: Terminal UI (like Claude Code)

**4. Replace web UI with terminal TUI using Textual**
- Build a rich terminal UI using `textual` library (Python TUI framework)
- Panels: Live Transcript, Tasks, Tags, Stats, Settings
- Runs in terminal — no browser needed
- Background mode: `deskvoice start` (daemon), `deskvoice attach` (connect to running instance)
- Keep the web API for programmatic access
- Files: new `src/tui/` directory with `app.py`, `screens/`, `widgets/`

**5. Settings panel in TUI**
- Settings screen accessible via keybinding (e.g., `s`)
- Edit: LLM provider, model, API key, audio device, VAD thresholds, session timeout
- Persisted to `~/.deskvoice/config.toml`
- Live reload — changes apply without restart
- Files: new `src/tui/screens/settings.py`, modify `src/config.py`

### Phase 4: Task Management

**6. Enhanced task management**
- Edit task description, assignee, priority, due date
- Set reminders (store reminder_at timestamp)
- Background reminder checker — desktop notification when due
- TUI: inline task editing, `r` to set reminder
- API: `PUT /api/tasks/:id`, `POST /api/tasks/:id/remind`
- Database: add `reminder_at`, `edited_at` columns to tasks table
- Files: modify `src/storage/database.py`, `src/models/models.py`, `src/web/app.py`, new `src/tui/screens/tasks.py`

### Phase 5: Live Transcript

**7. Streaming live transcript**
- Instead of waiting for full chunk → LLM → response, show partial results
- Use Gemini streaming API (`stream_generate_content`)
- As transcript tokens arrive, broadcast to TUI/web immediately
- TUI shows text appearing in real-time
- Files: modify `src/processing/llm_provider.py`, `src/agent.py`

### Phase 6: Packaging

**8. macOS DMG packaging**
- Use `py2app` or `PyInstaller` to create standalone .app
- Wrap in DMG using `create-dmg`
- Menu bar icon (system tray) — always running
- Add `Makefile` or `build.sh` with: `make dmg`
- Launchd plist for auto-start on login
- Files: new `packaging/` directory, `Makefile`, `Info.plist`

## Dependencies to Add

```
textual>=0.50.0          # Terminal UI framework
resemblyzer>=0.1.3       # Speaker embeddings (or speechbrain)
toml>=0.10               # Config file format
plyer>=2.1.0             # Desktop notifications (cross-platform)
py2app>=0.28             # macOS app bundling
```

## Implementation Notes

- Phase 1 is quick (< 1 hour) and fixes immediate issues
- Phase 2 adds real value for meeting use case
- Phase 3 is the biggest change — replaces the interaction model
- Phases 4-6 build on top of Phase 3
- Each phase is independently useful and shippable
