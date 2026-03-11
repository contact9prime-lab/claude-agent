# DeskVoice — Always-On Desk Voice Agent

An intelligent, always-on audio agent that sits on your desk, listens to calls and conversations, and extracts actionable insights — tasks, topics, decisions — using Gemini Flash. Local-first, low bandwidth, runs quietly in the background.

## How it works

```
System Audio (BlackHole) ──→ Silero VAD (local, ~1MB)
                                  │
                            Speech detected?
                           ╱              ╲
                         No                Yes
                      (discard)             │
                                   Local Classifier
                                    (speech/music/noise)
                                   ╱              ╲
                              Not speech        Speech
                              (discard)            │
                                          Gemini Flash API
                                       (transcribe + extract)
                                              │
                                         SQLite DB
                                    (transcripts, tasks,
                                     hashtags, decisions)
```

**Token-saving design:**
- Silero VAD runs 100% locally — silence/background noise never hits the network
- Local audio classifier filters out music and noise before Gemini
- Only real speech segments get sent to Gemini Flash (cheapest model)
- One API call per chunk does transcription + insight extraction together

## Quick start

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set your Gemini API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Start listening
python -m src.cli listen
```

### System audio capture (recommended)

To capture all system audio (Zoom, Meet, Teams, phone calls, etc.):

1. Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) (virtual audio device)
2. Open **Audio MIDI Setup** on macOS
3. Create a **Multi-Output Device** with your speakers + BlackHole
4. Set that as your system output
5. DeskVoice auto-detects BlackHole and captures from it

Without BlackHole, DeskVoice falls back to the microphone (works for speaker-phone calls and in-person conversations).

## CLI Commands

```bash
deskvoice listen              # Start the agent
deskvoice listen --mic        # Use microphone instead of system audio
deskvoice listen --device 5   # Use specific audio device by index
deskvoice devices             # List audio input devices

deskvoice tasks               # Show pending tasks extracted from conversations
deskvoice tasks --all         # Include completed tasks
deskvoice done 42             # Mark task #42 as completed

deskvoice sessions            # Show recent conversation sessions
deskvoice search "quarterly"  # Full-text search across all transcripts
deskvoice tags                # Show all extracted hashtags/topics
```

## What it extracts

From every conversation chunk, Gemini identifies:

| Insight | Example |
|---|---|
| **Tasks** | "John will send the proposal by Friday" → task with assignee + due date |
| **Hashtags** | #budget, #hiring, #product-launch — meaningful topic tags |
| **Decisions** | "We decided to go with vendor A" |
| **Questions** | Open questions raised but not answered |
| **Speakers** | Speaker diarization (Speaker 1, Speaker 2) |

## Project structure

```
src/
├── cli.py                    # CLI interface (click + rich)
├── agent.py                  # Main orchestration loop
├── config.py                 # Configuration from environment
├── capture/
│   ├── audio_stream.py       # System audio capture (sounddevice)
│   └── vad.py                # Silero VAD speech detection
├── processing/
│   ├── classifier.py         # Local audio type classification
│   └── gemini.py             # Gemini Flash transcription + extraction
├── storage/
│   └── database.py           # SQLite local storage
└── models/
    └── models.py             # Data models
```

## Configuration

All config via environment variables (or `.env` file):

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google AI Studio API key |
| `LOG_LEVEL` | No | INFO | Logging verbosity |
| `DB_PATH` | No | ~/.deskvoice/deskvoice.db | SQLite database path |
| `AUDIO_DIR` | No | ~/.deskvoice/audio | Saved audio chunks path |

## Data storage

Everything is stored locally in SQLite at `~/.deskvoice/deskvoice.db`:
- **sessions** — conversation sessions with timestamps and transcripts
- **tasks** — extracted action items with assignees and priorities
- **hashtags** — topic tags linked to sessions
- **transcript_fts** — full-text search index
