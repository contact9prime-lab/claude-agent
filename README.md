# Meeting Transcript Recorder

Records Google Meet meetings and generates transcripts using OpenAI Whisper, deployed as Google Cloud Functions.

## Architecture

```
Google Meet → Recording saved to Google Drive
                        ↓
              Cloud Scheduler (every 15 min)
                        ↓
              Cloud Function: poll_upcoming_meetings
                        ↓
              Download recording from Drive
                        ↓
              Upload to Cloud Storage (archival)
                        ↓
              Transcribe via OpenAI Whisper API
                        ↓
              Store transcript in Firestore
```

## Two Cloud Functions

| Function | Purpose | Trigger |
|---|---|---|
| `process_meeting_recording` | Process a single meeting by meet code or GCS audio URI | HTTP POST |
| `poll_upcoming_meetings` | Scan calendar for recent meetings, auto-transcribe new recordings | HTTP (Cloud Scheduler) |

## Setup

### 1. Google Cloud project

```bash
# Enable required APIs
gcloud services enable \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  calendar-json.googleapis.com \
  drive.googleapis.com
```

### 2. Service account

Create a service account with these roles:
- **Cloud Functions Invoker** (for Cloud Scheduler)
- **Cloud Datastore User** (for Firestore)
- **Storage Object Admin** (for GCS)

For Google Workspace access (Calendar + Drive), set up [domain-wide delegation](https://developers.google.com/identity/protocols/oauth2/service-account#delegatingauthority) with these scopes:
- `https://www.googleapis.com/auth/calendar.readonly`
- `https://www.googleapis.com/auth/drive.readonly`

### 3. Environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

### 4. Deploy

```bash
chmod +x deploy.sh
./deploy.sh          # Deploy both functions + create bucket
./deploy.sh scheduler  # Set up 15-minute polling
```

## Manual usage

Trigger transcription for a specific meeting:

```bash
curl -X POST "$FUNCTION_URL" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"meet_code": "abc-defg-hij"}'
```

Or process an audio file already in GCS:

```bash
curl -X POST "$FUNCTION_URL" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"audio_gcs_uri": "gs://your-bucket/recording.webm"}'
```

## Local development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run locally with functions-framework
functions-framework --target=process_meeting_recording --debug
```

## Project structure

```
├── main.py                  # Cloud Function entry points
├── config.py                # Configuration from environment
├── deploy.sh                # Deployment script
├── requirements.txt         # Python dependencies
├── models/
│   └── transcript.py        # Pydantic data models
├── services/
│   ├── google_meet.py       # Google Calendar + Drive integration
│   ├── transcription.py     # OpenAI Whisper transcription
│   ├── storage.py           # Google Cloud Storage operations
│   └── firestore_db.py      # Firestore CRUD operations
└── tests/
```
