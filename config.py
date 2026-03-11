import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "meeting-transcripts-bucket")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "transcripts")
    GOOGLE_WORKSPACE_DOMAIN = os.environ.get("GOOGLE_WORKSPACE_DOMAIN", "")

    # Whisper model to use (via OpenAI API)
    WHISPER_MODEL = "whisper-1"

    # Max audio file size for Whisper API (25MB)
    MAX_AUDIO_SIZE_MB = 25

    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg")
