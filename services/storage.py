"""Google Cloud Storage service for storing meeting recordings."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from google.cloud import storage

from config import Config

logger = logging.getLogger(__name__)


def _get_client() -> storage.Client:
    """Create a GCS client."""
    return storage.Client(project=Config.GCP_PROJECT_ID)


def upload_audio(audio_data: bytes, meeting_id: str, filename: str) -> str:
    """Upload audio recording to Google Cloud Storage.

    Args:
        audio_data: Raw audio bytes.
        meeting_id: Meeting identifier used in the GCS path.
        filename: Original filename.

    Returns:
        The GCS URI (gs://bucket/path) of the uploaded file.
    """
    client = _get_client()
    bucket = client.bucket(Config.GCS_BUCKET_NAME)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    blob_path = f"recordings/{meeting_id}/{timestamp}_{filename}"
    blob = bucket.blob(blob_path)

    blob.upload_from_string(audio_data, content_type=_guess_content_type(filename))

    gcs_uri = f"gs://{Config.GCS_BUCKET_NAME}/{blob_path}"
    logger.info("Uploaded audio to %s", gcs_uri)
    return gcs_uri


def download_audio(gcs_uri: str) -> bytes:
    """Download audio from GCS.

    Args:
        gcs_uri: The GCS URI (gs://bucket/path).

    Returns:
        The file content as bytes.
    """
    client = _get_client()

    # Parse gs://bucket/path
    path = gcs_uri.replace("gs://", "")
    bucket_name, blob_path = path.split("/", 1)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    data = blob.download_as_bytes()
    logger.info("Downloaded %d bytes from %s", len(data), gcs_uri)
    return data


def ensure_bucket_exists() -> None:
    """Create the GCS bucket if it doesn't already exist."""
    client = _get_client()
    bucket = client.bucket(Config.GCS_BUCKET_NAME)

    if not bucket.exists():
        bucket.create(location="us-central1")
        logger.info("Created bucket: %s", Config.GCS_BUCKET_NAME)
    else:
        logger.info("Bucket already exists: %s", Config.GCS_BUCKET_NAME)


def _guess_content_type(filename: str) -> str:
    """Guess MIME type from filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_types = {
        "mp3": "audio/mpeg",
        "mp4": "video/mp4",
        "webm": "video/webm",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
        "mpeg": "audio/mpeg",
        "mpga": "audio/mpeg",
    }
    return content_types.get(ext, "application/octet-stream")
