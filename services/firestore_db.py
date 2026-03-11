"""Firestore service for storing and retrieving transcripts."""

from __future__ import annotations

import logging
from typing import Optional

from google.cloud import firestore

from config import Config
from models.transcript import MeetingTranscript

logger = logging.getLogger(__name__)


def _get_client() -> firestore.Client:
    """Create a Firestore client."""
    return firestore.Client(project=Config.GCP_PROJECT_ID)


def save_transcript(transcript: MeetingTranscript) -> str:
    """Save a meeting transcript to Firestore.

    Args:
        transcript: The MeetingTranscript to save.

    Returns:
        The Firestore document ID.
    """
    client = _get_client()
    collection = client.collection(Config.FIRESTORE_COLLECTION)

    doc_ref = collection.document(transcript.meeting_id)
    doc_ref.set(transcript.to_firestore_dict())

    logger.info("Saved transcript for meeting: %s", transcript.meeting_id)
    return transcript.meeting_id


def get_transcript(meeting_id: str) -> Optional[MeetingTranscript]:
    """Retrieve a transcript from Firestore by meeting ID.

    Args:
        meeting_id: The meeting identifier.

    Returns:
        MeetingTranscript if found, else None.
    """
    client = _get_client()
    doc = client.collection(Config.FIRESTORE_COLLECTION).document(meeting_id).get()

    if not doc.exists:
        logger.info("No transcript found for meeting: %s", meeting_id)
        return None

    return MeetingTranscript.from_firestore_dict(doc.to_dict())


def update_transcript_status(meeting_id: str, status: str, error_message: str = "") -> None:
    """Update the processing status of a transcript.

    Args:
        meeting_id: The meeting identifier.
        status: New status (pending, processing, completed, failed).
        error_message: Error details if status is 'failed'.
    """
    client = _get_client()
    doc_ref = client.collection(Config.FIRESTORE_COLLECTION).document(meeting_id)

    update_data = {"status": status}
    if error_message:
        update_data["error_message"] = error_message

    doc_ref.update(update_data)
    logger.info("Updated meeting %s status to: %s", meeting_id, status)


def list_transcripts(limit: int = 50) -> list[MeetingTranscript]:
    """List recent transcripts ordered by creation time.

    Args:
        limit: Maximum number of transcripts to return.

    Returns:
        List of MeetingTranscript objects.
    """
    client = _get_client()
    query = (
        client.collection(Config.FIRESTORE_COLLECTION)
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )

    transcripts = []
    for doc in query.stream():
        transcripts.append(MeetingTranscript.from_firestore_dict(doc.to_dict()))

    logger.info("Listed %d transcripts", len(transcripts))
    return transcripts
