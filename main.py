"""Cloud Function entry points for the Meeting Transcript Recorder.

This module exposes two Cloud Functions:

1. `process_meeting_recording` (HTTP-triggered)
   - Accepts a POST with a meeting code or Google Calendar event ID
   - Finds the recording in Google Drive, downloads it, transcribes it, and stores the result

2. `poll_upcoming_meetings` (HTTP-triggered, called by Cloud Scheduler)
   - Scans the calendar for recent meetings, checks for recordings, and kicks off transcription
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime

import functions_framework
from flask import Request, jsonify

from config import Config
from models.transcript import MeetingTranscript, TranscriptSegment
from services import google_meet, transcription, storage
from services.firestore_db import (
    get_transcript,
    save_transcript,
    update_transcript_status,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.http
def process_meeting_recording(request: Request):
    """Process a single meeting recording.

    Expects a JSON body with:
    - meet_code (str): The Google Meet code (e.g., "abc-defg-hij")
    - title (str, optional): Meeting title
    - organizer (str, optional): Organizer email
    - attendees (list[str], optional): Attendee emails
    - event_id (str, optional): Google Calendar event ID

    Alternatively, provide:
    - audio_gcs_uri (str): Direct GCS URI of an already-uploaded audio file

    Returns JSON with the transcript result.
    """
    try:
        body = request.get_json(silent=True) or {}
        meet_code = body.get("meet_code", "")
        audio_gcs_uri = body.get("audio_gcs_uri", "")

        if not meet_code and not audio_gcs_uri:
            return jsonify({"error": "Either 'meet_code' or 'audio_gcs_uri' is required"}), 400

        meeting_id = meet_code or body.get("event_id", audio_gcs_uri)

        # Check if already processed
        existing = get_transcript(meeting_id)
        if existing and existing.status == "completed":
            logger.info("Meeting %s already transcribed, returning existing", meeting_id)
            return jsonify(existing.to_firestore_dict())

        # Create initial transcript record
        transcript_record = MeetingTranscript(
            meeting_id=meeting_id,
            title=body.get("title", ""),
            organizer=body.get("organizer", ""),
            attendees=body.get("attendees", []),
            status="processing",
        )
        save_transcript(transcript_record)

        # Step 1: Get the audio data
        if audio_gcs_uri:
            logger.info("Downloading audio from GCS: %s", audio_gcs_uri)
            audio_data = storage.download_audio(audio_gcs_uri)
            filename = audio_gcs_uri.rsplit("/", 1)[-1]
            transcript_record.audio_gcs_uri = audio_gcs_uri
        else:
            logger.info("Looking for recording for meet code: %s", meet_code)
            recording = google_meet.find_meeting_recording(meet_code)
            if not recording:
                update_transcript_status(
                    meeting_id, "failed", "No recording found in Google Drive"
                )
                return jsonify({"error": "No recording found for this meeting"}), 404

            audio_data = google_meet.download_recording(recording["file_id"])
            filename = recording["name"]

            # Upload to GCS for permanent storage
            gcs_uri = storage.upload_audio(audio_data, meeting_id, filename)
            transcript_record.audio_gcs_uri = gcs_uri

        # Step 2: Transcribe
        logger.info("Starting transcription for %s", meeting_id)
        result = transcription.transcribe_audio(audio_data, filename)

        # Step 3: Save results
        transcript_record.transcript_text = result["text"]
        transcript_record.segments = [
            TranscriptSegment(**seg) for seg in result["segments"]
        ]
        transcript_record.language = result["language"]
        transcript_record.duration_seconds = result["duration"]
        transcript_record.status = "completed"

        save_transcript(transcript_record)
        logger.info("Successfully processed meeting: %s", meeting_id)

        return jsonify(transcript_record.to_firestore_dict())

    except Exception as e:
        logger.error("Error processing meeting: %s", traceback.format_exc())
        if meeting_id:
            update_transcript_status(meeting_id, "failed", str(e))
        return jsonify({"error": str(e)}), 500


@functions_framework.http
def poll_upcoming_meetings(request: Request):
    """Poll for recently completed meetings and process any available recordings.

    This function is designed to be triggered by Cloud Scheduler on a regular
    interval (e.g., every 15 minutes). It:
    1. Lists recent calendar events with Google Meet links
    2. Checks if recordings are available in Google Drive
    3. Triggers transcription for any new recordings found

    Returns JSON summary of processed meetings.
    """
    try:
        hours_back = 4  # Look at meetings from the last 4 hours
        meetings = google_meet.list_upcoming_meetings(hours_ahead=hours_back)

        processed = []
        skipped = []

        for meeting in meetings:
            meet_code = meeting.get("meet_code")
            if not meet_code:
                continue

            # Skip already-processed meetings
            existing = get_transcript(meet_code)
            if existing and existing.status in ("completed", "processing"):
                skipped.append(meet_code)
                continue

            # Check if recording is available
            recording = google_meet.find_meeting_recording(meet_code)
            if not recording:
                continue

            logger.info("Found new recording for meeting: %s", meet_code)

            # Create transcript record
            meeting_start = None
            meeting_end = None
            if meeting.get("start"):
                try:
                    meeting_start = datetime.fromisoformat(meeting["start"])
                except (ValueError, TypeError):
                    pass
            if meeting.get("end"):
                try:
                    meeting_end = datetime.fromisoformat(meeting["end"])
                except (ValueError, TypeError):
                    pass

            transcript_record = MeetingTranscript(
                meeting_id=meet_code,
                title=meeting.get("title", ""),
                organizer=meeting.get("organizer", ""),
                attendees=meeting.get("attendees", []),
                meeting_start=meeting_start,
                meeting_end=meeting_end,
                status="processing",
            )
            save_transcript(transcript_record)

            # Download, upload to GCS, transcribe
            try:
                audio_data = google_meet.download_recording(recording["file_id"])
                gcs_uri = storage.upload_audio(audio_data, meet_code, recording["name"])

                result = transcription.transcribe_audio(audio_data, recording["name"])

                transcript_record.audio_gcs_uri = gcs_uri
                transcript_record.transcript_text = result["text"]
                transcript_record.segments = [
                    TranscriptSegment(**seg) for seg in result["segments"]
                ]
                transcript_record.language = result["language"]
                transcript_record.duration_seconds = result["duration"]
                transcript_record.status = "completed"

                save_transcript(transcript_record)
                processed.append(meet_code)
                logger.info("Successfully transcribed: %s", meet_code)

            except Exception as e:
                logger.error("Failed to process %s: %s", meet_code, e)
                update_transcript_status(meet_code, "failed", str(e))

        return jsonify({
            "processed": processed,
            "skipped": skipped,
            "total_meetings_found": len(meetings),
        })

    except Exception as e:
        logger.error("Error polling meetings: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
