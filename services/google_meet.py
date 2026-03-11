"""Google Meet & Calendar integration.

This service handles:
1. Listing upcoming Google Calendar events that have Google Meet links
2. Detecting when meeting recordings become available in Google Drive
3. Downloading recorded audio files for transcription
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import Config

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def _get_credentials() -> service_account.Credentials:
    """Load Google service account credentials."""
    credentials = service_account.Credentials.from_service_account_file(
        Config.GOOGLE_CREDENTIALS_PATH,
        scopes=SCOPES,
    )
    return credentials


def _get_calendar_service():
    """Build a Google Calendar API service client."""
    credentials = _get_credentials()
    return build("calendar", "v3", credentials=credentials)


def _get_drive_service():
    """Build a Google Drive API service client."""
    credentials = _get_credentials()
    return build("drive", "v3", credentials=credentials)


def list_upcoming_meetings(hours_ahead: int = 24) -> list[dict]:
    """List upcoming Google Calendar events that have Google Meet conference data.

    Args:
        hours_ahead: How many hours ahead to look for meetings.

    Returns:
        List of meeting event dicts with keys:
        - event_id, title, organizer, attendees, start, end, meet_link, meet_code
    """
    service = _get_calendar_service()

    now = datetime.now(timezone.utc)
    time_max = now + timedelta(hours=hours_ahead)

    events_result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=time_max.isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    meetings = []
    for event in events_result.get("items", []):
        conference_data = event.get("conferenceData")
        if not conference_data:
            continue

        # Look for a Google Meet entry point
        meet_link = None
        meet_code = None
        for entry_point in conference_data.get("entryPoints", []):
            if entry_point.get("entryPointType") == "video":
                meet_link = entry_point.get("uri", "")
                # Extract meeting code from the URI
                if "meet.google.com/" in meet_link:
                    meet_code = meet_link.split("meet.google.com/")[-1]
                break

        if not meet_link:
            continue

        attendees = [
            a.get("email", "")
            for a in event.get("attendees", [])
            if a.get("email")
        ]

        meetings.append({
            "event_id": event["id"],
            "title": event.get("summary", "Untitled Meeting"),
            "organizer": event.get("organizer", {}).get("email", ""),
            "attendees": attendees,
            "start": event["start"].get("dateTime", event["start"].get("date")),
            "end": event["end"].get("dateTime", event["end"].get("date")),
            "meet_link": meet_link,
            "meet_code": meet_code,
        })

    logger.info("Found %d upcoming meetings with Google Meet", len(meetings))
    return meetings


def find_meeting_recording(meet_code: str) -> Optional[dict]:
    """Search Google Drive for a recording file matching a Google Meet code.

    Google Meet stores recordings in the organizer's Google Drive under
    "Meet Recordings" folder with the meeting code in the filename.

    Args:
        meet_code: The Google Meet meeting code (e.g., "abc-defg-hij").

    Returns:
        Dict with file_id, name, and mime_type if found, else None.
    """
    service = _get_drive_service()

    # Search for video files whose name contains the meeting code
    query = f"name contains '{meet_code}' and mimeType contains 'video/'"
    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, mimeType, createdTime)",
        orderBy="createdTime desc",
        pageSize=5,
    ).execute()

    files = results.get("files", [])
    if not files:
        logger.info("No recording found for meet code: %s", meet_code)
        return None

    recording = files[0]
    logger.info("Found recording: %s (%s)", recording["name"], recording["id"])
    return {
        "file_id": recording["id"],
        "name": recording["name"],
        "mime_type": recording["mimeType"],
    }


def download_recording(file_id: str) -> bytes:
    """Download a recording file from Google Drive.

    Args:
        file_id: The Google Drive file ID of the recording.

    Returns:
        The file content as bytes.
    """
    service = _get_drive_service()

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            logger.info("Download progress: %d%%", int(status.progress() * 100))

    logger.info("Download complete, size: %d bytes", buffer.tell())
    buffer.seek(0)
    return buffer.read()
