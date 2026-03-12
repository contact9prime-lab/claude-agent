"""Slack bot integration for DeskVoice.

One-press interaction in Slack:
- Share audio/video clips → transcribed + insights extracted
- Use slash commands → /dv-briefing, /dv-tasks, /dv-search
- DM the bot → natural language queries
- Get proactive notifications in a channel

Setup:
1. Create a Slack app at api.slack.com/apps
2. Enable Socket Mode (no public URL needed) or Events API
3. Set SLACK_BOT_TOKEN, SLACK_APP_TOKEN in .env
4. Bot scopes needed: chat:write, files:read, im:history, commands
5. Run: deskvoice bot --slack
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.config import AgentConfig
from src.integrations.chat_handler import ChatHandler, ChatResponse

logger = logging.getLogger(__name__)


class SlackBot:
    """Slack bot powered by DeskVoice second brain.

    Supports Socket Mode (no webhook needed) and Events API.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.handler = ChatHandler(config)

        self.bot_token = os.getenv("SLACK_BOT_TOKEN", "")
        self.app_token = os.getenv("SLACK_APP_TOKEN", "")  # For Socket Mode
        self.signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")

        if not self.bot_token:
            raise ValueError("SLACK_BOT_TOKEN not set. Create a Slack app first.")

    def run(self) -> None:
        """Start the Slack bot using Socket Mode."""
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        app = App(token=self.bot_token, signing_secret=self.signing_secret)

        # --- Message handler (DMs) ---
        @app.message("")
        def handle_message(message, say):
            text = message.get("text", "")
            user_id = message.get("user", "")

            # Check if it's a file share (audio/video)
            files = message.get("files", [])
            audio_file = None
            for f in files:
                if f.get("mimetype", "").startswith("audio/") or f.get("mimetype", "").startswith("video/"):
                    audio_file = f
                    break

            if audio_file:
                self._handle_audio_file(audio_file, user_id, say)
            elif text:
                response = self.handler.handle_text(text, user_id)
                self._send_response(say, response)

        # --- Slash commands ---
        @app.command("/dv-briefing")
        def cmd_briefing(ack, respond, command):
            ack()
            user_id = command.get("user_id", "")
            response = self.handler.handle_text("/briefing", user_id)
            respond(self._format_slack_response(response))

        @app.command("/dv-tasks")
        def cmd_tasks(ack, respond, command):
            ack()
            user_id = command.get("user_id", "")
            response = self.handler.handle_text("/tasks", user_id)
            respond(self._format_slack_response(response))

        @app.command("/dv-search")
        def cmd_search(ack, respond, command):
            ack()
            query = command.get("text", "")
            user_id = command.get("user_id", "")
            response = self.handler.handle_text(f"/search {query}", user_id)
            respond(self._format_slack_response(response))

        @app.command("/dv-people")
        def cmd_people(ack, respond, command):
            ack()
            user_id = command.get("user_id", "")
            response = self.handler.handle_text("/people", user_id)
            respond(self._format_slack_response(response))

        @app.command("/dv-digest")
        def cmd_digest(ack, respond, command):
            ack()
            user_id = command.get("user_id", "")
            response = self.handler.handle_text("/digest", user_id)
            respond(self._format_slack_response(response))

        # --- Button actions ---
        @app.action("dv_action")
        def handle_action(ack, body, respond):
            ack()
            action = body.get("actions", [{}])[0]
            callback = action.get("value", "")
            user_id = body.get("user", {}).get("id", "")
            response = self.handler.handle_callback(callback, user_id)
            respond(self._format_slack_response(response))

        # Start
        if self.app_token:
            logger.info("Starting Slack bot in Socket Mode...")
            handler = SocketModeHandler(app, self.app_token)
            handler.start()
        else:
            logger.info("Starting Slack bot (Events API mode)...")
            app.start(port=int(os.getenv("SLACK_PORT", "3000")))

    def _handle_audio_file(self, file_info: dict, user_id: str, say) -> None:
        """Download and process an audio file shared in Slack."""
        import httpx

        url = file_info.get("url_private_download") or file_info.get("url_private")
        if not url:
            say("Could not access the audio file.")
            return

        try:
            resp = httpx.get(
                url,
                headers={"Authorization": f"Bearer {self.bot_token}"},
            )
            audio_bytes = resp.content
        except Exception as e:
            logger.error("Failed to download Slack file: %s", e)
            say("Failed to download the audio. Try again?")
            return

        mime_type = file_info.get("mimetype", "audio/ogg")
        duration = file_info.get("duration_ms", 0) / 1000 if file_info.get("duration_ms") else 0

        say("🎧 Processing your audio...")
        response = self.handler.handle_voice(audio_bytes, duration, user_id, mime_type)
        self._send_response(say, response)

    def _send_response(self, say, response: ChatResponse) -> None:
        """Send a ChatResponse to Slack."""
        blocks = self._format_slack_response(response)
        say(**blocks)

    def _format_slack_response(self, response: ChatResponse) -> dict:
        """Format ChatResponse as Slack Block Kit message."""
        blocks = []

        # Main text as section
        text = response.text
        # Convert Telegram markdown to Slack mrkdwn
        text = text.replace("*", "*")  # Bold stays same
        text = text.replace("_", "_")  # Italic stays same

        # Split into blocks of max 3000 chars
        for i in range(0, len(text), 3000):
            chunk = text[i:i + 3000]
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": chunk},
            })

        # Add action buttons
        if response.buttons:
            elements = []
            for btn in response.buttons[:5]:  # Slack max 5 per block
                elements.append({
                    "type": "button",
                    "text": {"type": "plain_text", "text": btn["label"][:75]},
                    "action_id": "dv_action",
                    "value": btn["callback"],
                })
            blocks.append({"type": "actions", "elements": elements})

        return {"text": response.text[:200], "blocks": blocks}

    def register_webhook_routes(self, app) -> None:
        """Register Slack Events API routes on a FastAPI app (alternative to Socket Mode)."""
        from fastapi import Request

        @app.post("/api/slack/events")
        async def slack_events(request: Request):
            body = await request.json()

            # URL verification challenge
            if body.get("type") == "url_verification":
                return {"challenge": body.get("challenge")}

            # Event handling
            event = body.get("event", {})
            event_type = event.get("type")

            if event_type == "message" and not event.get("bot_id"):
                text = event.get("text", "")
                user_id = event.get("user", "")
                channel = event.get("channel", "")

                # Check for files
                files = event.get("files", [])
                audio_file = None
                for f in files:
                    if f.get("mimetype", "").startswith("audio/"):
                        audio_file = f
                        break

                if audio_file:
                    import httpx
                    url = audio_file.get("url_private_download") or audio_file.get("url_private")
                    if url:
                        async with httpx.AsyncClient() as client:
                            resp = await client.get(url, headers={"Authorization": f"Bearer {self.bot_token}"})
                            audio_bytes = resp.content
                        response = self.handler.handle_voice(audio_bytes, user_id=user_id, mime_type=audio_file.get("mimetype", "audio/ogg"))
                    else:
                        response = ChatResponse(text="Could not download audio.")
                elif text:
                    response = self.handler.handle_text(text, user_id)
                else:
                    return {"status": "ok"}

                # Send reply
                import httpx
                async with httpx.AsyncClient() as client:
                    slack_msg = self._format_slack_response(response)
                    await client.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                        json={"channel": channel, **slack_msg},
                    )

            return {"status": "ok"}

        @app.post("/api/slack/commands")
        async def slack_commands(request: Request):
            form = await request.form()
            command = form.get("command", "")
            text = form.get("text", "")
            user_id = form.get("user_id", "")

            cmd_map = {
                "/dv-briefing": "/briefing",
                "/dv-tasks": "/tasks",
                "/dv-search": f"/search {text}",
                "/dv-people": "/people",
                "/dv-digest": "/digest",
            }

            mapped = cmd_map.get(command, f"/search {text}")
            response = self.handler.handle_text(mapped, user_id)
            return self._format_slack_response(response)
