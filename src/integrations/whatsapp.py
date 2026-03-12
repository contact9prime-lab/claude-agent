"""WhatsApp integration for DeskVoice via Meta Cloud API / Twilio.

One-press interaction on WhatsApp:
- Send voice note → transcribed, insights extracted, saved to brain
- Type a question → searches your memory
- Receive daily briefings and task reminders

Setup (Meta Cloud API — free tier available):
1. Create a Meta Business account + WhatsApp Business app
2. Set WHATSAPP_TOKEN, WHATSAPP_PHONE_ID, WHATSAPP_VERIFY_TOKEN in .env
3. Point your webhook to: https://your-server/api/whatsapp/webhook
4. Run: deskvoice cloud (starts web server with WhatsApp webhook)

Setup (Twilio — easier):
1. Get a Twilio account + WhatsApp sandbox
2. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in .env
3. Point Twilio webhook to: https://your-server/api/whatsapp/webhook
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Optional

import httpx

from src.config import AgentConfig
from src.integrations.chat_handler import ChatHandler, ChatResponse

logger = logging.getLogger(__name__)


class WhatsAppIntegration:
    """WhatsApp Business API integration.

    Works with both Meta Cloud API and Twilio.
    Exposes FastAPI routes to mount on the main app.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.handler = ChatHandler(config)

        # Meta Cloud API config
        self.token = os.getenv("WHATSAPP_TOKEN", "")
        self.phone_id = os.getenv("WHATSAPP_PHONE_ID", "")
        self.verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "deskvoice-verify")
        self.app_secret = os.getenv("WHATSAPP_APP_SECRET", "")

        # Twilio config (alternative)
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_from = os.getenv("TWILIO_WHATSAPP_FROM", "")

        self.use_twilio = bool(self.twilio_sid)

    def register_routes(self, app) -> None:
        """Register WhatsApp webhook routes on a FastAPI app."""
        from fastapi import Request
        from fastapi.responses import PlainTextResponse

        @app.get("/api/whatsapp/webhook")
        async def verify_webhook(request: Request):
            """Meta webhook verification (challenge-response)."""
            params = request.query_params
            mode = params.get("hub.mode")
            token = params.get("hub.verify_token")
            challenge = params.get("hub.challenge")

            if mode == "subscribe" and token == self.verify_token:
                logger.info("WhatsApp webhook verified")
                return PlainTextResponse(challenge)
            return PlainTextResponse("Forbidden", status_code=403)

        @app.post("/api/whatsapp/webhook")
        async def handle_webhook(request: Request):
            """Handle incoming WhatsApp messages."""
            body = await request.json()
            logger.debug("WhatsApp webhook: %s", body)

            if self.use_twilio:
                return await self._handle_twilio(request)

            # Meta Cloud API format
            try:
                for entry in body.get("entry", []):
                    for change in entry.get("changes", []):
                        value = change.get("value", {})
                        for message in value.get("messages", []):
                            await self._process_meta_message(message)
            except Exception as e:
                logger.error("WhatsApp processing error: %s", e, exc_info=True)

            return {"status": "ok"}

    async def _process_meta_message(self, message: dict) -> None:
        """Process a message from Meta Cloud API."""
        sender = message.get("from", "")
        msg_type = message.get("type", "")

        if msg_type == "text":
            text = message.get("text", {}).get("body", "")
            response = self.handler.handle_text(text, user_id=sender)
            await self._send_meta_message(sender, response)

        elif msg_type == "audio":
            audio_info = message.get("audio", {})
            media_id = audio_info.get("id", "")
            mime_type = audio_info.get("mime_type", "audio/ogg")

            # Download audio from Meta
            audio_bytes = await self._download_meta_media(media_id)
            if audio_bytes:
                response = self.handler.handle_voice(
                    audio_bytes, user_id=sender, mime_type=mime_type,
                )
                await self._send_meta_message(sender, response)
            else:
                await self._send_meta_message(
                    sender, ChatResponse(text="Could not download audio. Try again?")
                )

        elif msg_type == "interactive":
            # Button reply
            callback = message.get("interactive", {}).get("button_reply", {}).get("id", "")
            if callback:
                response = self.handler.handle_callback(callback, user_id=sender)
                await self._send_meta_message(sender, response)

    async def _download_meta_media(self, media_id: str) -> Optional[bytes]:
        """Download media from Meta Cloud API."""
        try:
            async with httpx.AsyncClient() as client:
                # Get media URL
                resp = await client.get(
                    f"https://graph.facebook.com/v18.0/{media_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                media_url = resp.json().get("url")
                if not media_url:
                    return None

                # Download
                resp = await client.get(
                    media_url,
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                return resp.content
        except Exception as e:
            logger.error("Failed to download WhatsApp media: %s", e)
            return None

    async def _send_meta_message(self, to: str, response: ChatResponse) -> None:
        """Send a message via Meta Cloud API."""
        if not self.token or not self.phone_id:
            logger.error("WhatsApp not configured (missing token/phone_id)")
            return

        url = f"https://graph.facebook.com/v18.0/{self.phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Send text message
        text = response.text
        # WhatsApp doesn't support Markdown * for bold, uses _ instead for italic
        # Keep it simple — strip markdown
        text = text.replace("*", "").replace("_", "")

        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text[:4096]},  # WhatsApp limit
        }

        # If we have buttons, use interactive message
        if response.buttons:
            buttons = []
            for btn in response.buttons[:3]:  # WhatsApp max 3 buttons
                buttons.append({
                    "type": "reply",
                    "reply": {
                        "id": btn["callback"],
                        "title": btn["label"][:20],  # WhatsApp max 20 chars
                    },
                })
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": text[:1024]},
                    "action": {"buttons": buttons},
                },
            }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.error("WhatsApp send failed: %s", resp.text)
        except Exception as e:
            logger.error("WhatsApp send error: %s", e)

    async def _handle_twilio(self, request) -> dict:
        """Handle Twilio-format WhatsApp webhook."""
        form = await request.form()
        sender = form.get("From", "").replace("whatsapp:", "")
        body = form.get("Body", "")
        num_media = int(form.get("NumMedia", "0"))

        if num_media > 0:
            # Voice message
            media_url = form.get("MediaUrl0", "")
            media_type = form.get("MediaContentType0", "audio/ogg")
            if media_url:
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            media_url,
                            auth=(self.twilio_sid, self.twilio_token),
                        )
                        audio_bytes = resp.content
                except Exception as e:
                    logger.error("Twilio media download failed: %s", e)
                    return {"status": "error"}

                response = self.handler.handle_voice(audio_bytes, user_id=sender, mime_type=media_type)
                await self._send_twilio_message(sender, response)
        elif body:
            response = self.handler.handle_text(body, user_id=sender)
            await self._send_twilio_message(sender, response)

        return {"status": "ok"}

    async def _send_twilio_message(self, to: str, response: ChatResponse) -> None:
        """Send via Twilio WhatsApp."""
        if not self.twilio_sid:
            return

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json",
                    auth=(self.twilio_sid, self.twilio_token),
                    data={
                        "From": f"whatsapp:{self.twilio_from}",
                        "To": f"whatsapp:{to}",
                        "Body": response.text[:1600],
                    },
                )
        except Exception as e:
            logger.error("Twilio send failed: %s", e)

    async def send_proactive(self, phone: str, text: str) -> None:
        """Send a proactive message (e.g., daily briefing, reminder).

        Note: WhatsApp requires template messages for proactive outreach.
        This works within 24h of last user message.
        """
        response = ChatResponse(text=text)
        if self.use_twilio:
            await self._send_twilio_message(phone, response)
        else:
            await self._send_meta_message(phone, response)
