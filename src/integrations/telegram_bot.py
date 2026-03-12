"""Telegram bot integration for DeskVoice.

One-press interaction:
- Send voice message → auto-transcribed, tasks extracted, saved to second brain
- Tap inline buttons → mark tasks done, view sessions, see people
- Type anything → searches your second brain
- /briefing → today's summary pushed right to your chat

Setup:
1. Talk to @BotFather on Telegram, create a bot, get the token
2. Set TELEGRAM_BOT_TOKEN in your .env
3. Run: deskvoice bot --telegram

The bot uses long polling (no webhook needed), so it works behind NAT/firewalls.
For production, set TELEGRAM_WEBHOOK_URL for webhook mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from src.config import AgentConfig
from src.integrations.chat_handler import ChatHandler, ChatResponse

logger = logging.getLogger(__name__)

# Lazy imports — only load telegram lib when actually used
_telegram = None
_Application = None


def _ensure_telegram():
    global _telegram, _Application
    if _telegram is None:
        import telegram
        from telegram.ext import Application
        _telegram = telegram
        _Application = Application


class TelegramBot:
    """Telegram bot that connects DeskVoice to your pocket.

    Supports:
    - Voice messages (auto-transcribed)
    - Text commands and natural language queries
    - Inline keyboard buttons for task actions
    - Scheduled daily briefings
    """

    def __init__(self, config: AgentConfig, token: Optional[str] = None):
        _ensure_telegram()
        self.config = config
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not set. Get one from @BotFather on Telegram."
            )
        self.handler = ChatHandler(config)
        self._app = None
        self._allowed_users: set[int] = set()

        # Parse allowed user IDs (comma-separated)
        allowed = os.getenv("TELEGRAM_ALLOWED_USERS", "")
        if allowed:
            self._allowed_users = {int(uid.strip()) for uid in allowed.split(",") if uid.strip()}

    def _check_access(self, user_id: int) -> bool:
        """Check if user is allowed (if restrictions are set)."""
        if not self._allowed_users:
            return True  # No restrictions
        return user_id in self._allowed_users

    async def _handle_message(self, update, context) -> None:
        """Handle incoming text messages."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        msg = update.message
        if not msg or not msg.text:
            return

        if not self._check_access(msg.from_user.id):
            await msg.reply_text("🔒 Access denied. Ask the admin to add your Telegram user ID.")
            return

        user_id = str(msg.from_user.id)
        response = self.handler.handle_text(msg.text, user_id)
        await self._send_response(msg, response)

    async def _handle_voice(self, update, context) -> None:
        """Handle incoming voice messages — the 'one press' magic."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        msg = update.message
        if not msg:
            return

        if not self._check_access(msg.from_user.id):
            await msg.reply_text("🔒 Access denied.")
            return

        # Show typing indicator while processing
        await msg.chat.send_action("typing")

        # Download voice file
        voice = msg.voice or msg.audio or msg.document
        if not voice:
            await msg.reply_text("No audio found in message.")
            return

        try:
            file = await context.bot.get_file(voice.file_id)
            audio_bytes = await file.download_as_bytearray()
        except Exception as e:
            logger.error("Failed to download voice: %s", e)
            await msg.reply_text("Failed to download audio. Try again?")
            return

        # Determine mime type
        mime_type = "audio/ogg"  # Telegram voice messages are OGG/Opus
        if msg.audio:
            mime_type = msg.audio.mime_type or "audio/mpeg"
        elif msg.document:
            mime_type = msg.document.mime_type or "audio/ogg"

        duration = voice.duration if hasattr(voice, "duration") else 0

        # Process through chat handler
        await msg.chat.send_action("typing")
        user_id = str(msg.from_user.id)
        response = self.handler.handle_voice(bytes(audio_bytes), duration, user_id, mime_type)
        await self._send_response(msg, response)

    async def _handle_callback(self, update, context) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        if not query:
            return

        await query.answer()

        # Handle meta-commands
        if query.data.startswith("cmd:"):
            cmd = "/" + query.data.split(":")[1]
            user_id = str(query.from_user.id)
            response = self.handler.handle_text(cmd, user_id)
        else:
            user_id = str(query.from_user.id)
            response = self.handler.handle_callback(query.data, user_id)

        try:
            await query.edit_message_text(
                text=response.text,
                parse_mode="Markdown",
            )
        except Exception:
            # If editing fails (e.g., message too old), send new message
            await query.message.reply_text(
                text=response.text,
                parse_mode="Markdown",
            )

    async def _send_response(self, msg, response: ChatResponse) -> None:
        """Send a ChatResponse back to Telegram."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        # Build inline keyboard if buttons exist
        reply_markup = None
        if response.buttons:
            keyboard = []
            row = []
            for btn in response.buttons:
                row.append(InlineKeyboardButton(btn["label"], callback_data=btn["callback"]))
                if len(row) >= 2:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)
            reply_markup = InlineKeyboardMarkup(keyboard)

        # Split long messages (Telegram has 4096 char limit)
        text = response.text
        if len(text) > 4000:
            # Send in chunks
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for i, chunk in enumerate(chunks):
                if i == len(chunks) - 1:
                    await msg.reply_text(chunk, parse_mode="Markdown", reply_markup=reply_markup)
                else:
                    await msg.reply_text(chunk, parse_mode="Markdown")
        else:
            await msg.reply_text(text, parse_mode="Markdown", reply_markup=reply_markup)

    def run(self, webhook_url: Optional[str] = None) -> None:
        """Start the Telegram bot.

        Uses long polling by default, webhook if URL provided.
        """
        from telegram.ext import (
            Application,
            CallbackQueryHandler,
            CommandHandler,
            MessageHandler,
            filters,
        )

        logger.info("Starting Telegram bot...")

        app = Application.builder().token(self.token).build()

        # Command handlers
        for cmd in ["start", "help", "briefing", "today", "tasks", "people",
                     "search", "recent", "digest", "status"]:
            app.add_handler(CommandHandler(cmd, self._handle_message))

        # Voice message handler
        app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO | filters.Document.AUDIO, self._handle_voice
        ))

        # Text message handler (catch-all for queries)
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        # Inline button callbacks
        app.add_handler(CallbackQueryHandler(self._handle_callback))

        self._app = app

        if webhook_url:
            webhook_url = webhook_url.rstrip("/")
            logger.info("Starting webhook mode at %s", webhook_url)
            app.run_webhook(
                listen="0.0.0.0",
                port=int(os.getenv("TELEGRAM_WEBHOOK_PORT", "8443")),
                webhook_url=webhook_url,
            )
        else:
            logger.info("Starting long polling mode...")
            app.run_polling(drop_pending_updates=True)
