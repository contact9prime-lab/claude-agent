"""CLI interface for DeskVoice agent.

Usage:
    deskvoice listen          Start listening (main agent loop)
    deskvoice devices         List available audio input devices
    deskvoice tasks           Show pending tasks
    deskvoice sessions        Show recent sessions
    deskvoice search <query>  Search transcripts
    deskvoice tags            Show all hashtags
"""

from __future__ import annotations

import logging
import signal
import sys

import click
from rich.console import Console
from rich.table import Table

from src.agent import DeskVoiceAgent
from src.capture.audio_stream import find_blackhole_device, list_audio_devices
from src.config import AgentConfig
from src.storage.database import Database

logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """DeskVoice — always-on meeting & call transcript agent."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--device", type=int, default=None, help="Audio device index (use 'devices' to list)")
@click.option("--mic", is_flag=True, help="Use default microphone instead of system audio")
def listen(device: int | None, mic: bool) -> None:
    """Start listening and recording transcripts."""
    config = AgentConfig()

    errors = config.validate()
    if errors:
        for e in errors:
            console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Device selection
    if device is None and not mic:
        bh = find_blackhole_device()
        if bh is not None:
            console.print(f"[green]Found BlackHole device (index {bh}) for system audio capture[/green]")
            device = bh
        else:
            console.print(
                "[yellow]BlackHole not found. Using default microphone.[/yellow]\n"
                "For system audio capture, install BlackHole: "
                "https://github.com/ExistentialAudio/BlackHole"
            )

    agent = DeskVoiceAgent(config, device=device)

    # Handle graceful shutdown
    def _signal_handler(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        agent.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    console.print("[bold green]DeskVoice listening...[/bold green] Press Ctrl+C to stop.")
    console.print(f"  Provider: {config.llm.provider}")
    console.print(f"  Model: {config.llm.model}")
    console.print(f"  Database: {config.storage.db_path}")
    console.print()

    agent.start()


@cli.command()
def devices() -> None:
    """List available audio input devices."""
    devs = list_audio_devices()
    bh = find_blackhole_device()

    table = Table(title="Audio Input Devices")
    table.add_column("Index", style="cyan")
    table.add_column("Name")
    table.add_column("Channels", justify="right")
    table.add_column("Sample Rate", justify="right")
    table.add_column("Note", style="green")

    for dev in devs:
        note = ""
        if dev["index"] == bh:
            note = "← System audio (BlackHole)"
        table.add_row(
            str(dev["index"]),
            dev["name"],
            str(dev["max_input_channels"]),
            f"{dev['default_samplerate']:.0f}",
            note,
        )

    console.print(table)
    if bh is None:
        console.print(
            "\n[yellow]Tip:[/yellow] Install BlackHole for system audio capture: "
            "https://github.com/ExistentialAudio/BlackHole"
        )


@cli.command()
@click.option("--all", "show_all", is_flag=True, help="Show completed tasks too")
def tasks(show_all: bool) -> None:
    """Show extracted tasks."""
    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()

    task_list = db.list_tasks(pending_only=not show_all)

    if not task_list:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(title="Tasks")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Task")
    table.add_column("Assignee", style="yellow")
    table.add_column("Priority", style="magenta")
    table.add_column("Due", style="green")
    table.add_column("Session")
    table.add_column("Done", justify="center")

    for t in task_list:
        done = "✓" if t["completed"] else ""
        table.add_row(
            str(t["id"]),
            t["description"],
            t["assignee"] or "-",
            t["priority"],
            t["due_hint"] or "-",
            t.get("session_title") or f"#{t['session_id']}",
            done,
        )

    console.print(table)
    db.close()


@cli.command()
@click.option("--limit", default=10, help="Number of sessions to show")
def sessions(limit: int) -> None:
    """Show recent conversation sessions."""
    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()

    session_list = db.list_sessions(limit=limit)

    if not session_list:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Recent Sessions")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Started")
    table.add_column("Duration", justify="right")
    table.add_column("Status", style="green")
    table.add_column("Title / Summary")

    for s in session_list:
        duration = f"{s['total_speech_seconds']:.0f}s"
        title = s["title"] or s["summary"][:60] if s["summary"] else "[dim]untitled[/dim]"
        table.add_row(
            str(s["id"]),
            s["started_at"][:19],
            duration,
            s["status"],
            title,
        )

    console.print(table)
    db.close()


@cli.command()
@click.argument("query")
def search(query: str) -> None:
    """Search across all transcripts."""
    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()

    results = db.search_transcripts(query)

    if not results:
        console.print(f"[dim]No results for '{query}'[/dim]")
        return

    for s in results:
        console.print(f"\n[bold cyan]Session #{s['id']}[/bold cyan] — {s['started_at'][:19]}")
        if s["summary"]:
            console.print(f"  [green]{s['summary']}[/green]")
        # Show snippet of transcript with match
        transcript = s.get("transcript", "")
        if transcript:
            lower_t = transcript.lower()
            idx = lower_t.find(query.lower())
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(transcript), idx + len(query) + 50)
                snippet = transcript[start:end].replace("\n", " ")
                console.print(f"  ...{snippet}...")

    db.close()


@cli.command()
def tags() -> None:
    """Show all extracted hashtags."""
    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()

    hashtag_list = db.list_hashtags()

    if not hashtag_list:
        console.print("[dim]No hashtags found yet.[/dim]")
        return

    # Group by tag
    by_tag: dict[str, list] = {}
    for h in hashtag_list:
        by_tag.setdefault(h["tag"], []).append(h)

    table = Table(title="Hashtags")
    table.add_column("Tag", style="cyan")
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Latest Context")

    for tag, entries in sorted(by_tag.items(), key=lambda x: -len(x[1])):
        table.add_row(
            f"#{tag}",
            str(len(entries)),
            entries[0]["context"][:60] if entries[0]["context"] else "-",
        )

    console.print(table)
    db.close()


@cli.command()
def speakers() -> None:
    """List enrolled speaker voice profiles."""
    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()

    speaker_list = db.list_speakers()
    if not speaker_list:
        console.print("[dim]No speakers enrolled yet. Use 'deskvoice enroll <name>' to add one.[/dim]")
        return

    table = Table(title="Speaker Profiles")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Name", style="yellow")
    table.add_column("Enrolled")

    for s in speaker_list:
        table.add_row(str(s["id"]), s["name"], s["created_at"][:19])

    console.print(table)
    db.close()


@cli.command()
@click.argument("name")
@click.option("--duration", default=10, help="Recording duration in seconds")
def enroll(name: str, duration: int) -> None:
    """Enroll a speaker by recording a voice sample."""
    from src.processing.speaker_id import extract_embedding, record_enrollment_audio

    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()

    console.print(f"[bold]Enrolling speaker: {name}[/bold]")
    console.print(f"Please speak naturally for {duration} seconds...")
    console.print("[yellow]Recording starts now![/yellow]")

    audio = record_enrollment_audio(duration_sec=duration)
    console.print("[green]Recording complete. Processing...[/green]")

    embedding = extract_embedding(audio)
    speaker_id = db.add_speaker(name, embedding)
    console.print(f"[bold green]Speaker '{name}' enrolled (ID: {speaker_id})[/bold green]")
    db.close()


@cli.command()
@click.option("--device", type=int, default=None, help="Audio device index")
@click.option("--mic", is_flag=True, help="Use default microphone")
@click.option("--port", default=8765, help="Web UI port")
@click.option("--no-listen", is_flag=True, help="UI only, no audio capture")
def ui(device: int | None, mic: bool, port: int, no_listen: bool) -> None:
    """Start the web dashboard with live transcript view."""
    import threading
    import uvicorn
    from src.web.app import create_app, set_agent_ref

    config = AgentConfig()

    errors = config.validate()
    if not no_listen and errors:
        for e in errors:
            console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    app = create_app(config)

    # Start agent in background thread (unless --no-listen)
    agent = None
    if not no_listen:
        if device is None and not mic:
            bh = find_blackhole_device()
            if bh is not None:
                device = bh

        agent = DeskVoiceAgent(config, device=device)
        set_agent_ref(agent)

        def run_agent():
            try:
                agent.start()
            except Exception as e:
                logger.error("Agent error: %s", e)

        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()

    console.print(f"[bold green]DeskVoice UI running at http://localhost:{port}[/bold green]")
    console.print(f"  Provider: {config.llm.provider} / {config.llm.model}")
    if not no_listen:
        console.print("  Audio capture: active")
    console.print()

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        if agent:
            agent.stop()


@cli.command("tui")
@click.option("--device", type=int, default=None, help="Audio device index")
@click.option("--mic", is_flag=True, help="Use default microphone")
@click.option("--no-listen", is_flag=True, help="TUI only, no audio capture")
def tui_cmd(device: int | None, mic: bool, no_listen: bool) -> None:
    """Start the terminal UI (like Claude Code)."""
    from src.tui.app import DeskVoiceTUI

    config = AgentConfig()

    errors = config.validate()
    if not no_listen and errors:
        for e in errors:
            console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    agent = None
    if not no_listen:
        if device is None and not mic:
            bh = find_blackhole_device()
            if bh is not None:
                device = bh
        agent = DeskVoiceAgent(config, device=device)

    app = DeskVoiceTUI(config=config, agent=agent)
    app.run()


@cli.command()
@click.argument("task_id", type=int)
def done(task_id: int) -> None:
    """Mark a task as completed."""
    config = AgentConfig()
    db = Database(config.storage.db_path)
    db.connect()
    db.complete_task(task_id)
    console.print(f"[green]Task #{task_id} marked as done.[/green]")
    db.close()


@cli.command()
@click.option("--telegram", is_flag=True, help="Start Telegram bot")
@click.option("--whatsapp", is_flag=True, help="Enable WhatsApp webhook")
@click.option("--slack", is_flag=True, help="Start Slack bot")
@click.option("--all", "all_bots", is_flag=True, help="Start all configured bots")
def bot(telegram: bool, whatsapp: bool, slack: bool, all_bots: bool) -> None:
    """Start chat bot integrations (Telegram, WhatsApp, Slack).

    Examples:
        deskvoice bot --telegram
        deskvoice bot --slack
        deskvoice bot --all
    """
    import threading

    config = AgentConfig()

    if all_bots:
        telegram = whatsapp = slack = True

    if not (telegram or whatsapp or slack):
        console.print("[yellow]Specify at least one bot: --telegram, --slack, --whatsapp, or --all[/yellow]")
        return

    started = []

    if telegram:
        try:
            from src.integrations.telegram_bot import TelegramBot
            tg = TelegramBot(config)
            console.print("[green]Starting Telegram bot...[/green]")
            # Telegram runs its own event loop, so run in thread if other bots need to start
            if slack:
                t = threading.Thread(target=tg.run, daemon=True)
                t.start()
                started.append("Telegram")
            else:
                started.append("Telegram")
                console.print(f"[bold green]Bots running: {', '.join(started)}[/bold green]")
                tg.run()  # Blocks
                return
        except ValueError as e:
            console.print(f"[red]Telegram:[/red] {e}")
        except ImportError:
            console.print("[red]Telegram:[/red] pip install python-telegram-bot>=21.0")

    if slack:
        try:
            from src.integrations.slack_bot import SlackBot
            sb = SlackBot(config)
            console.print("[green]Starting Slack bot...[/green]")
            started.append("Slack")
            console.print(f"[bold green]Bots running: {', '.join(started)}[/bold green]")
            sb.run()  # Blocks
        except ValueError as e:
            console.print(f"[red]Slack:[/red] {e}")
        except ImportError:
            console.print("[red]Slack:[/red] pip install slack-bolt>=1.18")

    if whatsapp and not slack:
        console.print("[yellow]WhatsApp uses webhooks — start with 'deskvoice cloud' to enable.[/yellow]")

    if started:
        console.print(f"[bold green]Bots running: {', '.join(started)}[/bold green]")


@cli.command()
@click.option("--port", default=8765, help="Web UI port")
@click.option("--telegram", is_flag=True, help="Also start Telegram bot")
@click.option("--no-listen", is_flag=True, default=True, help="Skip audio capture (cloud mode)")
def cloud(port: int, telegram: bool, no_listen: bool) -> None:
    """Start DeskVoice in cloud mode: Web UI + chat bots + webhooks.

    This is the 'deploy to cloud' mode — no microphone needed.
    Receives audio via web uploads, Telegram voice messages, WhatsApp, etc.

    Examples:
        deskvoice cloud                     # Web UI + WhatsApp webhook
        deskvoice cloud --telegram          # + Telegram bot
        docker-compose up                   # Same thing, in Docker
    """
    import os
    import threading
    import uvicorn
    from src.web.app import create_app

    config = AgentConfig()

    errors = config.validate()
    if errors:
        for e in errors:
            console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    app = create_app(config)

    # Register WhatsApp webhook if configured
    whatsapp_token = os.getenv("WHATSAPP_TOKEN") or os.getenv("TWILIO_ACCOUNT_SID")
    if whatsapp_token:
        try:
            from src.integrations.whatsapp import WhatsAppIntegration
            wa = WhatsAppIntegration(config)
            wa.register_routes(app)
            console.print("[green]WhatsApp webhook enabled at /api/whatsapp/webhook[/green]")
        except ImportError:
            console.print("[yellow]WhatsApp: pip install httpx for WhatsApp support[/yellow]")

    # Register Slack webhook if configured
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if slack_token:
        try:
            from src.integrations.slack_bot import SlackBot
            sb = SlackBot(config)
            sb.register_webhook_routes(app)
            console.print("[green]Slack webhook enabled at /api/slack/events[/green]")
        except (ImportError, ValueError):
            pass

    # Start Telegram bot in background if requested
    if telegram or os.getenv("TELEGRAM_BOT_TOKEN"):
        try:
            from src.integrations.telegram_bot import TelegramBot
            tg = TelegramBot(config)
            tg_thread = threading.Thread(target=tg.run, daemon=True)
            tg_thread.start()
            console.print("[green]Telegram bot started (long polling)[/green]")
        except (ValueError, ImportError) as e:
            console.print(f"[yellow]Telegram: {e}[/yellow]")

    console.print(f"\n[bold green]DeskVoice Cloud running at http://0.0.0.0:{port}[/bold green]")
    console.print(f"  LLM: {config.llm.provider} / {config.llm.model}")
    console.print(f"  DB: {config.storage.db_path}")
    console.print()

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
