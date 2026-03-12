FROM python:3.11-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    pulseaudio-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install chat integration deps
RUN pip install --no-cache-dir \
    python-telegram-bot>=21.0 \
    slack-bolt>=1.18 \
    httpx>=0.25

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

# Data directory (mount a volume here to persist across runs)
RUN mkdir -p /data/audio
ENV DB_PATH=/data/deskvoice.db
ENV AUDIO_DIR=/data/audio

# PulseAudio config — connect to host's PulseAudio over unix socket
ENV PULSE_SERVER=unix:/tmp/pulseaudio.socket

# Web UI port
EXPOSE 8765
# Telegram webhook port
EXPOSE 8443

ENTRYPOINT ["deskvoice"]
CMD ["cloud"]
