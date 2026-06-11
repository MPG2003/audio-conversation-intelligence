# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt && \
    pip install --prefix=/install --no-cache-dir \
        "openai-whisper" \
        "torch" --index-url https://download.pytorch.org/whl/cpu && \
    python -c "import spacy; spacy.cli.download('en_core_web_sm')" || \
    /install/bin/python -m spacy download en_core_web_sm || true


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install ffmpeg (needed by Whisper for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Download spaCy model at build time (so it's baked into the image)
RUN python -m spacy download en_core_web_sm

# Copy application source
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Make sure __init__.py exists at root so src is importable as a package
RUN touch __init__.py || true

# Expose the port Railway will route traffic to
EXPOSE 8000

# Health check so Railway knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Start the FastAPI server — use shell form so $PORT gets expanded by the shell
CMD ["sh", "-c", "python -m uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
