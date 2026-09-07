# =============================================================================
#  Clarion API — containerised backend (FastAPI over s2n.service)
# =============================================================================
#  WHAT RUNS HERE: the Python service only — transcription (Whisper), note
#  generation + verification (via Ollama), the reliability flag, the audit
#  trail, and the HTTP API. The LLMs themselves run in OLLAMA ON THE HOST, not
#  in this image: they are many GB and want the host's GPU/Metal. The container
#  reaches them over HTTP via S2N_OLLAMA_HOST (see docker-compose.yml).
#
#  BUILD:  docker build -t clarion-api .
#  RUN:    see docker-compose.yml (wires the host Ollama + a volume for the DB)
# =============================================================================

FROM python:3.11-slim

# ffmpeg lets Whisper read audio; it is NOT a pip package. --no-install-recommends
# keeps the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-install the (slow) ML stack.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# The application: the installable package (src/) plus the API, config, and
# prompts it reads at runtime. Data, models, notebooks and results are excluded
# by .dockerignore — none belong in a service image.
COPY src/ ./src/
COPY app/api/ ./app/api/
COPY app/__init__.py ./app/
COPY config/ ./config/
COPY prompts/ ./prompts/
RUN pip install --no-cache-dir -e .

# Defaults for a container talking to Ollama on the host. Overridable at run
# time (compose sets the host explicitly for the platform).
ENV S2N_OLLAMA_HOST=http://host.docker.internal:11434 \
    PYTHONUNBUFFERED=1

# The audit DB (results/runs.db) and Whisper's downloaded weights are the two
# things worth persisting across container restarts — mount volumes here.
RUN mkdir -p /app/results /root/.cache/whisper

EXPOSE 8000

# Liveness: the API's own health route (also reports whether models are warm).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# 0.0.0.0 so the port is reachable from outside the container. No --reload: this
# is the run image, not the dev loop.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
