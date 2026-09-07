# Running Clarion (the product)

Clarion is the productised side of this project: the same reliability pipeline
(transcribe → note → verify → flag) behind an HTTP API, with an audit trail and
per-domain routing. This doc is about **running it**; the research is documented
elsewhere.

There are two ways to run it. Both need one thing first.

## Prerequisite (both paths): Ollama + models on this machine

The large language models run in [Ollama](https://ollama.com/download) on the
host — not inside any container — because they are many GB and want the host's
GPU/Metal.

```bash
ollama serve                     # or just open the Ollama app
ollama pull medgemma:4b          # clinical generator
ollama pull qwen3:14b            # meetings generator
ollama pull llama3.1:8b          # shared verifier (both domains)
```

Whisper (transcription) is separate and downloads its own small weights on first
use — nothing to pre-pull.

---

## Path A — local launch (recommended for day-to-day)

Runs the API (`:8000`) and the web UI (`:3000`) directly, with a preflight that
checks Ollama, pulls any missing models, and checks Node.

```bash
conda activate ds        # or your venv
make run                 # = python scripts/run_product.py
```

Preflight only, without launching:

```bash
make check
```

Open <http://localhost:3000> for the UI, or hit the API at
<http://localhost:8000/api/health>.

---

## Path B — containerised API (portable backend)

Runs the **API** in Docker. The container reaches Ollama on the host over HTTP,
so the prerequisite above still applies. The web UI is not containerised — use
Path A for the UI, or call the API directly.

```bash
docker compose up --build        # or: make docker
```

Then:

```bash
curl http://localhost:8000/api/health
```

`docker compose down` (or `make docker-down`) stops it. Two named volumes persist
across restarts: the audit database (`results/runs.db`) and Whisper's downloaded
weights.

### On Linux
`host.docker.internal` is mapped to the host gateway in `docker-compose.yml`, so
the container can reach the host's Ollama the same way it does on macOS/Windows.

---

## Configuration (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `S2N_OLLAMA_HOST` | `http://localhost:11434` | Where the LLM backend (Ollama) is. Set to `http://host.docker.internal:11434` from a container, or a remote URL. |
| `S2N_DOMAIN` | `clinical` | The default/warm pipeline (`clinical` or `meetings`). A request can still override per call. |

## The API, briefly

| Route | Purpose |
|---|---|
| `POST /api/analyze` | Run the pipeline on an uploaded audio file (streamed). Optional `domain` form field. |
| `GET /api/domains` | Which domains are available, the default, and which is warm. |
| `GET /api/runs`, `/api/runs/{id}` | The audit trail: recent runs, and one run's full record with provenance. |
| `GET /api/stats`, `/api/metrics` | Aggregate counts, and per-stage latency (mean/p50/p90). |
| `GET /api/health` | Liveness + whether models are warm. |

Every response carries an `X-Request-ID`; that id is written into the audit
record, so a run in the database ties back to its request and logs.
