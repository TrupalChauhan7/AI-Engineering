"""Clarion API — a thin FastAPI shell over ``s2n.service``.

WHY thin: every number this serves comes from the same pipeline the research
harness uses. This module transports and formats; it never scores. Two routes
replay pre-computed sample results instantly, one route runs the real local
pipeline on an upload and streams each stage as it completes.

LEAK SAFETY: imports ``s2n.service`` only, which itself can reach the
transcription/generation/verification paths and nothing else. No gold note or
human-eval loader is reachable from this process.

Run:  uvicorn app.api.main:app --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.api.spans import locate_claims
from app.api.warmup import WarmPool
from s2n.config import ROOT, load_config
from s2n.service import run_pipeline_staged

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("clarion")

SAMPLES_DIR = ROOT / "app" / "web" / "public" / "samples"
ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # a 30-min consultation WAV is ~55 MB

POOL: WarmPool | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the demo models before the first request rather than during it."""
    global POOL
    cfg = load_config()
    POOL = WarmPool(cfg)
    await asyncio.to_thread(POOL.warm)  # never block the event loop
    yield
    POOL = None


app = FastAPI(title="Clarion API", version="1.0.0", lifespan=lifespan)

# The Next dev server proxies /api/* here, but allow direct browser calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# samples — pre-computed results, replayed instantly (no models run)
# --------------------------------------------------------------------------


def _sample_files() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(SAMPLES_DIR.glob("*.json"))


@app.get("/api/samples")
def list_samples() -> list[dict]:
    """Sample consultations available for instant replay (may be empty)."""
    out = []
    for path in _sample_files():
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "id": path.stem,
                "label": data.get("label", path.stem),
                "durationLabel": data.get("durationLabel", ""),
            }
        )
    return out


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: str) -> dict:
    """The full cached result for one sample."""
    path = SAMPLES_DIR / f"{Path(sample_id).name}.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Unknown sample. Run scripts/build_samples.py to generate the cache.",
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# analyze — the real local pipeline, streamed stage by stage
# --------------------------------------------------------------------------


def _sse(event: str, payload: dict) -> str:
    """One Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _analyze_stream(audio_path: Path, cfg: dict, components: dict, timeout_s: float):
    """Yield SSE frames as each pipeline stage lands.

    The pipeline runs on a worker thread and reports through a queue, so this
    generator is never blocked by Whisper or Ollama. That buys two things the
    demo needs: a stage that overruns ``timeout_s`` becomes a clear error
    instead of a hang, and a client that disconnects (cancel, closed tab) sets
    the stop flag so the worker stops at the next stage boundary rather than
    grinding on invisibly.

    A ``heartbeat`` frame goes out every second while a stage is in flight, so
    the UI can prove liveness rather than showing a frozen label.
    """
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        try:
            for event in run_pipeline_staged(audio_path=audio_path, cfg=cfg, **components):
                if stop.is_set():
                    return
                events.put(("event", event))
        except Exception as exc:  # noqa: BLE001 — the client must see *why* it failed
            events.put(("error", str(exc) or exc.__class__.__name__))
        finally:
            events.put(("end", None))

    thread = threading.Thread(target=worker, name="clarion-pipeline", daemon=True)
    thread.start()

    note = ""  # request-local: never shared between concurrent streams
    stage = "transcribe"
    stage_started = time.perf_counter()
    try:
        while True:
            try:
                kind, payload = events.get(timeout=1.0)
            except queue.Empty:
                waited = time.perf_counter() - stage_started
                if waited > timeout_s:
                    log.warning("stage '%s' exceeded %.0fs — aborting", stage, timeout_s)
                    yield _sse(
                        "error",
                        {
                            "message": f"The {stage} stage took longer than "
                            f"{int(timeout_s // 60)} minutes and was stopped. "
                            f"Try a shorter clip.",
                        },
                    )
                    return
                # liveness: the UI shows its own timer, this proves the stream is open
                yield _sse("heartbeat", {"stage": stage, "elapsed_s": round(waited, 1)})
                continue

            if kind == "end":
                return
            if kind == "error":
                yield _sse("error", {"message": payload})
                return

            event = payload
            if event["stage"] == "done":
                yield _sse("done", {"timings": event["timings"]})
                continue

            frame = dict(event)
            if frame.get("status") == "running":
                stage = frame["stage"]
                stage_started = time.perf_counter()
            if frame.get("stage") == "generate" and frame.get("status") == "done":
                note = frame["note"]
            # give the UI the spans it needs to underline flagged text
            if "reliability" in frame:
                frame["spans"] = locate_claims(note, frame["reliability"]["unsupported_claims"])
            yield _sse("stage", frame)
    finally:
        stop.set()  # cancel / disconnect: stop at the next stage boundary
        shutil.rmtree(audio_path.parent, ignore_errors=True)


@app.post("/api/analyze")
async def analyze(audio: UploadFile) -> StreamingResponse:
    """Run the real pipeline on an uploaded consultation, streaming each stage."""
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '{suffix or 'unknown'}'. "
            f"Use one of: {', '.join(sorted(ALLOWED_AUDIO))}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="clarion-"))
    dest = tmp_dir / f"upload{suffix}"
    size = 0
    with dest.open("wb") as fh:
        while chunk := await audio.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail="Audio file too large (max 200 MB).")
            fh.write(chunk)
    if size == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Empty audio file.")

    cfg = POOL.cfg if POOL else load_config()
    components = POOL.components() if POOL else {}
    timeout_s = float(load_config().get("demo", {}).get("stage_timeout_s", 300))

    return StreamingResponse(
        _analyze_stream(dest, cfg, components, timeout_s),
        media_type="text/event-stream",
        headers={
            # `no-transform` is the load-bearing one: browsers send
            # `Accept-Encoding: gzip`, and the dev proxy will happily gzip an
            # event stream — which buffers every frame until the end and makes
            # a live run look frozen, then teleport. It tells any intermediary
            # to leave the bytes alone. X-Accel-Buffering covers nginx.
            "Cache-Control": "no-cache, no-transform",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health() -> dict:
    bands = load_config()["reliability"]
    return {
        "ok": True,
        "samples": len(_sample_files()),
        "warm": bool(POOL and POOL.ready),
        "demo_asr": POOL.asr_model if POOL else None,
        # the UI scales the verdict ring against these — read from config so the
        # ring cannot drift out of step with the thresholds it depicts
        "reliability": {
            "reliable_max": int(bands["reliable_max"]),
            "unreliable_min": int(bands["unreliable_min"]),
        },
    }
