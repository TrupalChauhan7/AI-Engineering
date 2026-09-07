"""Clarion API: sample replay, SSE streaming, upload validation, claim spans.

No models run here — the pipeline is monkeypatched. These tests pin the wire
contract the frontend is built against.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import main as api
from app.api.spans import locate_claims

NOTE = (
    "S: 3/7 hx of watery diarrhea. No blood.\n"
    "O: Abdomen soft, non-tender.\n"
    "A: Likely viral gastroenteritis.\n"
    "P: Fluids, safety-net advice."
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "SAMPLES_DIR", tmp_path)
    return TestClient(api.app)


def _write_sample(dir_, sid="day1_consultation02", **over):
    data = {
        "id": sid,
        "label": "Day 1 · Consultation 02",
        "durationLabel": "7:12",
        "transcript": "Doctor: what brings you in?",
        "note": NOTE,
        "reliability": {
            "unsupported_claims": ["Patient denies blood in stool"],
            "omitted_facts": ["Patient is taking ibuprofen"],
            "n_unsupported": 1,
            "n_omitted": 1,
            "combined": 2,
            "verdict": "reliable",
            "score_note": "…",
        },
        "timings": {"transcribe_s": 1.0, "generate_s": 2.0, "reliability_s": 3.0},
        **over,
    }
    (dir_ / f"{sid}.json").write_text(json.dumps(data))
    return data


# --- samples ------------------------------------------------------------


def test_samples_empty_when_cache_missing(client):
    assert client.get("/api/samples").json() == []


def test_samples_lists_and_replays(client, tmp_path):
    _write_sample(tmp_path)
    listing = client.get("/api/samples").json()
    assert listing == [
        {"id": "day1_consultation02", "label": "Day 1 · Consultation 02", "durationLabel": "7:12"}
    ]
    full = client.get("/api/samples/day1_consultation02").json()
    assert full["reliability"]["verdict"] == "reliable"
    assert full["note"] == NOTE


def test_unknown_sample_is_404_with_guidance(client):
    r = client.get("/api/samples/nope")
    assert r.status_code == 404 and "build_samples" in r.json()["detail"]


def test_sample_id_cannot_escape_the_cache_dir(client, tmp_path):
    """A traversal attempt must not read outside the samples folder."""
    (tmp_path.parent / "secret.json").write_text('{"x":1}')
    assert client.get("/api/samples/..%2Fsecret").status_code in (307, 404)


# --- analyze (SSE) ------------------------------------------------------


def _fake_staged(**_kw):
    yield {"stage": "transcribe", "status": "running"}
    yield {"stage": "transcribe", "status": "done", "transcript": "Doctor: hello"}
    yield {"stage": "generate", "status": "running"}
    yield {"stage": "generate", "status": "done", "note": NOTE}
    yield {"stage": "reliability", "status": "running"}
    yield {
        "stage": "reliability",
        "status": "done",
        "reliability": {
            "unsupported_claims": ["No blood in the stool was reported"],
            "omitted_facts": ["ibuprofen"],
            "n_unsupported": 1,
            "n_omitted": 1,
            "combined": 2,
            "verdict": "reliable",
            "score_note": "…",
        },
    }
    yield {"stage": "done", "timings": {"transcribe_s": 1, "generate_s": 2, "reliability_s": 3}}


def _parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(ln.split(": ", 1) for ln in block.splitlines() if ": " in ln)
        events.append((lines.get("event"), json.loads(lines["data"])))
    return events


def test_analyze_streams_every_stage_in_order(client, monkeypatch):
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)
    r = client.post("/api/analyze", files={"audio": ("c.wav", b"RIFFfake", "audio/wav")})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    assert [e for e, _ in events] == ["stage"] * 6 + ["done"]
    # the `done` frame carries only timings — no "stage" key (brief §4)
    assert [d.get("stage") for _, d in events[:-1]] == [
        "transcribe",
        "transcribe",
        "generate",
        "generate",
        "reliability",
        "reliability",
    ]
    _, done = events[-1]
    assert set(done["timings"]) == {"transcribe_s", "generate_s", "reliability_s"}


def test_analyze_attaches_spans_for_the_scan_animation(client, monkeypatch):
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)
    r = client.post("/api/analyze", files={"audio": ("c.wav", b"RIFFfake", "audio/wav")})
    reliability_done = [
        d for e, d in _parse_sse(r.text) if d.get("stage") == "reliability" and "reliability" in d
    ][0]
    span = reliability_done["spans"][0]
    # sentence granularity: the flag lands on "No blood.", not the whole S line
    assert span["line"] == 1
    assert NOTE[span["start"] : span["end"]] == "No blood."


def test_analyze_reports_pipeline_failure_as_an_error_event(client, monkeypatch):
    def boom(**_kw):
        yield {"stage": "transcribe", "status": "running"}
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(api, "run_pipeline_staged", boom)
    r = client.post("/api/analyze", files={"audio": ("c.wav", b"RIFFfake", "audio/wav")})
    events = _parse_sse(r.text)
    assert events[-1][0] == "error" and "whisper exploded" in events[-1][1]["message"]


@pytest.mark.parametrize("name", ["notes.txt", "scan.pdf", "noext"])
def test_analyze_rejects_non_audio(client, name):
    r = client.post("/api/analyze", files={"audio": (name, b"x", "application/octet-stream")})
    assert r.status_code == 400


def test_analyze_rejects_empty_audio(client):
    r = client.post("/api/analyze", files={"audio": ("c.wav", b"", "audio/wav")})
    assert r.status_code == 400 and "Empty" in r.json()["detail"]


# --- claim -> span mapping ---------------------------------------------


def test_locate_claims_matches_the_right_sentence():
    spans = locate_claims(NOTE, ["Abdomen was soft and non-tender on examination"])
    assert spans[0]["line"] == 2
    assert NOTE[spans[0]["start"] : spans[0]["end"]] == "O: Abdomen soft, non-tender."


def test_spans_are_sentence_sized_not_whole_sections():
    """A generated note puts a whole SOAP section on one line; a single flagged
    claim must not paint the entire section red."""
    note = (
        "**Subjective:** Patient reports a cough. Denies fever. States he is 50. "
        "No chest pain reported."
    )
    s = locate_claims(note, ["The patient denies having a fever"])[0]
    assert s["start"] >= 0
    assert note[s["start"] : s["end"]] == "Denies fever."
    assert (s["end"] - s["start"]) < len(note) / 2


def test_locate_claims_reports_no_match_rather_than_guessing():
    spans = locate_claims(NOTE, ["Patient underwent a coronary angiogram in 2019"])
    assert spans[0]["start"] == -1 and spans[0]["line"] == -1


def test_locate_claims_is_span_accurate_for_every_returned_match():
    claims = ["watery diarrhoea for three days", "viral gastroenteritis suspected", "give fluids"]
    for s in locate_claims(NOTE, claims):
        if s["start"] >= 0:
            assert NOTE[s["start"] : s["end"]].strip() == NOTE[s["start"] : s["end"]]


# --- demo lane: warm pool, timeout, cancellation -------------------------


def test_demo_config_swaps_only_the_asr_model():
    """The demo must never touch the sealed research settings."""
    from app.api.warmup import demo_config

    cfg = {
        "transcription": {"model": "large-v3-turbo", "language": "en", "temperature": [0.0, 0.2]},
        "demo": {"transcription_model": "base"},
        "judge": {"model": "atla/selene-mini"},
        "llm": {"num_ctx": 8192, "temperature": 0},
        "seed": 42,
    }
    d = demo_config(cfg)
    assert d["transcription"]["model"] == "base"
    # everything else identical, and the ORIGINAL is untouched
    assert cfg["transcription"]["model"] == "large-v3-turbo"
    assert d["transcription"]["temperature"] == cfg["transcription"]["temperature"]
    assert d["judge"] == cfg["judge"] and d["llm"] == cfg["llm"] and d["seed"] == 42


def test_demo_config_is_a_noop_without_the_demo_block():
    from app.api.warmup import demo_config

    cfg = {"transcription": {"model": "large-v3-turbo"}}
    assert demo_config(cfg)["transcription"]["model"] == "large-v3-turbo"


def test_stage_timeout_reports_an_error_instead_of_hanging(tmp_path, monkeypatch):
    """A stage that overruns must end the stream with a clear message."""
    import time as _t

    def stuck(**_kw):
        yield {"stage": "transcribe", "status": "running"}
        _t.sleep(5)  # never finishes within the timeout

    monkeypatch.setattr(api, "run_pipeline_staged", stuck)
    audio = tmp_path / "d" / "a.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x")

    frames = list(api._analyze_stream(audio, {}, {}, timeout_s=1.5))
    events = _parse_sse("".join(frames))
    assert events[-1][0] == "error"
    assert "transcribe" in events[-1][1]["message"]
    assert "longer than" in events[-1][1]["message"]


def test_heartbeats_prove_liveness_during_a_slow_stage(tmp_path, monkeypatch):
    import time as _t

    def slow(**_kw):
        yield {"stage": "transcribe", "status": "running"}
        _t.sleep(2.4)
        yield {"stage": "transcribe", "status": "done", "transcript": "hi"}

    monkeypatch.setattr(api, "run_pipeline_staged", slow)
    audio = tmp_path / "d" / "a.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x")

    events = _parse_sse("".join(api._analyze_stream(audio, {}, {}, timeout_s=30)))
    beats = [d for e, d in events if e == "heartbeat"]
    assert len(beats) >= 2
    assert beats[0]["stage"] == "transcribe"
    assert beats[-1]["elapsed_s"] > beats[0]["elapsed_s"]


def test_abandoning_the_stream_cleans_up_the_upload(tmp_path, monkeypatch):
    """Closing the generator early (cancel/disconnect) must remove the temp dir."""
    import time as _t

    def slow(**_kw):
        yield {"stage": "transcribe", "status": "running"}
        _t.sleep(10)

    monkeypatch.setattr(api, "run_pipeline_staged", slow)
    audio = tmp_path / "d" / "a.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x")

    gen = api._analyze_stream(audio, {}, {}, timeout_s=30)
    next(gen)  # first heartbeat
    gen.close()  # what Starlette does when the client disconnects
    assert not audio.parent.exists()


def test_warm_pool_passes_its_components_into_the_pipeline(tmp_path, monkeypatch):
    """The warmed instances must actually be reused, or every upload reloads."""
    seen = {}

    def capture(**kw):
        seen.update(kw)
        yield {"stage": "done", "timings": {"transcribe_s": 0, "generate_s": 0, "reliability_s": 0}}

    monkeypatch.setattr(api, "run_pipeline_staged", capture)
    audio = tmp_path / "d" / "a.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x")

    components = {"transcriber": "WARM_ASR", "generator": "WARM_GEN", "verifier": "WARM_VER"}
    list(api._analyze_stream(audio, {"x": 1}, components, timeout_s=30))
    assert seen["transcriber"] == "WARM_ASR"
    assert seen["generator"] == "WARM_GEN"
    assert seen["verifier"] == "WARM_VER"


# --- Track B: the training command must not drift from the recorded run ---


def test_finetune_command_matches_the_recorded_adapter_config():
    """The scripted LoRA command must still match what actually produced the adapter.

    The Phase-15b result is only reproducible if this command is the one MLX
    recorded beside the weights — so a silent edit to either should fail here.
    """
    import importlib.util
    from pathlib import Path as _P

    path = _P(__file__).resolve().parents[1] / "pipelines" / "12_finetune_train.py"
    spec = importlib.util.spec_from_file_location("finetune_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not mod.RECORDED_CONFIG.is_file():
        pytest.skip("adapter artefacts not present (results/ is gitignored)")
    assert mod.verify_against_recorded_run() == []
    # and the flags the log calls out by name are actually there
    joined = " ".join(mod.COMMAND)
    for expected in ("--train", "--num-layers 16", "--batch-size 2", "--iters 1500"):
        assert expected in joined


# --- audit trail: persistence + read routes ------------------------------
#
# The store is injected as a temp DB (never the real results/runs.db). The
# mocked pipeline (_fake_staged) drives a full run, so we can assert the
# completed run was recorded, with its provenance, and served back.


@pytest.fixture
def audited_client(client, tmp_path, monkeypatch):
    """A client whose module-level STORE points at a throwaway database."""
    from s2n.store import RunStore

    store = RunStore(tmp_path / "audit.db")
    monkeypatch.setattr(api, "STORE", store)
    return client, store


def test_analyze_records_the_completed_run(audited_client, monkeypatch):
    client, store = audited_client
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)

    r = client.post("/api/analyze", files={"audio": ("visit.wav", b"RIFFfake", "audio/wav")})
    # drain the stream so the run completes and _record_run fires
    _parse_sse(r.text)

    assert store.count() == 1
    rec = store.get(store.list()[0]["run_id"])
    assert rec["verdict"] == "reliable"
    assert rec["source"] == "audio" and rec["source_name"] == "visit.wav"
    assert rec["note"] == NOTE
    assert rec["unsupported_claims"] == ["No blood in the stool was reported"]
    # provenance was captured from the resolved config
    assert rec["generator_model"] and rec["verifier_model"]
    assert rec["domain"]


def test_analyze_does_not_record_a_failed_run(audited_client, monkeypatch):
    client, store = audited_client

    def boom(**_kw):
        yield {"stage": "transcribe", "status": "running"}
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(api, "run_pipeline_staged", boom)
    r = client.post("/api/analyze", files={"audio": ("c.wav", b"RIFFfake", "audio/wav")})
    _parse_sse(r.text)
    assert store.count() == 0  # nothing to audit without a verdict


def test_runs_routes_list_get_and_stats(audited_client, monkeypatch):
    client, store = audited_client
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)
    _parse_sse(client.post("/api/analyze", files={"audio": ("a.wav", b"RIFF", "audio/wav")}).text)

    listing = client.get("/api/runs").json()
    assert len(listing) == 1 and listing[0]["verdict"] == "reliable"
    # the list stays light — no transcript/note in summary rows
    assert "transcript" not in listing[0] and "note" not in listing[0]

    run_id = listing[0]["run_id"]
    full = client.get(f"/api/runs/{run_id}").json()
    assert full["note"] == NOTE and full["source_name"] == "a.wav"

    stats = client.get("/api/stats").json()
    assert stats["total"] == 1 and stats["by_verdict"] == {"reliable": 1}


def test_unknown_run_is_404(audited_client):
    client, _ = audited_client
    assert client.get("/api/runs/nope").status_code == 404


def test_audit_routes_are_503_when_store_uninitialised(client, monkeypatch):
    """Without lifespan (STORE is None) the read routes fail cleanly, not with a 500."""
    monkeypatch.setattr(api, "STORE", None)
    assert client.get("/api/runs").status_code == 503
    assert client.get("/api/runs/x").status_code == 503
    assert client.get("/api/stats").status_code == 503


# --- domain-aware routing ------------------------------------------------
#
# The pipeline is domain-general; the API must let a caller pick the domain per
# request. Constructing a NoteGenerator/ClaimVerifier is offline (models load
# only when actually called), so these run without a network or warm pool.


def test_components_for_reuses_warm_default_and_builds_other_domain():
    from app.api.warmup import WarmPool
    from s2n.config import load_config, load_config_for_domain
    from s2n.evaluation.claim_verifier import ClaimVerifier
    from s2n.generation.generator import NoteGenerator

    pool = WarmPool(load_config())  # __init__ only — no warm() (no models loaded)
    pool.transcriber, pool.generator, pool.verifier = "ASR", "GEN", "VER"

    # the warm default domain reuses the warm trio verbatim
    same = pool.components_for(load_config_for_domain(pool.domain))
    assert same == {"transcriber": "ASR", "generator": "GEN", "verifier": "VER"}

    # a different domain reuses the ASR but builds its own generator + verifier
    other = "meetings" if pool.domain != "meetings" else "clinical"
    built = pool.components_for(load_config_for_domain(other))
    assert built["transcriber"] == "ASR"  # ASR is domain-independent
    assert isinstance(built["generator"], NoteGenerator)
    assert isinstance(built["verifier"], ClaimVerifier)

    # cached: a second request in that domain reuses the same instances
    again = pool.components_for(load_config_for_domain(other))
    assert again["generator"] is built["generator"]
    assert again["verifier"] is built["verifier"]


def test_domains_route_lists_what_the_server_can_run(client):
    d = client.get("/api/domains").json()
    assert set(d["available"]) >= {"clinical", "meetings"}
    assert d["default"]  # a default domain is always resolved


def test_analyze_rejects_an_unknown_domain(client):
    r = client.post(
        "/api/analyze",
        files={"audio": ("c.wav", b"RIFFfake", "audio/wav")},
        data={"domain": "banana"},
    )
    assert r.status_code == 400 and "Unknown domain" in r.json()["detail"]


def test_analyze_records_the_requested_domain(audited_client, monkeypatch):
    """A meetings request must run — and be audited as — the meetings stack."""
    client, store = audited_client
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)

    r = client.post(
        "/api/analyze",
        files={"audio": ("m.wav", b"RIFFfake", "audio/wav")},
        data={"domain": "meetings"},
    )
    _parse_sse(r.text)

    rec = store.get(store.list()[0]["run_id"])
    assert rec["domain"] == "meetings"
    assert rec["generator_model"] == "qwen3:14b"
    assert rec["verify_prompt"] == "claim_verify_v2"


# --- observability: request id, metrics ----------------------------------


def test_every_response_carries_a_request_id_header(client):
    assert client.get("/api/health").headers.get("X-Request-ID")


def test_analyze_stores_the_request_id_from_the_header(audited_client, monkeypatch):
    """The audited run must be findable from the request's trace id."""
    client, store = audited_client
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)

    r = client.post("/api/analyze", files={"audio": ("a.wav", b"RIFFfake", "audio/wav")})
    _parse_sse(r.text)  # drain so the run is recorded

    rec = store.get(store.list()[0]["run_id"])
    assert rec["request_id"] == r.headers["X-Request-ID"]


def test_metrics_reports_counts_and_per_stage_latency(audited_client, monkeypatch):
    client, store = audited_client
    monkeypatch.setattr(api, "run_pipeline_staged", _fake_staged)
    _parse_sse(client.post("/api/analyze", files={"audio": ("a.wav", b"RIFF", "audio/wav")}).text)

    m = client.get("/api/metrics").json()
    assert m["total"] == 1 and m["by_verdict"] == {"reliable": 1}
    assert set(m["latency_s"]) == {"transcribe_s", "generate_s", "reliability_s"}
    assert m["latency_s"]["generate_s"]["n"] == 1


def test_metrics_is_503_when_store_uninitialised(client, monkeypatch):
    monkeypatch.setattr(api, "STORE", None)
    assert client.get("/api/metrics").status_code == 503
