"""RunStore: the audit-trail contract — record, get, list, stats.

No models, no network. A temporary SQLite file per test (tmp_path) keeps each
test isolated and the real results/runs.db untouched. The point is the durable
record's shape and provenance, not model quality.
"""

import json

import pytest

from s2n.store import RunStore

# A resolved config, shaped like load_config() output for the meetings domain,
# and a pipeline result, shaped like s2n.service.run_pipeline() output.
CFG = {
    "active_domain": "meetings",
    "domain": "meetings",
    "llm": {"model": "qwen3:14b"},
    "generation": {"prompt_version": "meetings_v1"},
    "claim_verifier": {
        "model": "llama3.1:8b",
        "verify_prompt": "claim_verify_v2",
        "omission_verify_prompt": "omission_verify_v2",
        "transcript_facts_prompt": "transcript_facts_meetings_v1",
    },
    "reliability": {"reliable_max": 5, "unreliable_min": 9},
}

RESULT = {
    "transcript": "Chair: the motion passes 4 to 1.",
    "note": "Summary: a motion passed.",
    "reliability": {
        "unsupported_claims": ["the budget was $2m"],
        "omitted_facts": ["the vote was 4 to 1", "the deadline is March"],
        "n_unsupported": 1,
        "n_omitted": 2,
        "combined": 3,
        "verdict": "reliable",
        "score_note": "RELIABLE (combined=3)",
    },
    "timings": {"transcribe_s": 0.0, "generate_s": 1.2, "reliability_s": 3.4},
}


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs.db")


def _record(store, **kw):
    defaults = dict(result=RESULT, cfg=CFG, source="transcript", source_name="council.txt")
    return store.record(**{**defaults, **kw})


# --- round-trip: record -> get ------------------------------------------


def test_record_returns_an_id_and_get_round_trips(store):
    run_id = _record(store)
    assert isinstance(run_id, str) and run_id

    rec = store.get(run_id)
    # payload survives verbatim
    assert rec["transcript"] == RESULT["transcript"]
    assert rec["note"] == RESULT["note"]
    assert rec["verdict"] == "reliable"
    assert rec["combined"] == 3
    # JSON columns come back as real lists/dicts, not strings
    assert rec["unsupported_claims"] == ["the budget was $2m"]
    assert rec["omitted_facts"] == ["the vote was 4 to 1", "the deadline is March"]
    assert rec["timings"] == RESULT["timings"]


def test_provenance_is_captured_from_config(store):
    """The reason the store exists: a run must be reproducible from its record."""
    rec = store.get(_record(store))
    assert rec["domain"] == "meetings"
    assert rec["generator_model"] == "qwen3:14b"
    assert rec["verifier_model"] == "llama3.1:8b"
    assert rec["generation_prompt"] == "meetings_v1"
    assert rec["verify_prompt"] == "claim_verify_v2"
    assert rec["omission_verify_prompt"] == "omission_verify_v2"
    assert rec["transcript_facts_prompt"] == "transcript_facts_meetings_v1"
    assert rec["reliable_max"] == 5 and rec["unreliable_min"] == 9
    assert rec["source"] == "transcript" and rec["source_name"] == "council.txt"
    assert rec["created_at"]  # a timestamp was stamped


def test_get_unknown_id_returns_none(store):
    assert store.get("does-not-exist") is None


def test_audio_source_stores_no_transcript_leak_of_audio(store):
    """Audio is never a stored column — only its label."""
    rec = store.get(_record(store, source="audio", source_name="visit.wav"))
    assert rec["source"] == "audio" and rec["source_name"] == "visit.wav"
    assert "audio" not in rec  # no audio bytes/path column exists


# --- list / count / stats -----------------------------------------------


def test_list_is_newest_first_and_light(store):
    a = store.record(
        result=RESULT, cfg=CFG, source="transcript", created_at="2026-01-01T00:00:00+00:00"
    )
    b = store.record(
        result=RESULT, cfg=CFG, source="transcript", created_at="2026-02-01T00:00:00+00:00"
    )
    rows = store.list()
    assert [r["run_id"] for r in rows] == [b, a]  # newest first
    # summary rows must not carry the heavy fields
    assert "transcript" not in rows[0] and "note" not in rows[0]
    assert rows[0]["verdict"] == "reliable"


def test_list_limit_and_domain_filter(store):
    _record(store)  # meetings
    clinical_cfg = {**CFG, "active_domain": "clinical", "domain": "clinical"}
    store.record(result=RESULT, cfg=clinical_cfg, source="transcript")
    assert store.count() == 2
    assert len(store.list(limit=1)) == 1
    meetings_only = store.list(domain="meetings")
    assert len(meetings_only) == 1 and meetings_only[0]["domain"] == "meetings"


def test_stats_aggregates_by_verdict_and_domain(store):
    _record(store)  # reliable / meetings
    unrel = {**RESULT, "reliability": {**RESULT["reliability"], "verdict": "unreliable"}}
    store.record(result=unrel, cfg=CFG, source="transcript")
    s = store.stats()
    assert s["total"] == 2
    assert s["by_verdict"] == {"reliable": 1, "unreliable": 1}
    assert s["by_domain"] == {"meetings": 2}


def test_stored_record_is_json_serialisable(store):
    """A FastAPI route must be able to return a record untouched."""
    rec = store.get(_record(store))
    assert json.loads(json.dumps(rec)) == rec


# --- leak safety (project rule 1) ---------------------------------------


def test_store_module_never_loads_the_answer_key():
    import subprocess
    import sys

    probe = (
        "import sys; import s2n.store; "
        "bad=[m for m in sys.modules if 'evaluation_data' in m or 'targets' in m]; "
        "print(','.join(bad))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "", f"store imported an answer-side module: {loaded}"


# --- observability: request-id correlation + latency stats ---------------


def test_request_id_is_stored_and_summarised(store):
    rid = store.record(result=RESULT, cfg=CFG, source="audio", request_id="req-abc123")
    assert store.get(rid)["request_id"] == "req-abc123"
    assert store.list()[0]["request_id"] == "req-abc123"


def test_latency_stats_over_recorded_runs(store):
    for g in (10.0, 20.0, 30.0):
        r = {**RESULT, "timings": {"transcribe_s": 1.0, "generate_s": g, "reliability_s": 5.0}}
        store.record(result=r, cfg=CFG, source="transcript")
    lat = store.latency_stats()
    assert lat["generate_s"]["n"] == 3
    assert lat["generate_s"]["mean"] == 20.0
    assert lat["generate_s"]["p50"] == 20.0
    assert lat["generate_s"]["p90"] == 28.0
    assert lat["transcribe_s"]["mean"] == 1.0


def test_latency_stats_on_empty_store_is_all_none(store):
    assert store.latency_stats()["generate_s"] == {"n": 0, "mean": None, "p50": None, "p90": None}


# --- hardening: unicode + large payloads round-trip ----------------------


def test_unicode_and_large_payloads_round_trip(store):
    big_note = "SOAP — café patient, 5 mg × 2 daily. " * 500  # unicode + long
    result = {
        **RESULT,
        "note": big_note,
        "reliability": {**RESULT["reliability"], "unsupported_claims": ["naïve claim ✓"]},
    }
    rec = store.get(store.record(result=result, cfg=CFG, source="transcript"))
    assert rec["note"] == big_note
    assert rec["unsupported_claims"] == ["naïve claim ✓"]
