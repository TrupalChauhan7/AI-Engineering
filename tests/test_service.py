"""run_pipeline: shape, verdict bands, and the structural leak guarantee.

Everything here is mocked — no Whisper, no Ollama, no network. The point is the
wiring and the contract the future FastAPI route depends on, not model quality.
"""

import json
import subprocess
import sys

import pytest

from s2n.evaluation.claim_verifier import ClaimReport, OmissionReport
from s2n.reliability.flagger import verdict
from s2n.service import run_pipeline

CFG = {"reliability": {"reliable_max": 7, "unreliable_min": 12}}


class FakeTranscriber:
    def transcribe(self, audio_path):
        return f"Doctor: transcript of {audio_path}"


class FakeGenerator:
    """Mimics NoteGenerator.generate(...).note"""

    def __init__(self, note="S: cough x3/7. O: chest clear. A: URTI. P: rest."):
        self.note = note

    def generate(self, transcript, consultation_id=None):
        return type("GeneratedNote", (), {"note": self.note})()


class FakeVerifier:
    """Returns the real report dataclasses, so the shape under test is genuine."""

    def __init__(self, verdicts=("supported", "contradicted"), omissions=("present", "absent")):
        self._verdicts = list(verdicts)
        self._omissions = list(omissions)

    def score(self, transcript, note):
        claims = [f"claim {i}" for i in range(len(self._verdicts))]
        return ClaimReport(claims, self._verdicts)

    def decompose_transcript(self, transcript):
        return [f"fact {i}" for i in range(len(self._omissions))]

    def check_omissions(self, note, facts):
        return OmissionReport(list(facts), self._omissions)


def _run(**kw):
    defaults = dict(
        audio_path="sample.wav",
        cfg=CFG,
        transcriber=FakeTranscriber(),
        generator=FakeGenerator(),
        verifier=FakeVerifier(),
    )
    return run_pipeline(**{**defaults, **kw})


# --- shape / contract ---------------------------------------------------


def test_returns_the_documented_shape():
    out = _run()
    assert set(out) == {"transcript", "note", "reliability", "timings"}
    assert set(out["reliability"]) == {
        "unsupported_claims",
        "omitted_facts",
        "n_unsupported",
        "n_omitted",
        "combined",
        "verdict",
        "score_note",
    }
    assert set(out["timings"]) == {"transcribe_s", "generate_s", "reliability_s"}


def test_result_is_json_serialisable():
    """A FastAPI route must be able to `return` this untouched."""
    out = _run()
    assert json.loads(json.dumps(out)) == out


def test_transcribes_audio_but_skips_asr_when_transcript_given():
    assert _run()["transcript"] == "Doctor: transcript of sample.wav"
    # a supplied transcript wins, and ASR is never called
    out = _run(audio_path=None, transcript="Doctor: pre-cached transcript here.", transcriber=None)
    assert out["transcript"] == "Doctor: pre-cached transcript here."
    assert out["timings"]["transcribe_s"] >= 0.0


def test_requires_audio_or_transcript():
    with pytest.raises(ValueError):
        run_pipeline(cfg=CFG)


def test_reports_the_offending_claims_and_facts():
    """The reliability must be explainable — the 'why', not just a number."""
    out = _run(
        verifier=FakeVerifier(
            verdicts=("supported", "contradicted", "not_mentioned"),
            omissions=("present", "absent", "absent"),
        )
    )["reliability"]
    assert out["unsupported_claims"] == ["claim 1", "claim 2"]  # non-"supported"
    assert out["omitted_facts"] == ["fact 1", "fact 2"]  # "absent"
    assert out["n_unsupported"] == 2 and out["n_omitted"] == 2
    assert out["combined"] == 4


# --- verdict bands ------------------------------------------------------


@pytest.mark.parametrize(
    "combined,expected",
    [(0, "reliable"), (7, "reliable"), (8, "review"), (11, "review"), (12, "unreliable")],
)
def test_verdict_bands_on_synthetic_counts(combined, expected):
    assert verdict({"n_unsupported": combined, "n_omitted": 0}, CFG)["verdict"] == expected


def test_verdict_flows_through_the_pipeline():
    out = _run(verifier=FakeVerifier(verdicts=("contradicted",) * 12, omissions=()))
    assert out["reliability"]["combined"] == 12
    assert out["reliability"]["verdict"] == "unreliable"
    assert "UNRELIABLE" in out["reliability"]["score_note"]


# --- leak safety (project rule 1) -------------------------------------


def test_service_path_never_loads_the_answer_key():
    """Importing the service must not pull in any gold-note / human-eval loader.

    Run in a clean interpreter so an unrelated test's imports cannot mask a real
    leak. The reliability's credibility rests on the generator+verifier path having no
    route to primock57/notes/ or primock57/human_eval_data/.
    """
    probe = (
        "import sys; import s2n.service; "
        "bad=[m for m in sys.modules if 'evaluation_data' in m or 'targets' in m]; "
        "print(','.join(bad))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "", f"service imported an answer-side module: {loaded}"


# --- hardening: refuse to fabricate a note from a non-consultation -------


def test_empty_transcript_is_refused_not_summarised():
    """Silence / a wrong file yields (near-)empty ASR; the pipeline must NOT
    invent a note from it — the core failure a reliability tool prevents."""
    from s2n.service import EmptyTranscriptError

    with pytest.raises(EmptyTranscriptError):
        _run(audio_path=None, transcript="   ", transcriber=None)


def test_too_short_transcript_is_refused():
    from s2n.service import EmptyTranscriptError

    with pytest.raises(EmptyTranscriptError):
        _run(audio_path=None, transcript="uh ok", transcriber=None)  # 2 words


def test_a_real_transcript_still_runs():
    out = _run(
        audio_path=None,
        transcript="Doctor: what brings you in today? Patient: a bad cough for three days.",
        transcriber=None,
    )
    assert out["note"] and out["reliability"]["verdict"]
