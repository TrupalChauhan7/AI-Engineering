"""Claim verifier: decomposition parsing, verdict alignment + fallback,
aggregation. Mocked client — no live LLM."""

import json

from s2n.evaluation.claim_verifier import ClaimReport, ClaimVerifier
from s2n.llm.client import LLMResponse


class _ScriptedClient:
    """Returns queued responses in order; records the schemas it was asked for."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, *a, **k):
        self.calls.append(k)
        return LLMResponse(text=self.responses.pop(0), raw={})


def _cfg():
    return {
        "claim_verifier": {
            "model": "m",
            "decompose_prompt": "claim_decompose_v1",
            "verify_prompt": "claim_verify_v1",
            "transcript_facts_prompt": "transcript_facts_v2",
            "omission_verify_prompt": "omission_verify_v1",
            "criticality_prompt": "claim_criticality_v1",
            "nli_unsupported_threshold": 0.5,
        }
    }


def _verifier(responses):
    return ClaimVerifier(cfg=_cfg(), client=_ScriptedClient(responses))


def test_happy_path_batched_verify():
    v = _verifier(
        [
            json.dumps({"claims": ["Patient has a cough", "No fever", "On insulin"]}),
            json.dumps({"verdicts": ["supported", "supported", "not_mentioned"]}),
        ]
    )
    r = v.score("transcript", "note")
    assert r.n_claims == 3 and r.verdicts[2] == "not_mentioned"
    assert r.n_unsupported == 1 and not r.used_fallback and r.parse_ok


def test_misaligned_batch_falls_back_to_per_claim():
    v = _verifier(
        [
            json.dumps({"claims": ["a", "b"]}),
            json.dumps({"verdicts": ["supported"]}),  # 1 verdict for 2 claims
            json.dumps({"verdict": "contradicted"}),  # per-claim fallback x2
            json.dumps({"verdict": "supported"}),
        ]
    )
    r = v.score("t", "n")
    assert r.used_fallback and r.verdicts == ["contradicted", "supported"]
    assert r.n_unsupported == 1


def test_invalid_verdict_value_triggers_fallback():
    v = _verifier(
        [
            json.dumps({"claims": ["a"]}),
            json.dumps({"verdicts": ["maybe"]}),  # not in enum
            json.dumps({"verdict": "supported"}),
        ]
    )
    r = v.score("t", "n")
    assert r.used_fallback and r.verdicts == ["supported"]


def test_empty_decomposition_is_flagged_not_crashed():
    r = _verifier(["not json"]).score("t", "n")
    assert not r.parse_ok and r.n_claims == 0


def test_aggregation_counts():
    r = ClaimReport(
        claims=["a", "b", "c", "d"],
        verdicts=["supported", "contradicted", "not_mentioned", "not_mentioned"],
    )
    assert r.count("supported") == 1
    assert r.n_unsupported == 3


# ---- omission axis ---------------------------------------------------------
def test_transcript_facts_and_omission_check():
    v = _verifier(
        [
            json.dumps({"facts": ["Cough for 3 days", "PMH asthma", "No fever"]}),
            json.dumps({"verdicts": ["present", "absent", "present"]}),
        ]
    )
    facts = v.decompose_transcript("transcript")
    r = v.check_omissions("note", facts)
    assert r.n_facts == 3 and r.n_omitted == 1 and not r.used_fallback


def test_omission_misaligned_batch_falls_back_per_fact():
    v = _verifier(
        [
            json.dumps({"verdicts": ["present"]}),  # 1 verdict for 2 facts
            json.dumps({"verdict": "absent"}),  # per-fact fallback x2
            json.dumps({"verdict": "present"}),
        ]
    )
    r = v.check_omissions("note", ["a", "b"])
    assert r.used_fallback and r.verdicts == ["absent", "present"] and r.n_omitted == 1


def test_omission_empty_facts_flagged():
    r = _verifier([]).check_omissions("note", [])
    assert not r.parse_ok and r.n_omitted == 0


# ---- exact-length schemas (the ops fix) + criticality ----------------------
def test_schemas_fix_array_length():
    from s2n.evaluation.claim_verifier import criticality_schema, verify_schema

    s = verify_schema(7)["properties"]["verdicts"]
    assert s["minItems"] == 7 and s["maxItems"] == 7
    c = criticality_schema(3)["properties"]["labels"]
    assert c["minItems"] == 3 and c["maxItems"] == 3


def test_criticality_counts_critical_labels():
    v = _verifier([json.dumps({"labels": ["critical", "minor", "critical"]})])
    assert v.rate_criticality("t", ["a", "b", "c"]) == 2


def test_criticality_empty_and_misaligned_return_zero():
    assert _verifier([]).rate_criticality("t", []) == 0
    v = _verifier([json.dumps({"labels": ["critical"]})])  # 1 label for 2 claims
    assert v.rate_criticality("t", ["a", "b"]) == 0
