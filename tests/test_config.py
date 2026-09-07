"""Domain resolution: the per-request selector the domain-aware API depends on.

Reads the real config.yaml, so these also pin the locked domain profiles
(clinical = the academic setup; meetings = the improved Phase-2 stack).
"""

import pytest

from s2n.config import available_domains, load_config, load_config_for_domain


def test_both_domains_are_available():
    assert set(available_domains(load_config())) >= {"clinical", "meetings"}


def test_clinical_resolves_to_the_academic_stack():
    c = load_config_for_domain("clinical")
    assert c["active_domain"] == "clinical"
    assert c["llm"]["model"] == "medgemma:4b"
    assert c["generation"]["prompt_version"] == "v1.0"
    assert c["claim_verifier"]["verify_prompt"] == "claim_verify_v1"
    assert c["claim_verifier"]["omission_verify_prompt"] == "omission_verify_v1"
    assert c["claim_verifier"]["transcript_facts_prompt"] == "transcript_facts_v2"
    assert (c["reliability"]["reliable_max"], c["reliability"]["unreliable_min"]) == (7, 12)


def test_meetings_resolves_to_the_phase2_stack():
    m = load_config_for_domain("meetings")
    assert m["active_domain"] == "meetings"
    assert m["llm"]["model"] == "qwen3:14b"
    assert m["generation"]["prompt_version"] == "meetings_v1"
    assert m["claim_verifier"]["verify_prompt"] == "claim_verify_v2"
    assert m["claim_verifier"]["omission_verify_prompt"] == "omission_verify_v2"
    assert m["claim_verifier"]["transcript_facts_prompt"] == "transcript_facts_meetings_v1"
    assert (m["reliability"]["reliable_max"], m["reliability"]["unreliable_min"]) == (5, 9)


def test_verifier_model_is_shared_across_domains():
    """Rule 6 + the warm-pool design both rely on this: same verifier family."""
    assert (
        load_config_for_domain("clinical")["claim_verifier"]["model"]
        == load_config_for_domain("meetings")["claim_verifier"]["model"]
    )


def test_none_domain_falls_back_to_the_file_default():
    assert load_config_for_domain(None)["active_domain"] == load_config()["active_domain"]


def test_unknown_domain_raises():
    with pytest.raises(ValueError):
        load_config_for_domain("banana")


def test_explicit_domain_beats_the_env_var(monkeypatch):
    """The server picks a domain per request; S2N_DOMAIN must not override it."""
    monkeypatch.setenv("S2N_DOMAIN", "clinical")
    assert load_config_for_domain("meetings")["active_domain"] == "meetings"
