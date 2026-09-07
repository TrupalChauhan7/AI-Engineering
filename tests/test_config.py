"""Config resolution: the clinical default + deployment env overrides.

The product is clinical-only. ``load_config()`` resolves the clinical profile,
which is behaviour-preserving (its overrides equal the base values — the
academic setup). The meetings profile still lives in config.yaml, but only the
generalisation *eval* scripts (research, not the product) ever select it via
``S2N_DOMAIN``.
"""

from s2n.config import load_config


def test_clinical_is_the_default_stack():
    c = load_config()
    assert c["llm"]["model"] == "medgemma:4b"
    assert c["generation"]["prompt_version"] == "v1.0"
    assert c["claim_verifier"]["verify_prompt"] == "claim_verify_v1"
    assert c["claim_verifier"]["omission_verify_prompt"] == "omission_verify_v1"
    assert c["claim_verifier"]["transcript_facts_prompt"] == "transcript_facts_v2"
    assert (c["reliability"]["reliable_max"], c["reliability"]["unreliable_min"]) == (7, 12)


def test_ollama_host_env_override(monkeypatch):
    """S2N_OLLAMA_HOST repoints the backend without editing config (Docker/remote)."""
    monkeypatch.setenv("S2N_OLLAMA_HOST", "http://host.docker.internal:11434")
    assert load_config()["llm"]["host"] == "http://host.docker.internal:11434"


def test_ollama_host_defaults_without_env(monkeypatch):
    monkeypatch.delenv("S2N_OLLAMA_HOST", raising=False)
    assert load_config()["llm"]["host"] == "http://localhost:11434"
