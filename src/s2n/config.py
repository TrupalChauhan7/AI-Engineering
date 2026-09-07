"""Load project settings from config/config.yaml.

WHY: every module reads settings from ONE place. No hardcoded paths or
model names anywhere else. Change an experiment by editing config.yaml.

Phase 2 — domains: a ``domain`` selector plus ``domains`` profiles let the SAME
pipeline run in different domains (clinical / meetings) by overriding the
generator model and the generator/verifier prompts. ``domain: clinical``
reproduces the academic setup exactly (behaviour-preserving). The ``S2N_DOMAIN``
env var overrides the file's ``domain`` for a single run. With no ``domains``
block the config is returned unchanged (back-compat).
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # -> project/
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return the whole config as a dict, with the active domain profile applied."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _apply_domain(cfg)


def _apply_env_overrides(cfg: dict) -> None:
    """Apply deployment overrides that must come from the environment, in place.

    ``S2N_OLLAMA_HOST`` repoints the LLM backend without editing the config —
    the one setting that changes between machines (localhost in dev, e.g.
    ``http://host.docker.internal:11434`` from a container, or a remote host).
    Kept separate from domain logic because it is about WHERE the models run,
    not WHICH models.
    """
    host = os.environ.get("S2N_OLLAMA_HOST")
    if host and "llm" in cfg:
        cfg["llm"]["host"] = host


def _apply_domain(cfg: dict) -> dict:
    """Overlay the active ``domain`` profile's overrides onto the base config.

    The PRODUCT is clinical-only and never sets ``S2N_DOMAIN``, so this resolves
    to the clinical profile — which is behaviour-preserving (its overrides equal
    the base values, i.e. the academic setup). The only caller that sets
    ``S2N_DOMAIN`` is the meetings *evaluation* harness (scripts/), which uses
    this to reproduce the generalisation study; that path is research, not the
    product.
    """
    _apply_env_overrides(cfg)
    name = os.environ.get("S2N_DOMAIN") or cfg.get("domain")
    profile = (cfg.get("domains") or {}).get(name)
    if not profile:
        return cfg
    if "generator_model" in profile:
        cfg["llm"]["model"] = profile["generator_model"]
    if "generation_prompt" in profile:
        cfg["generation"]["prompt_version"] = profile["generation_prompt"]
    if "verify_prompt" in profile:
        cfg["claim_verifier"]["verify_prompt"] = profile["verify_prompt"]
    if "omission_verify_prompt" in profile:
        cfg["claim_verifier"]["omission_verify_prompt"] = profile["omission_verify_prompt"]
    if "transcript_facts_prompt" in profile:
        cfg["claim_verifier"]["transcript_facts_prompt"] = profile["transcript_facts_prompt"]
    if "reliable_max" in profile:
        cfg["reliability"]["reliable_max"] = profile["reliable_max"]
    if "unreliable_min" in profile:
        cfg["reliability"]["unreliable_min"] = profile["unreliable_min"]
    cfg["active_domain"] = name
    return cfg
