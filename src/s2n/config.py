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


def _apply_domain(cfg: dict) -> dict:
    """Overlay the active ``domain`` profile's overrides onto the base config.

    Behaviour-preserving: with domain=clinical the overrides equal the base
    values, so the resolved config is identical to the academic setup.
    """
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
