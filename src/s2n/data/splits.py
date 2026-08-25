"""Frozen dev/test split — BY CONSULTATION, seeded, written once.

WHY (project rule 2 + pre-registration §B5): we iterate the rubric on DEV and
score TEST exactly once. That discipline only holds if the split never moves, so
we generate it once (seeded) and persist it to disk. Every later run LOADS the
saved file; it is never regenerated. Splitting by *consultation* (not by note)
stops the 5 notes of one consultation from straddling dev/test — which would
leak information and break the cluster structure (effective n = 57).

Consultation IDs come from the transcripts folder via GenerationLoader — a
leak-safe source (never notes/ or human_eval/).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from s2n.config import ROOT, load_config
from s2n.data.generation_data import GenerationLoader


@dataclass
class Split:
    dev: list[str]
    test: list[str]
    seed: int

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "dev_size": len(self.dev),
            "test_size": len(self.test),
            "dev": self.dev,
            "test": self.test,
        }


def make_split(consultation_ids: list[str], dev_size: int, seed: int) -> Split:
    """Deterministically choose ``dev_size`` consultations for DEV, rest TEST."""
    ids = sorted(consultation_ids)  # sort first so order can't affect the draw
    rng = random.Random(seed)
    dev = sorted(rng.sample(ids, dev_size))
    test = sorted(set(ids) - set(dev))
    return Split(dev=dev, test=test, seed=seed)


def load_or_create_split(cfg: dict | None = None) -> Split:
    """Load the frozen split if it exists; otherwise create and persist it once."""
    cfg = cfg or load_config()
    path = ROOT / cfg["paths"]["splits"]

    if path.exists():
        data = json.loads(path.read_text())
        return Split(dev=data["dev"], test=data["test"], seed=data["seed"])

    ids = GenerationLoader.from_config(cfg).consultation_ids()
    dev_size = cfg["split"]["dev_size"]
    expected_total = dev_size + cfg["split"]["test_size"]
    if len(ids) != expected_total:
        raise ValueError(
            f"Expected {expected_total} consultations (config split), found {len(ids)}."
        )
    split = make_split(ids, dev_size=dev_size, seed=cfg["seed"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.as_dict(), indent=2) + "\n")
    return split
