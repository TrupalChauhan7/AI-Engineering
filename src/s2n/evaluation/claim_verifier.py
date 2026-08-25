"""Claim-level faithfulness verifier — the RQ1 headline metric (Amendment 4).

STATUS: LOCKED and PRE-REGISTERED. Built exploratorily on DEV, then promoted to a
pre-registered TEST endpoint before the one-shot run (Amendment 4, 19 Jul 2026), so
the promotion timestamp precedes every TEST number. The COMBINED score
(unsupported + omitted) is THE endpoint; components and criticality are secondary
diagnostics, never substituted for it. Selection risk from developing on DEV is
acknowledged in Amendment 4 — the held-out TEST run is the confirmation.

WHY explore this: RQ1's three judge configs all failed to beat BERTScore, and
the diagnosis was granularity/validity of a single holistic judgment. This
metric decomposes the problem instead: break the note into ATOMIC CLAIMS, check
each claim against the transcript, and count the unsupported ones. The count is
fine-grained by construction (one unit per factual error, not one 1–5 score per
note), and every verdict is inspectable — you can SHOW the examiner which claim
failed and why.

Pipeline per note:
  1. DECOMPOSE  note -> atomic claims        (LLM, claim_decompose_v1, JSON)
  2. VERIFY     each claim vs transcript ->  supported | contradicted |
                not_mentioned                (LLM, claim_verify_v1, one batched
                call per note; per-claim fallback if the verdict list
                misaligns)
  3. AGGREGATE  n_unsupported = contradicted + not_mentioned (higher = worse)

Secondary verifier (near-free): the SummaC NLI backbone per claim — max
entailment over transcript sentences, unsupported if below threshold. RQ1
showed raw-shorthand NLI fails; atomic claims are exactly what MNLI models are
trained on, so this tests whether decomposition rescues NLI.

Rules kept: verifier LLM = llama3.1:8b (different family from the gemma
generator), temperature 0, reasoning off, constrained JSON. Omission counting
is deferred (needs transcript decomposition, ~10x tokens) — this metric covers
the INCORRECTNESS axis only; the DEV harness reports that cap honestly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from s2n.config import load_config
from s2n.generation.prompts import load_prompt
from s2n.llm.client import LLMClient

VERDICTS = ("supported", "contradicted", "not_mentioned")
OMISSION_VERDICTS = ("present", "absent")
CRITICALITY = ("critical", "minor")


def _list_schema(key: str, items: dict, n: int | None = None) -> dict:
    """JSON schema for {key: [...]}. With ``n``, the array length is FIXED
    (minItems == maxItems == n): Ollama compiles this into the decoding
    grammar, so a wrong-length verdict list cannot be generated — the ops fix
    for the batched-verify misalignment fallbacks."""
    arr: dict = {"type": "array", "items": items}
    if n is not None:
        arr["minItems"] = n
        arr["maxItems"] = n
    return {"type": "object", "properties": {key: arr}, "required": [key]}


DECOMPOSE_SCHEMA = _list_schema("claims", {"type": "string"})
FACTS_SCHEMA = _list_schema("facts", {"type": "string"})


def verify_schema(n: int) -> dict:
    return _list_schema("verdicts", {"enum": list(VERDICTS)}, n)


def omission_schema(n: int) -> dict:
    return _list_schema("verdicts", {"enum": list(OMISSION_VERDICTS)}, n)


def criticality_schema(n: int) -> dict:
    return _list_schema("labels", {"enum": list(CRITICALITY)}, n)


VERIFY_ONE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"enum": list(VERDICTS)}},
    "required": ["verdict"],
}
OMISSION_ONE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"enum": list(OMISSION_VERDICTS)}},
    "required": ["verdict"],
}


@dataclass
class ClaimReport:
    """Per-note result: every claim, every verdict, and the aggregate counts."""

    claims: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    used_fallback: bool = False  # batched verify misaligned -> per-claim calls
    parse_ok: bool = True

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    def count(self, verdict: str) -> int:
        return sum(v == verdict for v in self.verdicts)

    @property
    def n_unsupported(self) -> int:
        """The note-level score: contradicted + not_mentioned (higher = worse)."""
        return self.count("contradicted") + self.count("not_mentioned")


@dataclass
class OmissionReport:
    """Per-note omission result: transcript facts + present/absent verdicts."""

    facts: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    used_fallback: bool = False
    parse_ok: bool = True

    @property
    def n_facts(self) -> int:
        return len(self.facts)

    @property
    def n_omitted(self) -> int:
        return sum(v == "absent" for v in self.verdicts)


class ClaimVerifier:
    """Decompose -> verify -> aggregate, config-driven. Two axes:
    incorrectness (note claims vs transcript) and omission (transcript facts
    vs note)."""

    def __init__(self, cfg: dict | None = None, client: LLMClient | None = None):
        cfg = cfg or load_config()
        self.client = client or LLMClient(cfg)
        vcfg = cfg["claim_verifier"]
        self.model = vcfg["model"]
        self.decompose_prompt = load_prompt("judge", vcfg["decompose_prompt"])
        self.verify_prompt = load_prompt("judge", vcfg["verify_prompt"])
        self.facts_prompt = load_prompt("judge", vcfg["transcript_facts_prompt"])
        self.omission_prompt = load_prompt("judge", vcfg["omission_verify_prompt"])
        self.criticality_prompt = load_prompt("judge", vcfg["criticality_prompt"])

    # ---- stage 1: decompose ----------------------------------------------
    def decompose(self, note: str) -> list[str]:
        rendered = self.decompose_prompt.render(note=note)
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=DECOMPOSE_SCHEMA,
        )
        try:
            claims = json.loads(resp.text).get("claims", [])
        except json.JSONDecodeError:
            return []
        return [c.strip() for c in claims if isinstance(c, str) and c.strip()]

    # ---- stage 2: verify ---------------------------------------------------
    def _verify_batched(self, transcript: str, claims: list[str]) -> list[str] | None:
        """One call for all claims; None if the verdict list misaligns."""
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        rendered = self.verify_prompt.render(transcript=transcript, claims=numbered)
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=verify_schema(len(claims)),  # length fixed in the grammar
        )
        try:
            verdicts = json.loads(resp.text).get("verdicts", [])
        except json.JSONDecodeError:
            return None
        if len(verdicts) != len(claims) or not all(v in VERDICTS for v in verdicts):
            return None
        return list(verdicts)

    def _verify_one(self, transcript: str, claim: str) -> str:
        rendered = self.verify_prompt.render(transcript=transcript, claims=f"1. {claim}")
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=VERIFY_ONE_SCHEMA,
        )
        try:
            verdict = json.loads(resp.text).get("verdict", "")
        except json.JSONDecodeError:
            verdict = ""
        return verdict if verdict in VERDICTS else "not_mentioned"  # conservative

    # ---- omission axis: transcript facts vs note ---------------------------
    def decompose_transcript(self, transcript: str) -> list[str]:
        """Clinically important atomic facts a good note should capture.
        Cache per consultation — facts are reusable across all its notes."""
        rendered = self.facts_prompt.render(transcript=transcript)
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=FACTS_SCHEMA,
        )
        try:
            facts = json.loads(resp.text).get("facts", [])
        except json.JSONDecodeError:
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def check_omissions(self, note: str, facts: list[str]) -> OmissionReport:
        """Which transcript facts are absent from the note (batched + fallback)."""
        if not facts:
            return OmissionReport(parse_ok=False)
        numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
        rendered = self.omission_prompt.render(note=note, facts=numbered)
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=omission_schema(len(facts)),  # length fixed in the grammar
        )
        try:
            verdicts = json.loads(resp.text).get("verdicts", [])
        except json.JSONDecodeError:
            verdicts = []
        if len(verdicts) == len(facts) and all(v in OMISSION_VERDICTS for v in verdicts):
            return OmissionReport(facts, list(verdicts))
        # misaligned batch -> per-fact fallback
        verdicts = [self._check_one_omission(note, f) for f in facts]
        return OmissionReport(facts, verdicts, used_fallback=True)

    def _check_one_omission(self, note: str, fact: str) -> str:
        rendered = self.omission_prompt.render(note=note, facts=f"1. {fact}")
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=OMISSION_ONE_SCHEMA,
        )
        try:
            verdict = json.loads(resp.text).get("verdict", "")
        except json.JSONDecodeError:
            verdict = ""
        return verdict if verdict in OMISSION_VERDICTS else "present"  # conservative

    # ---- criticality (SECONDARY column; separate pass so the locked verify
    # path is untouched) ------------------------------------------------------
    def rate_criticality(self, transcript: str, unsupported_claims: list[str]) -> int:
        """Count how many unsupported claims are clinically CRITICAL.

        Runs only on the (few) unsupported claims. Batched, exact-length
        schema; on any parse problem returns 0 rather than guessing (a missing
        secondary is better than an invented one).
        """
        if not unsupported_claims:
            return 0
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(unsupported_claims))
        rendered = self.criticality_prompt.render(transcript=transcript, claims=numbered)
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=False,
            fmt=criticality_schema(len(unsupported_claims)),
        )
        try:
            labels = json.loads(resp.text).get("labels", [])
        except json.JSONDecodeError:
            return 0
        if len(labels) != len(unsupported_claims):
            return 0
        return sum(label == "critical" for label in labels)

    # ---- full pipeline -----------------------------------------------------
    def score(self, transcript: str, note: str) -> ClaimReport:
        claims = self.decompose(note)
        if not claims:
            return ClaimReport(parse_ok=False)
        verdicts = self._verify_batched(transcript, claims)
        if verdicts is None:  # misaligned batch -> slower but robust path
            verdicts = [self._verify_one(transcript, c) for c in claims]
            return ClaimReport(claims, verdicts, used_fallback=True)
        return ClaimReport(claims, verdicts)


class ClaimNLI:
    """Secondary verifier: SummaC's NLI backbone applied per atomic claim.

    A claim counts as unsupported if its MAX entailment probability over all
    transcript sentences is below the threshold. Tests whether decomposition
    rescues NLI (which failed on raw clinical shorthand in RQ1).
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.threshold = cfg["claim_verifier"]["nli_unsupported_threshold"]
        from s2n.evaluation.summac_metric import SummaCZS, split_clinical_sentences

        self._split = split_clinical_sentences
        self._nli = SummaCZS(cfg)

    def n_unsupported(self, transcript: str, claims: list[str]) -> int:
        import numpy as np

        src = self._split(transcript)
        if not src or not claims:
            return 0
        pairs = [(s, c) for c in claims for s in src]
        probs = self._nli.model.predict(
            pairs, apply_softmax=True, batch_size=self._nli.batch_size, show_progress_bar=False
        )
        ent = np.asarray(probs)[:, 1].reshape(len(claims), len(src))
        return int((ent.max(axis=1) < self.threshold).sum())
