"""Tier-3: LLM-as-a-judge faithfulness score (the project's main method).

WHY: the judge reads the TRANSCRIPT + the generated NOTE and rates how faithful
the note is. This is the "reliability" whose value we test against the cheaper
baselines (pre-registration §B1 tier 3). It must be a DIFFERENT model family
from the generator (gemma) to avoid self-enhancement bias (project rule 6).

Two output modes, selected by ``judge.mode`` in config:
  * ``json`` (v1.0 holistic score, v1.1 enumerate-and-count): constrained-JSON
    decoding via a schema, so the output always parses.
  * ``rubric_absolute`` (v1.2, Selene-Mini): the purpose-built evaluator's
    trained Prometheus-style format — free text with ``**Result:** N`` (1–5),
    parsed and mapped to faithfulness (N−1)/(scale−1). NO JSON constraint, since
    forcing JSON would fight the format Selene was trained on.

Orientation: ``score`` is always a faithfulness score (higher = more faithful),
so the correlation harness treats every metric uniformly.
"""

from __future__ import annotations

import json
import re

from s2n.config import load_config
from s2n.generation.prompts import load_prompt
from s2n.llm.client import LLMClient

# JSON schema for constrained decoding (v1.0/v1.1). Only the enumerated lists are
# REQUIRED — the canonical count is the list length, so a holistic score may be
# absent (v1.1).
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "hallucinations": {"type": "array", "items": {"type": "string"}},
        "omissions": {"type": "array", "items": {"type": "string"}},
        "n_hallucinations": {"type": "integer"},
        "n_omissions": {"type": "integer"},
    },
    "required": ["hallucinations", "omissions"],
}

# Selene emits "**Result:** N"; be tolerant of markdown/spacing variants.
_RESULT_RE = re.compile(r"result\s*\**\s*:?\s*\**\s*([1-5])", re.IGNORECASE)


def _parse_json(text: str) -> dict:
    """Parse the judge's JSON; tolerate stray prose around the object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _parse_absolute(text: str, scale: int) -> int:
    """Extract the integer rubric score from Selene's '**Result:** N' output."""
    matches = _RESULT_RE.findall(text)
    if not matches:
        # Fallback: the last standalone 1..scale integer in the text.
        matches = re.findall(rf"\b([1-{scale}])\b", text)
    if not matches:
        raise ValueError(f"No rubric score found in: {text[:120]!r}")
    return min(max(int(matches[-1]), 1), scale)


def _blank(score, parse_ok, raw, **extra) -> dict:
    out = {
        "score": score,
        "hallucinations": [],
        "omissions": [],
        "n_hallucinations": 0,
        "n_omissions": 0,
        "parse_ok": parse_ok,
        "raw_text": raw,
    }
    out.update(extra)
    return out


class FaithfulnessJudge:
    """Scores note faithfulness against the transcript via a versioned prompt."""

    def __init__(self, cfg: dict | None = None, client: LLMClient | None = None):
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)
        jcfg = self.cfg["judge"]
        self.model = jcfg["model"]
        self.reasoning = jcfg.get("reasoning", False)
        self.mode = jcfg.get("mode", "json")
        self.rubric_scale = int(jcfg.get("rubric_scale", 5))
        self.prompt_version = jcfg["prompt_version"]
        self.prompt = load_prompt("judge", self.prompt_version)

    def judge(self, transcript: str, note: str) -> dict:
        rendered = self.prompt.render(transcript=transcript, note=note)
        if self.mode == "rubric_absolute":
            return self._judge_absolute(rendered)
        return self._judge_json(rendered)

    def _judge_json(self, rendered) -> dict:
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=self.reasoning,
            fmt=JUDGE_SCHEMA,
        )
        try:
            data = _parse_json(resp.text)
            score = float(data.get("score", float("nan")))
            hallucinations = list(data.get("hallucinations", []))
            omissions = list(data.get("omissions", []))
            parse_ok = True
        except (json.JSONDecodeError, ValueError, TypeError):
            score, hallucinations, omissions, parse_ok = float("nan"), [], [], False
        return {
            "score": min(max(score, 0.0), 1.0) if score == score else float("nan"),
            "hallucinations": hallucinations,
            "omissions": omissions,
            "n_hallucinations": len(hallucinations),
            "n_omissions": len(omissions),
            "parse_ok": parse_ok,
            "raw_text": resp.text,
        }

    def _judge_absolute(self, rendered) -> dict:
        # Plain text (no JSON constraint) — let Selene use its trained format.
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            model=self.model,
            reasoning=self.reasoning,
            fmt=None,
        )
        try:
            result = _parse_absolute(resp.text, self.rubric_scale)
            score = (result - 1) / (self.rubric_scale - 1)  # 1->0.0 .. scale->1.0
            return _blank(score, True, resp.text, rubric_result=result)
        except ValueError:
            return _blank(float("nan"), False, resp.text, rubric_result=None)

    def judge_many(self, transcripts: list[str], notes: list[str]) -> list[dict]:
        return [self.judge(t, n) for t, n in zip(transcripts, notes)]
