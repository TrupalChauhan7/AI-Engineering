"""Turn a transcript into a SOAP note via the LLM client + a versioned prompt.

WHY this is the whole "generation path": it takes a transcript string and
returns a note string. It imports the transcript-side loader and the LLM
client, but NEVER the evaluation loader — so the generator has no route to the
gold answers (project rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from s2n.config import load_config
from s2n.generation.prompts import load_prompt
from s2n.llm.client import LLMClient


@dataclass
class GeneratedNote:
    consultation_id: str | None
    note: str
    model: str
    prompt_version: str
    tokens_per_sec: float


class NoteGenerator:
    """Generates SOAP notes from transcripts using a fixed prompt version."""

    def __init__(self, cfg: dict | None = None, client: LLMClient | None = None):
        self.cfg = cfg or load_config()
        self.client = client or LLMClient(self.cfg)
        self.prompt_version = self.cfg["generation"]["prompt_version"]
        self.reasoning = self.cfg["generation"].get("reasoning", False)
        self.prompt = load_prompt("note_generation", self.prompt_version)

    def generate(self, transcript: str, consultation_id: str | None = None) -> GeneratedNote:
        rendered = self.prompt.render(transcript=transcript)
        resp = self.client.generate(
            user=rendered.user,
            system=rendered.system or None,
            reasoning=self.reasoning,  # OFF for the generator
        )
        return GeneratedNote(
            consultation_id=consultation_id,
            note=resp.text.strip(),
            model=self.cfg["llm"]["model"],
            prompt_version=self.prompt_version,
            tokens_per_sec=resp.tokens_per_sec,
        )


def generate_note(transcript: str, prompt_version: str | None = None) -> str:
    """Convenience: transcript -> note text (uses config's prompt version)."""
    cfg = load_config()
    if prompt_version:
        cfg["generation"]["prompt_version"] = prompt_version
    return NoteGenerator(cfg).generate(transcript).note
