"""One tiny, backend-agnostic interface: ``generate(system, user)``.

WHY: generation/ and evaluation/ must NOT know which model is behind them. All
vendor knowledge lives HERE, so we can stay free (local Ollama) now and swap the
backend in one place later without touching any caller.

Current backend: Ollama's local HTTP API (no extra pip dependency — plain
``requests`` to ``/api/chat``). Decoding is deterministic (temperature 0) and
generator reasoning is OFF (project rule 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from s2n.config import load_config


@dataclass
class LLMResponse:
    """The generated text plus the raw backend payload (for timing/tokens)."""

    text: str
    raw: dict

    @property
    def eval_tokens(self) -> int:
        return int(self.raw.get("eval_count", 0) or 0)

    @property
    def tokens_per_sec(self) -> float:
        ns = self.raw.get("eval_duration") or 0
        return (self.eval_tokens / (ns / 1e9)) if ns else 0.0


class LLMClient:
    """Config-driven wrapper. Callers use ``generate``; they never see a vendor."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or load_config())["llm"]
        self.backend = self.cfg.get("backend", "ollama")

    def generate(
        self,
        user: str,
        system: str | None = None,
        *,
        temperature: float | None = None,
        reasoning: bool = False,
        model: str | None = None,
        fmt: dict | str | None = None,
    ) -> LLMResponse:
        """Return an ``LLMResponse`` for a (system, user) message pair.

        ``reasoning=False`` keeps the generator's chain-of-thought OFF; a judge
        can pass ``reasoning=True``. ``fmt`` is Ollama's structured-output format
        ("json" or a JSON schema dict) for constrained decoding — used by the
        judge so its output always parses.
        """
        if self.backend != "ollama":
            raise NotImplementedError(f"Backend '{self.backend}' not wired yet.")
        return self._ollama_chat(
            user=user,
            system=system,
            temperature=self.cfg["temperature"] if temperature is None else temperature,
            reasoning=reasoning,
            model=model or self.cfg["model"],
            fmt=fmt,
        )

    def _ollama_chat(
        self,
        user: str,
        system: str | None,
        temperature: float,
        reasoning: bool,
        model: str,
        fmt: dict | str | None = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        options = {"temperature": temperature}
        if self.cfg.get("num_ctx"):
            # Raise the context window: Ollama defaults to 4096, which truncates
            # (and 400s) long PriMock transcripts. Sized to fit the longest
            # consultation + the judge rubric + output.
            options["num_ctx"] = self.cfg["num_ctx"]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": reasoning,  # False -> reasoning OFF
            "options": options,
        }
        if fmt is not None:
            payload["format"] = fmt  # constrained decoding (JSON schema or "json")
        url = f"{self.cfg['host'].rstrip('/')}/api/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.cfg.get("timeout", 1200))
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request to {url} failed: {e}") from e
        data = resp.json()
        return LLMResponse(text=data.get("message", {}).get("content", ""), raw=data)


def generate(user: str, system: str | None = None, **kwargs) -> str:
    """Module-level convenience: return just the generated text."""
    return LLMClient().generate(user, system=system, **kwargs).text
