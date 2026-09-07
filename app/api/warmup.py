"""Keep the demo's models resident so the first upload is not the slow one.

WHY: every component here loads lazily on first use — Whisper reads weights from
disk, and Ollama evicts a model that has been idle. Cold, that is minutes of
apparent nothing. The app builds these once at startup and hands the SAME warm
instances to every request via ``run_pipeline_staged``'s injection arguments.

THE DEMO IS NOT THE RESEARCH. ``demo_config`` returns a COPY of the config with
only ``transcription.model`` replaced by ``demo.transcription_model``. The
research config is never mutated, and the cached samples that show real numbers
are still built with the sealed research model.
"""

from __future__ import annotations

import copy
import logging
import time

from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.generation.generator import NoteGenerator
from s2n.transcription.whisper_asr import WhisperTranscriber

log = logging.getLogger("clarion.warmup")


def demo_config(cfg: dict) -> dict:
    """A copy of ``cfg`` using the demo ASR model. Never mutates the original."""
    demo = copy.deepcopy(cfg)
    model = cfg.get("demo", {}).get("transcription_model")
    if model:
        demo["transcription"]["model"] = model
    return demo


class WarmPool:
    """Warm, reusable pipeline components for the live-upload path."""

    def __init__(self, cfg: dict):
        self.cfg = demo_config(cfg)
        self.domain = self.cfg.get("active_domain") or self.cfg.get("domain")
        self.transcriber: WhisperTranscriber | None = None
        self.generator: NoteGenerator | None = None
        self.verifier: ClaimVerifier | None = None
        self.ready = False
        self.asr_model = self.cfg["transcription"]["model"]
        # generator+verifier built for a non-default domain on first use, cached
        # by domain so a second request in that domain reuses them. The verifier
        # MODEL (llama3.1:8b) is shared across domains, so only the generator's
        # model actually differs — Ollama keeps whichever is resident.
        self._by_domain: dict[str, tuple[NoteGenerator, ClaimVerifier]] = {}

    def warm(self) -> None:
        """Load Whisper and make Ollama resident. Safe to fail — we degrade."""
        t0 = time.perf_counter()

        self.transcriber = WhisperTranscriber(self.cfg)
        try:
            _ = self.transcriber.model  # forces the weights off disk
            log.info("whisper '%s' loaded (%.1fs)", self.asr_model, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 — a cold model is slow, not fatal
            log.warning("whisper preload failed (%s); it will load on first use", exc)

        self.generator = NoteGenerator(self.cfg)
        self.verifier = ClaimVerifier(self.cfg)

        # A tiny real call per model so Ollama pulls each into memory. The
        # answers are discarded; only residency matters.
        for label, client, model in (
            ("generator", self.generator.client, self.cfg["llm"]["model"]),
            ("verifier", self.verifier.client, self.cfg["claim_verifier"]["model"]),
        ):
            try:
                t1 = time.perf_counter()
                client.generate(user="ok", model=model, reasoning=False)
                log.info("ollama '%s' warm (%.1fs)", model, time.perf_counter() - t1)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s warmup failed (%s); first request will be slower", label, exc)

        self.ready = True
        log.info("models warmed in %.1fs (asr=%s)", time.perf_counter() - t0, self.asr_model)

    def components(self) -> dict:
        """Injection kwargs for ``run_pipeline_staged`` (the warm default domain)."""
        return {
            "transcriber": self.transcriber,
            "generator": self.generator,
            "verifier": self.verifier,
        }

    def components_for(self, cfg: dict) -> dict:
        """Injection kwargs for ``run_pipeline_staged`` in the domain of ``cfg``.

        The transcriber is domain-independent (ASR does not vary by domain), so
        the warm instance is ALWAYS reused. The generator and verifier bind their
        model + prompts at construction, so a non-default domain needs its own
        pair — built once and cached. This keeps the memory footprint honest: one
        warm generator at startup, others created only when actually requested.
        """
        domain = cfg.get("active_domain") or cfg.get("domain")
        if domain == self.domain:
            return self.components()
        if domain not in self._by_domain:
            log.info("building components for domain '%s' (first request)", domain)
            self._by_domain[domain] = (NoteGenerator(cfg), ClaimVerifier(cfg))
        generator, verifier = self._by_domain[domain]
        return {
            "transcriber": self.transcriber,  # ASR is domain-independent
            "generator": generator,
            "verifier": verifier,
        }
