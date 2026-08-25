"""Generate SOAP notes with an MLX model (base or base+LoRA adapter).

WHY separate from llm/client.py: the RQ1 client talks to Ollama; the Track-B
base-vs-fine-tuned comparison must run BOTH arms through the SAME inference
stack (MLX 4-bit MedGemma), differing ONLY in whether the LoRA adapter is
attached — otherwise a base(Ollama)-vs-tuned(MLX) gap would confound the
adapter's effect. Identical prompt (v1.0) and decoding (greedy, temp 0) for
both, per the eval design.
"""

from __future__ import annotations

from s2n.generation.prompts import load_prompt


class MLXNoteGenerator:
    """Loads an MLX model once (optionally with a LoRA adapter) and generates
    SOAP notes from transcripts using the v1.0 generation prompt."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        prompt_version: str = "v1.0",
        max_tokens: int = 900,
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.prompt = load_prompt("note_generation", prompt_version)
        self.max_tokens = max_tokens
        self.sampler = make_sampler(temp=0.0)  # greedy = deterministic
        self.adapter_path = adapter_path

    def generate(self, transcript: str) -> str:
        from mlx_lm import generate as mlx_generate

        rendered = self.prompt.render(transcript=transcript)
        messages = []
        if rendered.system:
            messages.append({"role": "system", "content": rendered.system})
        messages.append({"role": "user", "content": rendered.user})
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self.sampler,
            verbose=False,
        )
        return text.strip()
