"""Reproduce the Phase-15b LoRA adapter (Track B) — the TRAINING step.

WHY THIS EXISTS: data prep (pipeline 09), adapter inference
(``s2n.finetune.mlx_generate``) and the base-vs-fine-tuned eval (pipeline 10)
were all in the repo, but the training invocation that actually produced the
adapter was not — so the one artefact the Track-B result depends on could not be
regenerated from this repository. This script closes that gap.

PROVENANCE — these flags are NOT reconstructed from memory. MLX-LM writes its
full training configuration next to the weights, and every value below is read
straight off the run that produced the adapter:

    results/finetune/best_adapter/adapter_config.json

Only the arguments that DIFFER from mlx_lm's defaults strictly need passing;
they are all passed explicitly anyway, because a reproducibility record that
silently depends on a library default is not one. Verified against
``mlx_lm.lora.CONFIG_DEFAULTS`` (mlx_lm 0.31.3):

    non-default : batch_size 2 (4), iters 1500 (1000), max_seq_length 1024
                  (2048), steps_per_eval 150 (200), steps_per_report 50 (10),
                  save_every 150 (100)
    at default  : num_layers 16, learning_rate 1e-5, val_batches 25, seed 0,
                  fine_tune_type lora, optimizer adam
    not settable on the CLI: lora_parameters {rank 8, scale 20.0, dropout 0.0}
                  — these are the mlx_lm defaults the run used.

RESULT THIS REPRODUCES (PROJECT_LOG Phase 15b): train loss 3.42 -> 0.85, best
validation loss 1.226 at iter 1500, peak memory 14.4 GB. The adapter was then
selected BY BEST VALIDATION LOSS — which at iter 1500 coincided with the final
checkpoint — and staged to ``results/finetune/best_adapter/``.

The honest headline of that experiment was NEGATIVE: the adapter did not improve
PriMock faithfulness and measurably hurt surface fidelity (BERTScore
-0.039 [-0.054, -0.024]). It is reproducible here so that negative result is
checkable, not so it can be improved.

Usage:
    python pipelines/12_finetune_train.py --dry-run   # print the command, run nothing
    python pipelines/12_finetune_train.py             # ~hours on an M4 Pro; asks first
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from s2n.config import ROOT

ADAPTER_DIR = ROOT / "results" / "finetune" / "full_adapter"
BEST_DIR = ROOT / "results" / "finetune" / "best_adapter"
DATA_DIR = ROOT / "data" / "mts_dialog" / "mlx"
MODEL_DIR = ROOT / "models" / "medgemma-4b-it-4bit"
RECORDED_CONFIG = BEST_DIR / "adapter_config.json"

# Exactly the Phase-15b invocation. Keep in step with RECORDED_CONFIG.
# Flag/value pairs are kept on one line: this list exists to be READ against
# results/finetune/best_adapter/adapter_config.json.
# fmt: off
COMMAND = [
    sys.executable, "-m", "mlx_lm", "lora",
    "--model", str(MODEL_DIR),
    "--data", str(DATA_DIR),
    "--train",
    "--fine-tune-type", "lora",
    "--num-layers", "16",
    "--batch-size", "2",
    "--iters", "1500",
    "--learning-rate", "1e-5",
    "--max-seq-length", "1024",
    "--steps-per-report", "50",
    "--steps-per-eval", "150",
    "--val-batches", "25",
    "--save-every", "150",
    "--seed", "0",
    "--adapter-path", str(ADAPTER_DIR),
]
# fmt: on

# CLI flag -> key in the recorded adapter_config.json, for the drift check.
_CHECK = {
    "--num-layers": ("num_layers", int),
    "--batch-size": ("batch_size", int),
    "--iters": ("iters", int),
    "--max-seq-length": ("max_seq_length", int),
    "--steps-per-report": ("steps_per_report", int),
    "--steps-per-eval": ("steps_per_eval", int),
    "--val-batches": ("val_batches", int),
    "--save-every": ("save_every", int),
    "--seed": ("seed", int),
    "--learning-rate": ("learning_rate", float),
    "--fine-tune-type": ("fine_tune_type", str),
}


def _pretty(cmd: list[str]) -> str:
    """One flag (with its value) per line, paths shown repo-relative."""
    parts, i = [], 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            val = cmd[i + 1]
            try:
                val = str(Path(val).relative_to(ROOT))
            except ValueError:
                pass
            parts.append(f"{tok} {val}")
            i += 2
        else:
            parts.append(tok)
            i += 1
    head = " ".join(parts[:4])  # python -m mlx_lm lora
    return "    " + " \\\n      ".join([head, *parts[4:]])


def verify_against_recorded_run() -> list[str]:
    """Check COMMAND still matches the config MLX wrote beside the adapter."""
    if not RECORDED_CONFIG.is_file():
        return [f"no recorded config at {RECORDED_CONFIG.relative_to(ROOT)} — cannot verify"]
    recorded = json.loads(RECORDED_CONFIG.read_text())
    problems = []
    for flag, (key, cast) in _CHECK.items():
        ours = cast(COMMAND[COMMAND.index(flag) + 1])
        theirs = cast(recorded[key])
        if ours != theirs:
            problems.append(f"{flag}: script says {ours!r}, recorded run used {theirs!r}")
    if recorded.get("data") and not str(DATA_DIR).endswith(recorded["data"]):
        problems.append(f"--data: script {DATA_DIR}, recorded {recorded['data']}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    problems = verify_against_recorded_run()
    print("Phase-15b LoRA training — reproduction of results/finetune/best_adapter\n")
    print(_pretty(COMMAND) + "\n")
    if problems:
        print("  ⚠ DRIFT from the recorded run:")
        for p in problems:
            print(f"      {p}")
    else:
        print("  ✔ matches results/finetune/best_adapter/adapter_config.json")

    print(
        "\n  After training, the adapter is selected BY BEST VALIDATION LOSS from the\n"
        f"  checkpoints in {ADAPTER_DIR.relative_to(ROOT)}/ and copied to\n"
        f"  {BEST_DIR.relative_to(ROOT)}/. In the Phase-15b run the best val loss\n"
        "  (1.226) fell at iter 1500, so that was the final checkpoint."
    )

    if args.dry_run:
        print("\n  --dry-run: nothing was run.")
        return

    for path, what in ((MODEL_DIR, "4-bit base model"), (DATA_DIR, "MTS-Dialog JSONL")):
        if not path.exists():
            raise SystemExit(
                f"\nMissing {what}: {path.relative_to(ROOT)}\n"
                "  model: download the MLX 4-bit MedGemma into models/\n"
                "  data : python pipelines/09_format_mts.py"
            )

    if not args.yes:
        print("\n  This retrains the adapter (hours, ~14.4 GB peak) and OVERWRITES")
        print(f"  {ADAPTER_DIR.relative_to(ROOT)}/.")
        if input("  Type 'train' to continue: ").strip() != "train":
            raise SystemExit("  aborted.")

    raise SystemExit(subprocess.call(COMMAND))


if __name__ == "__main__":
    main()
