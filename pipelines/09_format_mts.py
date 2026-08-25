"""Format MTS-Dialog CSVs into MLX-LM chat-format JSONL (train + valid).

Writes data/mts_dialog/mlx/{train,valid}.jsonl — the layout MLX-LM's LoRA
trainer expects (--data <dir>). Train on MTS-Dialog; PriMock57 is the disjoint
evaluation set (never trained on).

Run:  python pipelines/09_format_mts.py
"""

from __future__ import annotations

from s2n.config import ROOT
from s2n.finetune.mts import format_csv

SRC = ROOT / "data" / "mts_dialog"
OUT = ROOT / "data" / "mts_dialog" / "mlx"


def main() -> None:
    n_train = format_csv(SRC / "MTS-Dialog-TrainingSet.csv", OUT / "train.jsonl")
    n_valid = format_csv(SRC / "MTS-Dialog-ValidationSet.csv", OUT / "valid.jsonl")
    print(f"train.jsonl: {n_train} examples")
    print(f"valid.jsonl: {n_valid} examples")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
