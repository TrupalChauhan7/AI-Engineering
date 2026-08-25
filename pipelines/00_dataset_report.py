"""Dataset audit — transcript tag frequencies + split summary.

WHY: we KEEP the text inside <UNSURE> and STRIP inaudible markers
(<UNIN/>, <INAUDIBLE_SPEECH/>). This script quantifies how often each tag
occurs so the choice is defensible ("inaudible speech is negligible, N=X") and
gives a free RQ2/limitations data point. Reads transcripts ONLY (leak-safe).

Run:  python pipelines/00_dataset_report.py
"""

from __future__ import annotations

from s2n.data.generation_data import GenerationLoader
from s2n.data.splits import load_or_create_split
from s2n.data.textgrid import TagCounts


def main() -> None:
    loader = GenerationLoader.from_config()
    ids = loader.consultation_ids()

    total = TagCounts()
    n_words = 0
    for c in loader.iter_consultations(ids):
        total = total.merge(c.tag_counts)
        n_words += len(c.dialogue.split())

    print(f"Consultations: {len(ids)}   approx words in dialogue: {n_words:,}")
    print("\nTag frequencies across all transcripts:")
    print(f"  <UNSURE> spans (kept as best-guess text): {total.unsure}")
    print(f"  inaudible markers (dropped, no content):  {total.inaudible}")
    for tag, n in sorted(total.per_tag.items()):
        print(f"    {tag:<22} {n}")
    if n_words:
        pct = 100 * total.inaudible / n_words
        print(f"\n  inaudible markers as % of words: {pct:.3f}%  (defence: negligible if small)")

    split = load_or_create_split()
    print(f"\nFrozen split (seed {split.seed}): DEV={len(split.dev)}  TEST={len(split.test)}")
    print(f"  DEV: {', '.join(split.dev)}")


if __name__ == "__main__":
    main()
