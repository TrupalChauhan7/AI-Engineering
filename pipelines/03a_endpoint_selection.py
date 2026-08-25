"""Endpoint selection on DEV (pre-registration §B4) — DEV ONLY, no TEST peek.

Inspects the marginal distribution of the candidate human targets on the DEV
split and reports which is usable as the primary correlation target. Reports
both views because the doctor-note include/exclude choice is still open:
  * ALL dev notes (machine + doctor)
  * MACHINE-only dev notes (doctor excluded)

Run:  python pipelines/03a_endpoint_selection.py
"""

from __future__ import annotations

from s2n.evaluation.targets import choose_primary_target, describe_targets
from s2n.evaluation.track_a import build_dev_table


def _report(name: str, table) -> None:
    print(f"\n=== {name}  (n={len(table)} notes, {table.consultation.nunique()} consults) ===")
    summary = describe_targets(table)
    print(summary.to_string(index=False))
    try:
        print("-> primary target (first usable, preference order):", choose_primary_target(summary))
    except ValueError as e:
        print("->", e)


def main() -> None:
    dev = build_dev_table()
    print("Endpoint selection — DEV split only (pre-registration B4).")
    _report("ALL dev notes (machine + doctor)", dev)
    _report("MACHINE-only dev notes (doctor excluded)", dev[~dev.is_doctor])
    print(
        "\nNote: choice is on DISTRIBUTION only (no metric correlations inspected)."
        "\nDoctor include/exclude for the final correlation is a separate open decision."
    )


if __name__ == "__main__":
    main()
