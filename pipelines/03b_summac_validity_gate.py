"""SummaC DEV validity gate — pre-registration Amendment A1.1.

WHY: SummaC's NLI backbone is trained on general-domain text; our notes are
dense CLINICAL SHORTHAND. If SummaC can't tell an obviously-faithful shorthand
note from an obviously-unfaithful one, then "the judge beats SummaC" is a hollow
win. So BEFORE SummaC counts as a baseline, it must separate hand-picked
obvious cases on DEV. If it fails: swap the NLI backbone (A1.1) or fall back to
Levenshtein as the primary comparator (A1.3).

Each case: a real DEV transcript + a faithful shorthand note (true to it) + an
unfaithful shorthand note (clinically plausible but absent from that transcript).

Run:  python pipelines/03b_summac_validity_gate.py
"""

from __future__ import annotations

import pandas as pd

from s2n.data.generation_data import GenerationLoader
from s2n.evaluation.summac_metric import SummaCZS

# (consultation_id, faithful shorthand note, unfaithful shorthand note)
CASES = [
    (
        "day1_consultation01",  # diarrhoea / gastroenteritis
        "3/7 hx watery diarrhoea, 6-7x/day.\nLLQ crampy pain, no radiation.\n"
        "PMH: asthma.\nImp: gastroenteritis.",
        "Productive cough w/ green sputum 2/52.\nPleuritic R chest pain.\n"
        "PMH: COPD.\nImp: community-acquired pneumonia.",
    ),
    (
        "day1_consultation02",  # itchy skin / eczema
        "4/7 hx itchy sore skin, chest and arms.\n"
        "PMH: eczema, tx steroid cream + emollient.\nItch disturbing sleep.",
        "Central crushing chest pain radiating to L arm.\n"
        "PMH: T2DM on insulin.\nImp: ACS, for angiography.",
    ),
    (
        "day1_consultation09",  # dysuria / UTI
        "Few days hx dysuria, stinging.\nAssoc lower abdo cramping.\n19F.\nImp: ?UTI.",
        "3/12 progressive R knee swelling and locking.\n"
        "H/o sports injury.\nImp: meniscal tear, for MRI.",
    ),
    (
        "day2_consultation03",  # hearing loss + facial numbness
        "3/52 hx reduced hearing + facial numbness.\n40F.\nImp: ?neurological cause.",
        "Acute severe epigastric pain radiating to back.\n"
        "Vomiting.\nEtOH excess.\nImp: acute pancreatitis.",
    ),
]

MARGIN = 0.05  # minimum required mean(faithful) - mean(unfaithful)


def main() -> None:
    gen = GenerationLoader.from_config()
    summac = SummaCZS()

    rows = []
    for cid, faithful, unfaithful in CASES:
        src = gen.load(cid).dialogue
        sf = summac.score_one(src, faithful)
        su = summac.score_one(src, unfaithful)
        rows.append(
            {
                "consultation": cid,
                "faithful": round(sf, 3),
                "unfaithful": round(su, 3),
                "separates": sf > su,
            }
        )
    df = pd.DataFrame(rows)
    print("SummaC-ZS validity gate (DEV) — higher = more faithful\n")
    print(df.to_string(index=False))

    margin = df["faithful"].mean() - df["unfaithful"].mean()
    all_sep = bool(df["separates"].all())
    passed = all_sep and margin >= MARGIN
    mf, mu = df["faithful"].mean(), df["unfaithful"].mean()
    print(f"\nmean(faithful)={mf:.3f}  mean(unfaithful)={mu:.3f}  margin={margin:.3f}")
    print(f"All cases separate: {all_sep}   Margin >= {MARGIN}: {margin >= MARGIN}")
    print(
        "GATE:",
        "PASS -> SummaC stays the primary comparator (pre-reg A1.3)"
        if passed
        else "FAIL -> swap NLI backbone (A1.1) or fall back to Levenshtein (A1.3)",
    )


if __name__ == "__main__":
    main()
