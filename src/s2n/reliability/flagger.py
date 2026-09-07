"""The reliability flag — decide whether a generated note is RELIABLE or FLAGGED.

WHY this is the star: the app is just the stage; this flag is the point. It
turns the best faithfulness signal from RQ1 into a single deployable decision.

Signal: the claim-verifier COMBINED error count = n_unsupported + n_omitted
(the pre-registered TEST endpoint, Amendment 4). Higher = worse. A note is
RELIABLE when its combined count is at or below a threshold, FLAGGED otherwise.

Bands: ``reliability.reliable_max`` / ``reliability.unreliable_min`` in config.yaml, CHOSEN
ON DEV ONLY (the DEV quartiles of the combined count — see the config comment).
These are COUNT thresholds, not a 0–1 fraction; the old skeleton default of 0.7
was a placeholder on the wrong scale.

``verdict`` gives the three-way traffic light the demo shows; ``flag`` keeps the
older two-way reliable/flagged decision for callers that want a single bool.

``flag`` takes the per-note claim-verifier output and returns a plain, auditable
decision — every flag can be explained by the exact counts behind it.
"""

from __future__ import annotations

from s2n.config import load_config


def _combined(note_eval: dict) -> int:
    """Combined error count from a claim-verifier note result."""
    if note_eval.get("claims_combined") is not None:
        return int(note_eval["claims_combined"])
    return int(note_eval.get("n_unsupported", 0)) + int(note_eval.get("n_omitted", 0))


def verdict(note_eval: dict, cfg: dict | None = None) -> dict:
    """Three-way reliability verdict: ``reliable`` | ``review`` | ``unreliable``.

    Bands come from config (DEV-calibrated quartiles of the combined count).
    Returns a JSON-serialisable dict — plain str/int only — so an API layer can
    hand it straight to a frontend.
    """
    bands = (cfg or load_config())["reliability"]
    lo, hi = int(bands["reliable_max"]), int(bands["unreliable_min"])
    combined = _combined(note_eval)
    if combined <= lo:
        name, why = "reliable", f"{combined} ≤ {lo}"
    elif combined >= hi:
        name, why = "unreliable", f"{combined} ≥ {hi}"
    else:
        name, why = "review", f"{lo} < {combined} < {hi}"

    n_u = note_eval.get("n_unsupported", "?")
    n_o = note_eval.get("n_omitted", "?")
    return {
        "verdict": name,
        "combined": combined,
        # Wording is deliberate: these are FLAGS the checker raised, not errors
        # established as fact. What the research validated is this combined
        # COUNT at note level — not any individual claim verdict.
        "score_note": (
            f"{n_u} unsupported claim(s) + {n_o} omitted fact(s) = {combined} "
            f"combined flags ({why}) → {name.upper()}"
        ),
    }


def flag(note_eval: dict, cfg: dict | None = None, threshold: float | None = None) -> dict:
    """Reliability decision for one note.

    ``note_eval`` carries the claim-verifier counts for the note, i.e. either a
    precomputed ``claims_combined`` or the parts ``n_unsupported`` + ``n_omitted``
    (which also populate the reason string).

    Returns ``{"reliable": bool, "reason": str, "score": float}`` where ``score``
    is the combined error count (higher = less reliable).
    """
    if threshold is None:
        threshold = (cfg or load_config())["reliability"]["reliable_max"]
    combined = _combined(note_eval)
    reliable = combined <= threshold

    n_u = note_eval.get("n_unsupported", "?")
    n_o = note_eval.get("n_omitted", "?")
    verdict = "RELIABLE" if reliable else "FLAGGED"
    reason = (
        f"{n_u} unsupported + {n_o} omitted = {combined} combined "
        f"flags ({'≤' if reliable else '>'} threshold {threshold}) → {verdict}"
    )
    return {"reliable": bool(reliable), "reason": reason, "score": float(combined)}
