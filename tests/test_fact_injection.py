"""Fact-controlled injection: the planted errors must be exactly what we claim.

These tests guard the DEFENSIBILITY of the exploratory reliability probe. The whole
study rests on the labels being exact, so what is asserted here is not "the code
runs" but "the label is true": every planted hallucination is genuinely absent
from its transcript, the omission genuinely removes the target, and the same
seed genuinely reproduces the same notes.
"""

from __future__ import annotations

import pytest

from s2n.config import load_config
from s2n.evaluation.fact_injection import (
    consultation_rng,
    inject_hallucinations,
    inject_omission,
    load_catalogue,
    usable_entries,
)

TRANSCRIPT = "\n".join(
    [
        "Good morning, what brings you in today?",
        "I have had a sore throat for three days and it hurts to swallow.",
        "Any fever?",
        "I felt hot last night but did not take my temperature.",
        "Do you smoke?",
        "No, I have never smoked.",
        "Any medical conditions?",
        "I have high blood pressure and I take ramipril for it.",
    ]
)

NOTE = "\n".join(
    [
        "Subjective:",
        "Sore throat for three days with pain on swallowing.",
        "Subjective feeling of fever last night.",
        "History of high blood pressure, taking ramipril.",
        "Objective:",
        "Not examined.",
        "Assessment:",
        "Likely viral pharyngitis.",
        "Plan:",
        "Simple analgesia and safety netting.",
    ]
)

CFG = load_config()
CATALOGUE = load_catalogue(CFG)
THRESHOLD = CFG["synthetic_probe"]["omission_match_threshold"]


def _rng():
    return consultation_rng(42, "Data_1")


# ---- the absence guarantee ------------------------------------------------
def test_every_injected_keyword_is_absent_from_the_transcript():
    note, log = inject_hallucinations(NOTE, TRANSCRIPT, 3, _rng(), CATALOGUE)
    low = TRANSCRIPT.lower()
    by_sentence = {e["sentence"]: e["keyword"] for e in CATALOGUE}
    assert len(log.inserted) == 3
    for s in log.inserted:
        assert by_sentence[s].lower() not in low  # a true hallucination, as stated
        assert s in note  # and it actually landed in the note


def test_entries_whose_keyword_is_present_are_skipped_and_logged():
    usable, skipped = usable_entries(CATALOGUE, TRANSCRIPT)
    kws = {e["keyword"] for e in skipped}
    # the fixture transcript talks about temperature, smoking and blood pressure
    assert {"temperature", "smok", "blood pressure"} <= kws
    assert all(e["keyword"].lower() not in TRANSCRIPT.lower() for e in usable)


def test_doses_nest_so_the_dose_axis_is_paired():
    logs = [inject_hallucinations(NOTE, TRANSCRIPT, k, _rng(), CATALOGUE)[1] for k in (1, 2, 3)]
    assert set(logs[0].inserted) < set(logs[1].inserted) < set(logs[2].inserted)


def test_dose_beyond_the_usable_catalogue_raises():
    usable, _ = usable_entries(CATALOGUE, TRANSCRIPT)
    with pytest.raises(ValueError):
        inject_hallucinations(NOTE, TRANSCRIPT, len(usable) + 1, _rng(), CATALOGUE)


def test_hallucinations_go_into_the_subjective_section():
    note, log = inject_hallucinations(NOTE, TRANSCRIPT, 1, _rng(), CATALOGUE)
    lines = note.split("\n")
    i = lines.index(log.inserted[0])
    assert lines.index("Subjective:") < i < lines.index("Objective:")


# ---- the omission guarantee -----------------------------------------------
def test_omission_removes_the_target_fact():
    note, log = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    assert len(log.removed) == 1
    target = log.removed[0]
    assert target in NOTE and target not in note
    assert len(note.split("\n")) < len(NOTE.split("\n"))


def test_omission_target_is_grounded_in_the_transcript():
    _, log = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    assert log.match_score >= THRESHOLD  # it really was said in the consultation


def test_omission_removes_only_the_target():
    note, log = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    survivors = [ln for ln in NOTE.split("\n") if ln.strip() and ln != log.removed[0]]
    for ln in survivors:
        assert ln in note  # nothing else was collaterally deleted


def test_verifier_fact_annotation_never_changes_the_choice():
    """The fact list annotates the target; it must not select it (circularity)."""
    plain, log_a = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    facts = ["The patient has high blood pressure and takes ramipril."]
    annotated, log_b = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD, facts=facts)
    assert plain == annotated and log_a.removed == log_b.removed
    assert log_a.in_verifier_facts is None and log_b.in_verifier_facts in (True, False)


def test_no_grounded_sentence_leaves_the_note_untouched():
    note, log = inject_omission(NOTE, "Hello.\nGoodbye.", _rng(), THRESHOLD)
    assert note == NOTE and log.removed == []


# ---- determinism ----------------------------------------------------------
def test_injection_is_deterministic_under_a_fixed_seed():
    for k in (1, 2, 3):
        a = inject_hallucinations(NOTE, TRANSCRIPT, k, _rng(), CATALOGUE)
        b = inject_hallucinations(NOTE, TRANSCRIPT, k, _rng(), CATALOGUE)
        assert a[0] == b[0] and a[1].inserted == b[1].inserted
    a = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    b = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    assert a[0] == b[0] and a[1].removed == b[1].removed


def test_streams_differ_by_consultation_but_not_by_run_order():
    """Per-consultation streams: --limit 3 must not change what consultation 3 gets."""
    one = consultation_rng(42, "Data_7").permutation(10)
    two = consultation_rng(42, "Data_7").permutation(10)
    other = consultation_rng(42, "Data_8").permutation(10)
    assert list(one) == list(two) and list(one) != list(other)


# ---- the shape the generator actually emits -------------------------------
# MedGemma-4B writes one paragraph per section with the label INLINE
# ("**Subjective:** Patient reports ..."), not as standalone header lines. The
# fixtures above use the tidy form, so this guards the real one.
INLINE_NOTE = "\n".join(
    [
        "**SOAP Note**",
        "",
        "**Subjective:** Sore throat for three days with pain on swallowing. "
        "History of high blood pressure, taking ramipril.",
        "",
        "**Objective:** (Not provided in transcript)",
        "",
        "**Assessment:** Likely viral pharyngitis.",
        "",
        "**Plan:** Simple analgesia and safety netting.",
    ]
)


def test_inline_header_note_gets_hallucinations_in_the_subjective_block():
    note, log = inject_hallucinations(INLINE_NOTE, TRANSCRIPT, 1, _rng(), CATALOGUE)
    lines = note.split("\n")
    i = lines.index(log.inserted[0])
    subj = next(j for j, ln in enumerate(lines) if ln.startswith("**Subjective:"))
    obj = next(j for j, ln in enumerate(lines) if ln.startswith("**Objective:"))
    assert subj < i < obj


def test_omission_on_an_inline_header_note_keeps_the_section_label():
    note, log = inject_omission(INLINE_NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    assert len(log.removed) == 1
    assert log.removed[0] not in note
    assert not log.removed[0].startswith("**")  # label never inside the deletable unit
    assert "**Subjective:**" in note  # section survives its content being cut
    for section in ("**Objective:**", "**Assessment:**", "**Plan:**"):
        assert section in note


# ---- the pre-declared S/O restriction -------------------------------------
# Pre-declared before the full run: omission targets come from Subjective and
# Objective only. The omission axis measures coverage of patient-stated clinical
# facts, so deleting management advice would test it out of scope. The
# restriction is by note SECTION, which the verifier never sees, so selection
# stays independent of its fact list.
def test_omission_never_targets_plan_or_assessment():
    plan_sentences = {"Simple analgesia and safety netting.", "Likely viral pharyngitis."}
    for note in (NOTE, INLINE_NOTE):
        _, log = inject_omission(note, TRANSCRIPT, _rng(), THRESHOLD)
        assert log.removed and log.removed[0] not in plan_sentences


def test_note_units_are_restricted_to_subjective_and_objective():
    from s2n.evaluation.fact_injection import _note_units

    for note in (NOTE, INLINE_NOTE):
        restricted = {u for _, u in _note_units(note)}
        everything = {u for _, u in _note_units(note, sections=None)}
        assert restricted < everything
        assert not any("analgesia" in u or "pharyngitis" in u for u in restricted)
        assert any("Sore throat" in u for u in restricted)


def test_candidate_count_is_logged():
    _, log = inject_omission(NOTE, TRANSCRIPT, _rng(), THRESHOLD)
    assert log.n_candidates >= 1
    _, empty = inject_omission(NOTE, "Hello.\nGoodbye.", _rng(), THRESHOLD)
    assert empty.n_candidates == 0
