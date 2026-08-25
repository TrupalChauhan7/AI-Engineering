"""Baseline metrics: orientation + clinical sentence segmentation.

Heavy models (BERTScore, the SummaC NLI backbone) are exercised in the
pipelines, not here — these tests stay fast and offline.
"""

from s2n.evaluation.metrics import levenshtein_similarity, rouge_l
from s2n.evaluation.summac_metric import split_clinical_sentences

REF = "3/7 hx of diarrhoea, mainly watery. LLQ pain. PMH: asthma."
FAITHFUL = "3/7 hx diarrhoea, watery. LLQ pain. PMH asthma."
NOISE = "Patient has a broken leg and chest pain radiating to the arm."


def test_reference_metrics_are_faithfulness_oriented():
    # Higher = more faithful: a matching note must beat unrelated noise.
    assert rouge_l(FAITHFUL, REF) > rouge_l(NOISE, REF)
    assert levenshtein_similarity(FAITHFUL, REF) > levenshtein_similarity(NOISE, REF)


def test_clinical_segmenter_splits_newlines_and_punctuation():
    sents = split_clinical_sentences("Doctor: How are you? I see.\nPMH: asthma\nDH: inhalers")
    assert "How are you?" in sents  # speaker label stripped, sentence split
    assert "I see." in sents
    assert "PMH: asthma" in sents  # newline-separated shorthand kept as its own unit
    assert "DH: inhalers" in sents


def test_clinical_segmenter_drops_empty_fragments():
    assert split_clinical_sentences("\n\n  \n") == []
