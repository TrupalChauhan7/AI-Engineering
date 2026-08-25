"""Canonical consultation-ID normalisation (guards a silent 19% data-loss bug).

human_eval writes ``day1_consultation010`` while transcripts/notes write
``day1_consultation10``. If the join isn't normalised, the 11 consultations
numbered >=10 silently vanish. These tests keep that from regressing.
"""

from s2n.data.evaluation_data import EvaluationLoader
from s2n.data.generation_data import GenerationLoader
from s2n.data.ids import canonical_consultation_id


def test_padding_variants_normalise_equal():
    assert canonical_consultation_id("day1_consultation010") == "day1_consultation10"
    assert canonical_consultation_id("day1_consultation10") == "day1_consultation10"
    assert canonical_consultation_id("day1_consultation01") == "day1_consultation01"
    assert canonical_consultation_id("day5_consultation011") == "day5_consultation11"


def test_human_eval_ids_all_match_transcripts():
    tx = set(GenerationLoader.from_config().consultation_ids())
    he = set(EvaluationLoader.from_config().load_human_eval()["Consultation"].unique())
    assert he <= tx, f"human_eval ids not in transcripts: {sorted(he - tx)}"
    assert len(he) == 57
