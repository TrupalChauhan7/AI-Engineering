"""results.csv flag parsing + aggregation (the RQ1 answer key)."""

from s2n.data.evaluation_data import EvaluationLoader, parse_flagged_items


def test_parse_flags_critical_vs_noncritical():
    cell = "! critical one\n- non critical\n ! another critical"
    out = parse_flagged_items(cell)
    assert out == {"critical": 2, "noncritical": 1, "unflagged": 0, "total": 3}


def test_parse_flags_handles_blank_and_nan():
    assert parse_flagged_items("")["total"] == 0
    assert parse_flagged_items(float("nan"))["total"] == 0


def test_aggregate_has_full_evaluator_overlap():
    agg = EvaluationLoader.from_config().aggregate_by_note()
    assert len(agg) == 285  # 57 consultations x 5 models
    # Every note rated by all 5 evaluators -> IAA / human ceiling is computable.
    assert set(agg["n_evaluators"].unique()) == {5}


def test_doctor_notes_are_flagged_not_dropped():
    el = EvaluationLoader.from_config()
    agg = el.aggregate_by_note()
    assert agg["is_doctor"].sum() == 57
    assert len(el.machine_notes_only(agg)) == 228
