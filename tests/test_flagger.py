"""The reliability alarm: threshold logic on the combined error count."""

from s2n.alarm.flagger import flag


def test_reliable_at_or_below_threshold():
    r = flag({"n_unsupported": 2, "n_omitted": 3}, threshold=6)  # combined 5 <= 6
    assert r["reliable"] is True and r["score"] == 5.0
    assert "RELIABLE" in r["reason"]


def test_flagged_above_threshold():
    r = flag({"n_unsupported": 5, "n_omitted": 4}, threshold=6)  # combined 9 > 6
    assert r["reliable"] is False and r["score"] == 9.0
    assert "FLAGGED" in r["reason"]


def test_boundary_is_inclusive():
    assert flag({"n_unsupported": 3, "n_omitted": 3}, threshold=6)["reliable"] is True


def test_accepts_precomputed_combined():
    r = flag({"claims_combined": 12}, threshold=8)
    assert r["reliable"] is False and r["score"] == 12.0


def test_threshold_defaults_to_config():
    # No explicit threshold -> reads alarm.reliable_max from config.
    r = flag({"n_unsupported": 0, "n_omitted": 0})
    assert r["score"] == 0.0 and r["reliable"] is True  # zero errors always reliable
