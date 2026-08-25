"""Judge parsing — JSON modes (v1.0/v1.1) and Selene absolute rubric (v1.2).

Mocked client — no live LLM in the test suite. Each judge pins its mode
explicitly rather than inheriting the global config default.
"""

from s2n.evaluation.judge import FaithfulnessJudge, _parse_absolute, _parse_json
from s2n.llm.client import LLMResponse


class _FakeClient:
    def __init__(self, text):
        self._text = text

    def generate(self, *a, **k):
        return LLMResponse(text=self._text, raw={})


def _json_cfg():
    return {"judge": {"model": "m", "prompt_version": "v1.0", "mode": "json"}}


def _selene_cfg():
    return {
        "judge": {
            "model": "m",
            "prompt_version": "v1.2",
            "mode": "rubric_absolute",
            "rubric_scale": 5,
        }
    }


# ---- JSON mode (v1.0 / v1.1) --------------------------------------------
def test_parse_tolerates_surrounding_prose():
    parsed = _parse_json('Sure! {"score": 0.5, "hallucinations": [], "omissions": []} ok')
    assert parsed["score"] == 0.5


def test_judge_parses_and_counts():
    text = '{"score": 0.7, "hallucinations": ["invented BP"], "omissions": ["a", "b"]}'
    r = FaithfulnessJudge(cfg=_json_cfg(), client=_FakeClient(text)).judge("t", "n")
    assert r["score"] == 0.7 and r["n_hallucinations"] == 1 and r["n_omissions"] == 2
    assert r["parse_ok"] is True


def test_judge_survives_bad_json():
    r = FaithfulnessJudge(cfg=_json_cfg(), client=_FakeClient("not json")).judge("t", "n")
    assert r["parse_ok"] is False and r["score"] != r["score"]  # NaN


def test_judge_clips_out_of_range_score():
    text = '{"score": 1.5, "hallucinations": [], "omissions": []}'
    r = FaithfulnessJudge(cfg=_json_cfg(), client=_FakeClient(text)).judge("t", "n")
    assert r["score"] == 1.0


# ---- Selene absolute-rubric mode (v1.2) ---------------------------------
def test_parse_absolute_extracts_result():
    assert _parse_absolute("**Reasoning:** blah\n\n**Result:** 4", 5) == 4
    assert _parse_absolute("Result: 2", 5) == 2
    assert _parse_absolute("nonsense 5 here", 5) == 5  # fallback: last in-range int


def test_selene_maps_rubric_to_faithfulness():
    # 5/5 -> 1.0, 1/5 -> 0.0, 3/5 -> 0.5
    for text, expect in [("**Result:** 5", 1.0), ("**Result:** 1", 0.0), ("**Result:** 3", 0.5)]:
        r = FaithfulnessJudge(cfg=_selene_cfg(), client=_FakeClient(text)).judge("t", "n")
        assert r["score"] == expect and r["parse_ok"] is True and r["rubric_result"] is not None


def test_selene_survives_no_score():
    j = FaithfulnessJudge(cfg=_selene_cfg(), client=_FakeClient("I cannot evaluate"))
    r = j.judge("t", "n")
    assert r["parse_ok"] is False and r["score"] != r["score"]  # NaN
