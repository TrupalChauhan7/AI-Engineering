"""B6 ceiling + B7 PR-AUC helpers (synthetic data; no models)."""

import numpy as np
import pandas as pd

from s2n.evaluation.ceiling_flagging import inter_evaluator_ceiling, pr_auc_bootstrap


def _human_eval(agree: bool):
    """3 evaluators over 6 notes; agree=True -> identical rankings (ceiling ~1)."""
    rows = []
    for c in range(6):
        base = c  # increasing difficulty
        for e in (1, 2, 3):
            val = base if agree else np.random.default_rng(c * 10 + e).integers(0, 6)
            rows.append(
                {
                    "Consultation": f"day1_consultation{c:02d}",
                    "Model": "model_1",
                    "Evaluator": f"evaluator_{e}",
                    "errors_total": val,
                    "is_doctor": False,
                }
            )
    return pd.DataFrame(rows)


def test_ceiling_high_when_evaluators_agree():
    ids = [f"day1_consultation{c:02d}" for c in range(6)]
    out = inter_evaluator_ceiling(_human_eval(agree=True), "errors_total", ids)
    assert out["n_pairs"] == 3 and out["ceiling"] > 0.95


def test_ceiling_returns_pairs():
    ids = [f"day1_consultation{c:02d}" for c in range(6)]
    out = inter_evaluator_ceiling(_human_eval(agree=False), "errors_total", ids)
    assert out["n_pairs"] == 3 and -1.0 <= out["ceiling"] <= 1.0


def test_pr_auc_perfect_separation():
    # score (unreliability) perfectly ranks the positives on top -> AP ~ 1.0
    df = pd.DataFrame(
        {
            "consultation": [f"c{i}" for i in range(10)],
            "unreliability": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "is_critical": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    out = pr_auc_bootstrap(df, "unreliability", "is_critical", n_boot=200)
    assert out["pr_auc"] == 1.0 and out["base_rate"] == 0.5


def test_pr_auc_reports_base_rate():
    df = pd.DataFrame(
        {
            "consultation": [f"c{i}" for i in range(8)],
            "unreliability": np.random.default_rng(0).random(8),
            "is_critical": [1, 1, 1, 1, 1, 1, 1, 0],  # 7/8 positive
        }
    )
    out = pr_auc_bootstrap(df, "unreliability", "is_critical", n_boot=100)
    assert out["base_rate"] == 0.875


# --- Amendment 5: EXPLORATORY worst-quartile flagging (post-hoc companion to §B7) ---


def _load_harness():
    """Import pipelines/11_test_oneshot.py (module name starts with a digit)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "pipelines" / "11_test_oneshot.py"
    spec = importlib.util.spec_from_file_location("oneshot_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scored_frame():
    """8 machine notes + 2 doctor notes over 5 consultations, errors_total 1..8."""
    rows = []
    for i in range(8):
        rows.append(
            {
                "consultation": f"day1_consultation{i % 5:02d}",
                "errors_total": float(i + 1),
                "is_doctor": False,
                "claims_neg_combined": -float(i + 1),
                "bertscore": 1.0 - i / 10,
                "summac": 0.5,
                "rougeL": 0.5,
                "levenshtein": 0.5,
                "judge_selene": 0.5,
            }
        )
    for j in range(2):
        r = dict(rows[j])
        r["is_doctor"] = True
        r["errors_total"] = 100.0  # doctor rows must not influence tau
        rows.append(r)
    return pd.DataFrame(rows)


def test_tau_is_75th_percentile_of_machine_only_errors_total():
    """The frozen tau recipe: 75th percentile, machine notes only (Amendment 2)."""
    df = _scored_frame()
    machine = df[~df.is_doctor]
    tau = np.percentile(machine.errors_total, 75)
    assert tau == 6.25  # 75th pct of 1..8; doctor rows (100.0) excluded
    # sanity: including doctors would corrupt it
    assert np.percentile(df.errors_total, 75) > tau


def test_exploratory_flagging_reads_frozen_tau_from_config(capsys):
    """tau must be READ from config, never recomputed on the split being scored."""
    mod = _load_harness()
    machine = _scored_frame().query("~is_doctor").reset_index(drop=True)

    # A tau no note can reach -> base rate 0. If the harness recomputed the
    # quartile from this frame it would instead report ~0.25.
    mod.report_flagging_exploratory(
        machine, {"flagging": {"exploratory_errors_total_threshold": 999.0}}, seed=42
    )
    out = capsys.readouterr().out
    assert "EXPLORATORY" in out and "NOT pre-registered §B7" in out
    assert "positive base rate = 0.000" in out

    # A frozen tau of 6.25 flags errors_total 7 and 8 -> 2/8 = 0.25.
    mod.report_flagging_exploratory(
        machine, {"flagging": {"exploratory_errors_total_threshold": 6.25}}, seed=42
    )
    assert "positive base rate = 0.250" in capsys.readouterr().out


def test_exploratory_class_gives_finite_pr_auc():
    """Non-degenerate class -> PR-AUC is finite (unlike the saturated §B7 class)."""
    d = _scored_frame().query("~is_doctor").reset_index(drop=True)
    d["is_worst_quartile"] = (d["errors_total"] >= 6.25).astype(int)
    d["_unrel"] = -d["claims_neg_combined"]
    out = pr_auc_bootstrap(d, "_unrel", "is_worst_quartile", seed=42)
    assert 0.0 < out["pr_auc"] <= 1.0
    assert out["base_rate"] == 0.25
