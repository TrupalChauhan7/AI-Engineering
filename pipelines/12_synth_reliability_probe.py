"""Fact-controlled RELIABILITY PROBE on the self-built 43-consultation dataset.

STATUS: EXPLORATORY external-validity extension. NOT pre-registered, NOT part of
RQ1/RQ2, and it touches nothing that is. It reuses the LOCKED generator, claim
verifier and reliability exactly as configured -- same prompts, same models,
temperature 0, reasoning off, num_ctx 8192 -- and wraps a new pipeline around
them. The PriMock57 code, the sealed TEST split and the pre-registration are
untouched.

THE QUESTION. RQ1 validated the reliability by rank-correlation against NOISY human
labels (rho ~0.26, ~0.48x the human ceiling). That is a lower bound, and it
cannot say which errors were caught. Here we AUTHOR the errors, so the label is
exact:

    base   the generated SOAP note, untouched
    +1H    base + 1 hand-written sentence whose subject the transcript never mentions
    +2H    base + 2 such sentences   (nested: the +1H sentence plus one more)
    +3H    base + 3 such sentences   (nested again)
    -1O    base with one genuinely transcribed fact deleted

The injected DOSE is monotone by construction -- we put the errors there. What
is NOT given is whether the reliability's combined flag count RESPONDS to that dose.
That response, measured paired within each consultation, is the endpoint.

NO LLM IN THE LABELLING LOOP: the hallucination catalogue is hand-written and
the absence test is a substring check; the omission target is chosen by token
arithmetic. See s2n.evaluation.fact_injection for the full argument, including
why the omission target is chosen INDEPENDENTLY of the verifier's fact list
(circularity avoidance) and what the fact-list-conditioned "upper bound"
diagnostic means.

DATA SAFETY. The score CSVs carry IDs, counts and verdicts only. Everything
containing clinical text -- the injection detail log and the note/fact cache --
is written under results/, which .gitignore excludes in full.

Run:  python pipelines/12_synth_reliability_probe.py --limit 3     # fast smoke
      python pipelines/12_synth_reliability_probe.py               # full 43
      python pipelines/12_synth_reliability_probe.py --no-figure
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from s2n.config import ROOT, load_config
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.evaluation.fact_injection import (
    consultation_rng,
    inject_hallucinations,
    inject_omission,
    load_catalogue,
)
from s2n.generation.generator import NoteGenerator
from s2n.reliability.flagger import verdict

BANDS = ["reliable", "review", "unreliable"]
SCORES_CSV = ROOT / "results" / "synth_probe_scores.csv"
DOSE_CSV = ROOT / "results" / "synth_probe_dose.csv"
FIG_PNG = ROOT / "results" / "synth_probe_dose.png"
DETAIL_LOG = ROOT / "results" / "synth_probe_injection_log.jsonl"  # TEXT -> gitignored
CACHE_JSON = ROOT / "results" / "synth_probe_cache.json"  # TEXT -> gitignored


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_scripts(cfg: dict) -> list[tuple[str, str]]:
    """(consultation_id, transcript) for every script, in numeric order.

    Fed VERBATIM: alternating doctor/patient turns, no speaker labels, no
    reformatting. The only edit is stripping the UTF-8 BOM some files carry,
    which is an encoding artefact rather than content.
    """
    d = ROOT / cfg["synthetic_probe"]["keys_dir"]
    paths = sorted(d.glob("*.txt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not paths:
        raise SystemExit(f"No scripts found in {d} -- copy the Keys/*.txt files there first.")
    return [(p.stem, p.read_text(encoding="utf-8-sig")) for p in paths]


def _load_cache() -> dict:
    return json.loads(CACHE_JSON.read_text()) if CACHE_JSON.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CACHE_JSON.write_text(json.dumps(cache, indent=1))


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------
def build_conditions(
    cid: str,
    base_note: str,
    transcript: str,
    cfg: dict,
    seed: int,
    catalogue: list[dict],
    facts: list[str],
) -> tuple[dict, list[dict]]:
    """The five notes for one consultation, plus the detail log for each.

    Each condition re-derives its rng from (seed, cid) so the hallucination
    permutation is identical across doses -> +1H subset of +2H subset of +3H.
    """
    sp = cfg["synthetic_probe"]
    notes: dict[str, str] = {"base": base_note}
    logs: list[dict] = []
    for k in sp["halluc_doses"]:
        note, log = inject_hallucinations(
            base_note, transcript, k, consultation_rng(seed, cid), catalogue
        )
        notes[f"+{k}H"] = note
        logs.append({**log.__dict__, "consultation": cid, "condition": f"+{k}H"})
    note, log = inject_omission(
        base_note,
        transcript,
        consultation_rng(seed, cid),
        sp["omission_match_threshold"],
        facts=facts,
    )
    notes[f"-{sp['omission_dose']}O"] = note
    logs.append({**log.__dict__, "consultation": cid, "condition": f"-{sp['omission_dose']}O"})
    return notes, logs


def score_note(
    verifier: ClaimVerifier, transcript: str, note: str, facts: list[str], cfg: dict
) -> dict:
    """The LOCKED reliability, in the same order as s2n.service.run_pipeline_staged."""
    rep = verifier.score(transcript, note)
    omit = verifier.check_omissions(note, facts)
    band = verdict({"n_unsupported": rep.n_unsupported, "n_omitted": omit.n_omitted}, cfg)
    return {
        "n_claims": rep.n_claims,
        "n_unsupported": rep.n_unsupported,
        "n_facts": omit.n_facts,
        "n_omitted": omit.n_omitted,
        "combined": band["combined"],
        "verdict_band": band["verdict"],
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean with a 95% bootstrap CI, following correlation.py's convention
    (np.random.default_rng(seed), nanpercentile at 2.5 / 97.5).

    HONEST NOTE ON THE CLUSTERING. RQ1 needed a CLUSTER bootstrap because its 285
    notes came from only 57 consultations. Here every consultation contributes
    exactly ONE paired delta per dose, so cluster == observation and the cluster
    bootstrap reduces exactly to a plain nonparametric bootstrap over the 43
    consultations. Same estimator, same seed convention -- no extra rigour is
    being claimed by the name.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def _paired(df: pd.DataFrame, condition: str, col: str) -> pd.DataFrame:
    """Per-consultation delta of ``col`` between a condition and its base."""
    base = df[df.condition == "base"].set_index("consultation")[col]
    cond = df[df.condition == condition].set_index("consultation")[col]
    return (
        pd.DataFrame({"base": base, "cond": cond})
        .dropna()
        .assign(delta=lambda d: d["cond"] - d["base"])
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(df: pd.DataFrame, logs: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    sp = cfg["synthetic_probe"]
    n_boot = sp["n_boot"]
    n_cons = df.consultation.nunique()
    omit_cond = f"-{sp['omission_dose']}O"

    print(
        f"\n{'=' * 78}\nSYNTHETIC RELIABILITY PROBE -- EXPLORATORY (not pre-registered)"
        f"\n{n_cons} consultations x {df.condition.nunique()} conditions "
        f"= {len(df)} scored notes\n{'=' * 78}"
    )
    print("Bootstrap: 1 observation per consultation per condition, so the cluster")
    print(
        f"bootstrap reduces to a plain nonparametric bootstrap over the {n_cons} " "consultations."
    )

    # ---- PRIMARY: dose-response -----------------------------------------
    print("\n--- PRIMARY: dose-response of the reliability's COMBINED flag count ---")
    print("The injected dose is monotone BY CONSTRUCTION (we planted the errors).")
    print("What is measured here is whether the reliability flag's response rises with it.")
    rows = []
    for k in sp["halluc_doses"]:
        p = _paired(df, f"+{k}H", "combined")
        mean, lo, hi = bootstrap_mean_ci(p["delta"].to_numpy(), n_boot, seed)
        u = _paired(df, f"+{k}H", "n_unsupported")
        rmean, rlo, rhi = bootstrap_mean_ci((u["delta"] / k).to_numpy(), n_boot, seed)
        rows.append(
            {
                "dose": k,
                "n": len(p),
                "mean_delta_combined": round(mean, 3),
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "recovery_per_planted_fact": round(rmean, 3),
                "recovery_lo": round(rlo, 3),
                "recovery_hi": round(rhi, 3),
            }
        )
        print(
            f"  +{k}H  mean d(combined) = {mean:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (n={len(p)})"
        )
    dose_tbl = pd.DataFrame(rows)
    monotone = dose_tbl.mean_delta_combined.is_monotonic_increasing
    print(f"  monotone increase across doses: {'YES' if monotone else 'NO'} (reported as-is)")

    # ---- hallucination recovery ------------------------------------------
    print("\n--- Hallucination recovery: new UNSUPPORTED claims per planted fact ---")
    print("  1.00 would mean every planted lie surfaced as exactly one new unsupported claim.")
    for r in rows:
        print(
            f"  +{r['dose']}H  d(n_unsupported)/dose = {r['recovery_per_planted_fact']:+.2f}"
            f"  95% CI [{r['recovery_lo']:+.2f}, {r['recovery_hi']:+.2f}]"
        )

    # ---- omission detection ----------------------------------------------
    print(f"\n--- Omission detection ({omit_cond}): the reliability's known blind spot ---")
    p = _paired(df, omit_cond, "n_omitted")
    attempted = logs[(logs.condition == omit_cond) & (logs.removed.map(len) > 0)]
    attempted_ids = set(attempted.consultation)
    p_att = p[p.index.isin(attempted_ids)]
    det = p_att["delta"] > 0
    mean, lo, hi = bootstrap_mean_ci(det.to_numpy(float), n_boot, seed)
    print("  Target = a grounded sentence from the SUBJECTIVE/OBJECTIVE sections only")
    print("  (pre-declared before the full run). The omission axis measures whether a")
    print("  note COVERS the clinically important facts the patient stated, so deleting")
    print("  Plan/advice would test it out of scope and a non-detection would say nothing")
    print("  about the blind spot. Restricting to S/O makes this a valid blind-spot test.")
    print("  HEADLINE (target chosen INDEPENDENTLY of the verifier's fact list -- see")
    print("  fact_injection docstring on circularity avoidance; the restriction is by note")
    print("  SECTION, which the verifier never sees):")
    print(
        f"    n_omitted increased in {int(det.sum())}/{len(p_att)} = {mean:.3f} "
        f"95% CI [{lo:.3f}, {hi:.3f}]"
    )
    dmean, dlo, dhi = bootstrap_mean_ci(p_att["delta"].to_numpy(), n_boot, seed)
    print(f"    mean d(n_omitted) = {dmean:+.2f}  95% CI [{dlo:+.2f}, {dhi:+.2f}]")
    skipped = len(p) - len(p_att)
    if skipped:
        print(
            f"    ({skipped} consultation(s) had no S/O sentence grounded at or above "
            f"threshold {sp['omission_match_threshold']} -- no omission injected, excluded)"
        )
    cand = logs[logs.condition == omit_cond]["n_candidates"]
    print(
        f"    eligible S/O targets per consultation: median {cand.median():.0f} "
        f"(min {cand.min()}, max {cand.max()})"
    )

    in_facts = set(attempted[attempted.in_verifier_facts == True].consultation)  # noqa: E712
    p_up = p_att[p_att.index.isin(in_facts)]
    print("  DIAGNOSTIC -- UPPER BOUND (restricted to targets that happened to also appear")
    print("  in the verifier's own fact list; NOT the headline, it is partly circular):")
    if len(p_up):
        det_u = p_up["delta"] > 0
        umean, ulo, uhi = bootstrap_mean_ci(det_u.to_numpy(float), n_boot, seed)
        print(
            f"    n_omitted increased in {int(det_u.sum())}/{len(p_up)} = {umean:.3f} "
            f"95% CI [{ulo:.3f}, {uhi:.3f}]"
        )
    else:
        print("    no target overlapped the verifier's fact list -- diagnostic unavailable")

    # ---- verdict-band shift ----------------------------------------------
    print(
        "\n--- Verdict-band shift vs base (config bands: reliable <= "
        f"{cfg['reliability']['reliable_max']} | review | unreliable >= {cfg['reliability']['unreliable_min']}) ---"
    )
    rank = {b: i for i, b in enumerate(BANDS)}
    base_band = df[df.condition == "base"].set_index("consultation")["verdict_band"]
    for cond in [f"+{k}H" for k in sp["halluc_doses"]] + [omit_cond]:
        cb = df[df.condition == cond].set_index("consultation")["verdict_band"]
        j = pd.DataFrame({"base": base_band, "cond": cb}).dropna()
        d = j["cond"].map(rank) - j["base"].map(rank)
        print(
            f"  {cond:<5} worse {int((d > 0).sum()):>3} | same {int((d == 0).sum()):>3} "
            f"| better {int((d < 0).sum()):>3}   "
            f"({' , '.join(f'{b}:{int((cb == b).sum())}' for b in BANDS)})"
        )
    print(f"  base  ({' , '.join(f'{b}:{int((base_band == b).sum())}' for b in BANDS)})")

    # ---- false-positive floor --------------------------------------------
    print("\n--- False-positive floor: combined count on the UNTOUCHED base notes ---")
    c = df[df.condition == "base"]["combined"]
    print(
        f"  min {c.min()} | q1 {c.quantile(.25):.1f} | median {c.median():.1f} | "
        f"q3 {c.quantile(.75):.1f} | max {c.max()} | mean {c.mean():.2f}"
    )
    print("  This is the floor any dose effect has to clear: the reliability already raises")
    print("  this many flags on notes we planted nothing in. The bands stay PriMock-locked")
    print("  (config reliability.*, DEV-calibrated); they are NOT recalibrated to this dataset,")
    print("  so a low floor simply means band crossings are rare here — reported, not tuned.")

    # ---- injection bookkeeping -------------------------------------------
    sk = logs[logs.condition.str.endswith("H")]
    if len(sk):
        n_skip = sk.drop_duplicates("consultation").skipped_keywords.map(len)
        print(
            f"\n--- Catalogue filter: mean {n_skip.mean():.1f} of "
            f"{len(load_catalogue(cfg))} entries skipped per consultation "
            "(keyword present in transcript) ---"
        )
        print("  A skip means the keyword occurs in the transcript, so planting that")
        print("  sentence would not be a hallucination. Passing the check proves the")
        print("  planted fact is absent AS STATED; paraphrase collisions are possible")
        print("  but rare.")
    return dose_tbl


def make_figure(dose_tbl: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = dose_tbl["dose"].to_numpy()
    y = dose_tbl["mean_delta_combined"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.fill_between(x, dose_tbl["lo"], dose_tbl["hi"], alpha=0.2, label="95% CI")
    ax.plot(x, y, "o-", label="mean Δ combined")
    ax.axhline(0, lw=0.8, color="grey", ls="--")
    ax.set_xticks(x)
    ax.set_xlabel("injected hallucinations per note (dose)")
    ax.set_ylabel("Δ combined flag count vs base (paired)")
    ax.set_title(
        "Reliability response to planted hallucinations\n(exploratory; self-built dataset)"
    )
    ax.legend()
    fig.tight_layout()
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N consultations")
    ap.add_argument("--seed", type=int, default=None, help="override config synthetic_probe.seed")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["synthetic_probe"]["seed"]
    catalogue = load_catalogue(cfg)
    scripts = load_scripts(cfg)
    if args.limit:
        scripts = scripts[: args.limit]
    print(f"EXPLORATORY synthetic reliability probe — {len(scripts)} consultation(s), seed {seed}")
    print(
        f"  generator {cfg['llm']['model']} / {cfg['generation']['prompt_version']} | "
        f"verifier {cfg['claim_verifier']['model']} (both LOCKED, reused as-is)"
    )

    generator, verifier = NoteGenerator(cfg), ClaimVerifier(cfg)
    cache = _load_cache()
    done = pd.read_csv(SCORES_CSV) if SCORES_CSV.exists() else pd.DataFrame()
    scored = set(zip(done.consultation, done.condition)) if len(done) else set()

    rows: list[dict] = list(done.to_dict("records")) if len(done) else []
    all_logs: list[dict] = []
    for n, (cid, transcript) in enumerate(scripts, 1):
        entry = cache.setdefault(cid, {})
        if "note" not in entry:
            print(f"[{n}/{len(scripts)}] {cid}: generating base note ...")
            entry["note"] = generator.generate(transcript, cid).note
            _save_cache(cache)
        if "facts" not in entry:
            print(f"[{n}/{len(scripts)}] {cid}: extracting transcript facts ...")
            entry["facts"] = verifier.decompose_transcript(transcript)
            _save_cache(cache)

        notes, logs = build_conditions(
            cid, entry["note"], transcript, cfg, seed, catalogue, entry["facts"]
        )
        for log, cond in zip(logs, [k for k in notes if k != "base"]):
            log["note"] = notes[cond]
        all_logs.extend(logs)

        for cond, note in notes.items():
            if (cid, cond) in scored:
                continue
            print(f"[{n}/{len(scripts)}] {cid}/{cond}: scoring with the locked reliability ...")
            rows.append(
                {
                    "consultation": cid,
                    "condition": cond,
                    "dose": _dose_of(cond),
                    **score_note(verifier, transcript, note, entry["facts"], cfg),
                }
            )
            df = pd.DataFrame(rows)
            SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(SCORES_CSV, index=False)

    df = pd.DataFrame(rows)
    df = df[df.consultation.isin({c for c, _ in scripts})].reset_index(drop=True)
    DETAIL_LOG.write_text("\n".join(json.dumps(r) for r in all_logs) + "\n")

    logs_df = pd.DataFrame(all_logs)
    dose_tbl = report(df, logs_df, cfg, seed)
    dose_tbl.to_csv(DOSE_CSV, index=False)
    if not args.no_figure:
        make_figure(dose_tbl)
        print(f"\nFigure  -> {FIG_PNG.relative_to(ROOT)}")
    print(f"Scores  -> {SCORES_CSV.relative_to(ROOT)}  (IDs/counts/verdicts only)")
    print(f"Dose    -> {DOSE_CSV.relative_to(ROOT)}")
    print(f"Detail  -> {DETAIL_LOG.relative_to(ROOT)}  (CONTAINS TEXT — gitignored)")


def _dose_of(condition: str) -> int:
    """Signed dose: +k for k hallucinations, -k for k omissions, 0 for base."""
    m = re.match(r"([+-])(\d+)([HO])", condition)
    return 0 if not m else int(m.group(2)) * (1 if m.group(1) == "+" else -1)


if __name__ == "__main__":
    main()
