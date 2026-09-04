# PROJECT CHECKPOINT — The Academic ↔ Further-Development Line

**Project:** FM-05 Speech-to-Clinical-Note — *Clarion* · A Reliability Flag for AI-Generated Consultation Notes
**Author:** Trupal Chauhan
**Context:** MSc dissertation-scale module project, ECS-8060 "AI Engineering", Queen's University Belfast (QUB)
**Checkpoint written:** 3 September 2026
**Status at this checkpoint:** Academic project COMPLETE — submitted and defended in a live viva/demo, August 2026.

> This is a public, portfolio-facing record. A fuller internal version is kept privately.

---

## 0. What this document is (read this first)

This is a **line in the sand.** Everything described here was built as an MSc module project under academic requirements, a locked research protocol, and a fixed deadline, by a single part-time author. It was designed to answer two narrow research questions and satisfy a marking rubric — **not** to be a shippable product.

**Everything before this checkpoint = the academic project.**
**Everything after this checkpoint = independent further development** beyond the academic scope, taking the work toward a genuinely useful, portfolio-grade, potentially commercial system.

The purpose is so that anyone — an employer, a collaborator, or a future version of me — can see exactly what was done for the degree and why, and where the academic project ended and independent development began. Nothing after this line is academic work; nothing before it should be judged as a finished product.

---

## 1. The project in one paragraph

An AI can turn a doctor–patient consultation recording into a structured clinical note (a "SOAP" note), but a fluent note is not necessarily a *faithful* one — it can state facts that were never said (**hallucinations**) or drop facts that were (**omissions**), both dangerous in medicine. This project built a **reliability flag**: a pipeline that transcribes audio (Whisper), writes a note (MedGemma), then runs a **claim-verifier** (a custom method on a stock Llama-3.1) that checks every claim in the note against the transcript and separately lists what the note omitted — turning the result into a traffic-light verdict (reliable / review / unreliable). The **flag is the research contribution**; the working app ("Clarion") is the stage it runs on. The whole system runs locally and free.

---

## 2. The academic brief and how it was satisfied

The work was assessed entirely by coursework, staged as a proposal, a progress report (with early results), and a final submission comprising a written report, a demo video, a recorded presentation, the working code, and a live viva/demo. The marking emphasis was on **presentation quality, results & methodology, and critical analysis**, with the standing requirement that every choice be **defensible in a viva** — the "why" mattering as much as the "what."

The assigned topic (FM-05, "Speech-to-Clinical-Note") asked for a pipeline (consultation audio → Whisper → LLM → SOAP note) with a research twist: measure how often notes hallucinate/omit facts and build a lightweight flag for unreliable notes.

The required method vocabulary — used throughout — was: evaluation-first design; model selection as *hard constraints → public benchmarks → task-specific evals*; AI-as-a-judge; versioned prompts.

**How each requirement was met:**

| Requirement | How it was met |
|---|---|
| Speech-to-note pipeline | 4-stage local pipeline: Whisper → MedGemma → claim-verifier → reliability flag |
| A research contribution | The reliability flag + a pre-registered evaluation of *which* faithfulness signal best flags unreliable notes |
| Model selection (course method) | Hard constraints (local, free, ≤8 GB) → benchmark shortlist → a Stage-1 smoke test on real transcripts |
| AI-as-a-judge | An LLM judge in 3 prompt versions as a baseline tier |
| Versioned prompts | Every prompt is a numbered file; nothing edited in place — full provenance |
| Multimodal capability | The Whisper transcription stage (audio → text; channel mixing; temperature-fallback schedule) |
| Fine-tuning | A LoRA experiment, reported as a rigorous **negative** result |
| Local inference | Entirely local/free via Ollama within an 8 GB memory budget (runs on lab-grade hardware) |
| Working demonstrator | "Clarion" — a FastAPI + Next.js app streaming the pipeline stage-by-stage over SSE |
| Critical analysis | Honest reporting of nulls, a negative result, a metric collapse the design was built to catch, and named blind spots |
| Working code | ~5,200 lines of Python (library + 12 numbered pipelines), 18 test files, a 20-file TypeScript front end, config-driven, leak-safety enforced by an automated test |

---

## 3. What was built (the system)

### 3.1 Architecture — four stages, one library

```
audio ──▶ Whisper (large-v3-turbo) ──▶ transcript
transcript ──▶ MedGemma-4B ──▶ SOAP note
note + transcript ──▶ claim-verifier (Llama-3.1-8B) ──▶ unsupported + omitted counts
counts ──▶ reliability flagger ──▶ 🟢 reliable / 🟡 review / 🔴 unreliable
```

The core is a Python library, **`s2n`**, with a single UI-agnostic entry point (`service.run_pipeline` / `run_pipeline_staged`) that takes audio *or* a transcript and returns plain JSON. The demo app is a thin layer on top.

### 3.2 Code map

- `src/s2n/transcription/` — Whisper wrapper (`whisper_asr.py`), in-house WER (`wer.py`)
- `src/s2n/generation/` — note generator (`generator.py`) + prompt loader (`prompts.py`)
- `src/s2n/llm/` — backend-agnostic Ollama client (`client.py`) — the one place any text model is called
- `src/s2n/evaluation/` — `metrics.py` (ROUGE-L / BERTScore / Levenshtein), `summac_metric.py`, `judge.py`, `claim_verifier.py` (**the star**), `correlation.py` (Spearman + cluster bootstrap), `track_a.py` (RQ1 harness), `rq2.py`, `targets.py`, `ceiling_flagging.py`, `fact_injection.py`
- `src/s2n/reliability/` — `flagger.py` (turns counts into the verdict)
- `src/s2n/finetune/` — MLX LoRA (`mts.py`, `mlx_generate.py`)
- `src/s2n/data/` — leak-safe loaders, the frozen split, TextGrid parsing, ID canonicalisation
- `pipelines/00 → 12` — runnable stages: dataset report → transcribe → generate → evaluate → one-shot TEST → fine-tune → self-built probe
- `prompts/` — versioned generator + judge + claim-verifier prompts
- `config/config.yaml` — one control panel (models, verdict bands, injection catalogue)
- `app/api/` (FastAPI + SSE) and `app/web/` (Next.js 15 / TypeScript / Tailwind) — the Clarion demo
- `tests/` — 18 test files incl. `test_leak_safety.py` (the structural firewall)

### 3.3 The models (all local via Ollama, ≤8 GB)

| Role | Model | Params | Ships in product? |
|---|---|---|---|
| Transcribe (ears) | Whisper large-v3-turbo | 809M | Yes |
| Generate (writer) | MedGemma-4B | 4B | Yes |
| Verify (checker ★) | Llama-3.1-8B (stock; custom method) | 8B | Yes |
| Judge (baseline) | Atla Selene-Mini (Llama-3.1-8B family) | 8B | **No — research baseline only** |

Model independence rule ("rule 6"): the verifier is a *different model family* from the generator, so the checker never grades its own writing.

---

## 4. The research (questions, methods, results)

### 4.1 Data

- **PriMock57** (primary) — 57 mock GP consultations; audio, human transcripts (.TextGrid), clinician notes, and a **human-evaluation set of 285 clinician-rated notes** (57 doctor-written + 228 machine-written across 4 models) marking made-up/missing facts — a ready-made answer key. Split **10 development / 47 held-out test, by whole consultation, seed 42, frozen to a split file.**
- **Self-built dataset** (evaluation-only) — 43 synthetic consultations (LLM-authored scripts; 16 real-mic recordings + 27 text-to-speech), used to probe the flag against *exactly known* planted errors. Never used for training.
- **MTS-Dialog** — ~1,200 dialogue→note pairs, used only for the fine-tuning experiment (disjoint from the test sets).

### 4.2 RQ1 — can an automatic score flag unreliable notes better than the standard metrics?

**Method:** compare faithfulness signals against the clinician error count using aligned Spearman ρ, with 95% confidence intervals from a **cluster bootstrap over whole consultations** (the 5 notes of one consultation aren't independent). Primary population = **machine-written notes only** (the clinician's own note is a near-error-free outlier that flips significance). The held-out test was **scored exactly once.**

**Held-out TEST result (machine-only, n = 188 notes / 47 consultations):**

| Rank | Signal | TEST ρ [95% CI] | DEV ρ |
|---|---|---|---|
| 1 | **Claim-verifier (combined)** | **0.26 [0.08, 0.42]** ✅ | 0.67 |
| 2 | Levenshtein | 0.25 [0.08, 0.40] | 0.25 |
| 3 | Selene judge | 0.20 [0.05, 0.34] | 0.03 |
| 4 | SummaC | 0.19 [0.01, 0.36] | −0.11 |
| 5 | ROUGE-L | 0.13 [−0.06, 0.33] | 0.30 |
| 6 | **BERTScore** | **0.06 [−0.12, 0.23]** | 0.56 |

**Headline:** BERTScore led on development (0.56) but **collapsed on unseen data (0.06)** — its dev strength was leverage from an easy outlier. The **source-grounded claim-verifier was the most robust** (0.26, ~0.48× the human agreement ceiling of 0.541) and the **only signal deployable without a reference note.** No signal statistically dominates (all paired-difference CIs span zero). The pre-registration/one-shot discipline is *what exposed* BERTScore's non-generalisation — the central methodological point.

### 4.3 RQ2 — how much do speech-recognition errors degrade the notes?

**Method:** a paired within-consultation design — for the same consultation, generate the note from the human transcript vs. Whisper's transcript, changing nothing else, and compare quality. Dose measure = content-WER.

**Result (dev pilot, n = 10):** a **clean null.** Across 8–21% content-WER, note quality did not measurably change (BERTScore Δ = +0.0013 [−0.0042, +0.0070]; dose-response ρ = −0.16, p = 0.65). Confirmed real, not an artefact: the two notes genuinely differ yet score equally well.

### 4.4 Fine-tuning — a rigorous negative result

LoRA (Apple MLX) on MTS-Dialog **hurt**: BERTScore Δ(ft − base) = −0.039 [−0.054, −0.024]; faithfulness change null; note length degenerate (1–751 words vs a base median ~195). Cause: a task/shape mismatch — MTS-Dialog dialogues are ~64 words with ~14-word target notes, ~20–25× smaller than a real ~1,545-word consultation. **Not adopted.** Reported honestly as a contribution.

### 4.5 Self-built probe — measured strengths and blind spot

On data with exactly known planted errors: the flag detected fabrications **almost one-for-one** (+0.91 / +2.02 / +3.19 flags at 1/2/3 planted facts, all clear of zero) but caught only **~36% of omissions** (15/42). Strong fabrication detector, weak omission detector — the same asymmetry seen on PriMock57, now quantified. Thresholds are population-specific (median 4 flags on clean synthetic notes vs. 9 on PriMock).

---

## 5. Methodology and guidelines followed

The project was run under a **locked pre-registration** (protocol + 5 amendments). The governing rules:

1. **No data leakage** — the generator sees only the transcript/audio; the clinician notes are answers used only for scoring. Enforced *structurally* (separate loaders + an AST + subprocess leak-test).
2. **Fixed split** — 10 dev / 47 test, by consultation, seed 42, frozen; iterate on dev only.
3. **One-shot test** — the held-out test scored **exactly once** (now spent).
4. **≤3 judge rubric versions** on dev (all 3 used).
5. **Statistics** — aligned Spearman ρ; cluster bootstrap over 57 consultations (5,000×); paired Δρ recomputed inside each resample; a 95% CI excluding 0 ⇒ "distinguishable."
6. **Three-tier fair-fight baselines** — reference-based (ROUGE-L/BERTScore/Levenshtein), source-grounded non-LLM (SummaC), source-grounded LLM-judge.
7. **Pre-registered amendments** (before the test): machine-only primary population; content-WER dose + Whisper fallback schedule; the claim-verifier combined score promoted to a pre-registered test metric; an exploratory worst-quartile flag added alongside.
8. Deterministic decoding (temperature 0); generator reasoning off; judge a different family from the generator; a raised context window for all test inference; config-driven paths.

**Two-track design:** Track A (validate the reliability flag → a defensible RQ1 even if the pipeline broke) and Track B (own pipeline + RQ2 + fine-tuning), which de-risked the outcome.

---

## 6. Major decisions and why

| Decision | Why | Status |
|---|---|---|
| Frame RQ1 as "which signal flags best," not "my judge wins" | Removes over-claiming; a null becomes a legitimate result | LOCKED |
| Two tracks (validate flag / own pipeline) | Guarantees a defensible RQ1 even if generation breaks | DONE |
| MedGemma-4B as generator (Llama-8B as control) | Medically tuned, faster, won a Stage-1 head-to-head | LOCKED |
| Verifier = Llama-3.1-8B (different family) | The checker must not grade its own writing | LOCKED |
| Judge = Atla Selene-Mini | Purpose-built 8B evaluator = fairest strong baseline | LOCKED |
| Reimplement SummaC (not the pip package) | The `summac` package broke the environment | DONE |
| Machine-only primary population | The doctor outlier flips significance | LOCKED |
| Promote the claim-verifier to a test metric | First source-grounded signal to match BERTScore on dev | CONFIRMED #1 on test |
| Fine-tuning **not** adopted | Negative transfer + overfit risk on a small set | DECIDED |
| Demo = a real built app (Clarion), not slideware | "The app is the stage"; demo-speed lane isolated from the sealed research config | DONE |
| Rename "alarm" → "reliability flag" / "claim-verifier" | Clearer, less alarmist terminology | DONE |

---

## 7. What was achieved at this checkpoint

- A complete, working, leakage-safe speech-to-note pipeline running fully local and free.
- A reliability flag validated with pre-registered discipline, reaching ~half the human agreement ceiling and the only tier deployable without a reference note.
- A defensible RQ1 result that *reframed* the development story (caught BERTScore's non-generalisation) rather than merely confirming it.
- A clean RQ2 null and a rigorous fine-tuning negative result — both honestly reported.
- A self-built dataset quantifying the flag's strengths and its named blind spot.
- A polished demonstrator app (Clarion) with a stage-by-stage streaming UI.
- All final deliverables submitted and the live viva/demo completed (August 2026).
- Scale: ~5,200 lines of Python, 18 test files, a 20-file TypeScript front end, 12 numbered reproducible pipelines.

---

## 8. Limitations, constraints, and restrictions of the academic version

These are the boundaries the academic project lived inside — and the natural starting agenda for further development.

**Research / scope:**
- Single primary dataset (PriMock57); results not yet shown to generalise to other data or accents.
- RQ2 and fine-tuning are development-scale (n = 10) pilots.
- The self-built probe is synthetic; its planted fabrications are obvious, so its detection rate is an *upper bound* for subtler real errors.
- The one-shot test rule means the held-out test cannot be re-used — new evaluation needs new held-out data.

**Known technical limitations:**
- The flag catches fabrications well but **misses ~two-thirds of omissions** — the single biggest weakness.
- Verdict **thresholds are population-specific** and must be recalibrated per deployment.
- A **near-empty note can look "faithful"** — the flag needs a length/coverage guard.
- The system measures **faithfulness, not relevance** — off-topic audio isn't rejected (no "is this a consultation?" gate).
- No speaker diarization (RQ2 used mixed audio).
- Individual flags can be wrong; only the note-level *count* is validated.

**Product / engineering gaps (never in academic scope):**
- No authentication, multi-user support, persistence, or audit trail.
- No packaging/deployment story (runs from source).
- Not tested for clinical safety, regulatory compliance, or real EHR integration.
- No performance/scale work; single-request, single-machine.

**Practical constraints during the project:** solo, part-time, first MSc, hard fixed deadline; free/local models only, ≤8 GB (lab-hardware ceiling) — deliberately no GPT-4-class models.

---

## 9. Making this checkpoint a real, restorable marker

To turn this written boundary into a locked technical one:

1. Commit the final academic state as one clean commit.
2. **Tag it `v1.0-academic`** — the restorable technical dividing line.
3. Start all further development on a **new branch** (e.g. `develop`) above that tag, so the academic version is never overwritten.

Everything at or before the `v1.0-academic` tag is the academic project; everything after it is independent further development.

---

## 10. THE CHECKPOINT LINE

> **▲ ABOVE THIS LINE — the MSc ECS-8060 academic project (Queen's University Belfast).**
> Built mid-2026 under the module's requirements and a locked pre-registration; submitted and defended in a live viva/demo, August 2026. Constrained by academic rules, a fixed deadline, a solo part-time effort, and free/local ≤8 GB models. Its purpose was to answer two research questions and satisfy a marking rubric — **not** to be a finished product.
>
> **▼ BELOW THIS LINE — independent further development (from September 2026 onward).**
> No longer bound by the academic scope, the pre-registration, or the coursework limits. The goal is a genuinely useful, portfolio-grade, potentially commercial system — starting from the limitations in §8. New research uses **new** held-out data (the academic test set is spent). All new work lives on a separate branch above the `v1.0-academic` tag.

This document is the human-readable record of that boundary.
