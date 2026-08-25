# Speech-to-Clinical-Note (s2n)

Turn a GP consultation recording into a SOAP note — then **measure how faithful
that note is and raise an alarm on unreliable ones.** The alarm is the star; the
app is just the stage.

**Module:** ECS-8060 AI Engineering · **Author:** Trupal Chauhan · Dataset: PriMock57

## Research questions
- **RQ1 (main):** How faithful are the AI-generated notes, and can we *automatically
  flag* the unreliable ones? Does an LLM-as-a-judge faithfulness score beat
  traditional metrics (ROUGE, BERTScore, Levenshtein) at flagging?
- **RQ2:** How much do speech-to-text (Whisper) errors degrade the notes?

## Pipeline
```
audio ──Whisper──▶ transcript ──LLM+prompt──▶ SOAP note ──evaluation──▶ ALARM 🚨
                                                   │
                        validated against PriMock57 human ratings (RQ1)
```

## Setup
```bash
conda activate ds                     # your existing environment
pip install -r requirements.txt       # transcription + metrics + data tools
pip install -e .                      # makes `import s2n` work everywhere

# get the 2 GB audio (one time, on your Mac):
cd primock57 && python3 download_audio.py && cd ..
```
Copy `.env.example` to `.env` and fill it in once you pick an LLM backend
(all free options — course API or local Ollama).

## Run
```bash
make transcribe     # 01 · Whisper over the audio (RQ2)
make generate       # 02 · transcripts → SOAP notes
make evaluate       # 03 · metrics + judge + alarm, validated vs humans (RQ1)
make test           # sanity check the skeleton
```

## Reproduce the pipeline (00 → 11)
Every stage is a standalone script in `pipelines/`, run in order with
`conda activate ds` and Ollama up (`medgemma:4b`, `llama3.1:8b`,
`atla/selene-mini`). All settings come from `config/config.yaml`.

```bash
python pipelines/00_dataset_report.py        # dataset inventory + integrity
python pipelines/01_transcribe.py            # Whisper over the audio          (RQ2)
python pipelines/02_generate_notes.py        # transcripts → SOAP notes
python pipelines/03_evaluate.py              # metrics + judge + alarm         (RQ1)
python pipelines/03a_endpoint_selection.py   # pick the human-label endpoint
python pipelines/03b_summac_validity_gate.py # SummaC-ZS sanity gate
python pipelines/03c_track_a_dev.py          # Track A DEV: baselines + judge v1.0
python pipelines/03d_judge_v11_dev.py        # judge rubric v1.1 (count-to-count)
python pipelines/03e_judge_selene_dev.py     # judge v1.2 Selene-Mini — FROZEN
python pipelines/04_transcribe_dev.py        # DEV ASR + WER                   (RQ2)
python pipelines/05_rq2_paired_dev.py        # RQ2 paired DEV pilot (§C)
python pipelines/06_claim_verifier_dev.py    # claim verifier: incorrectness axis
python pipelines/07_claim_verifier_omissions_dev.py   # + omission axis
python pipelines/08_claim_verifier_final_dev.py       # LOCKED verifier config
python pipelines/09_format_mts.py            # MTS-Dialog → LoRA training data (Track B)
python pipelines/10_finetune_eval_dev.py     # base vs LoRA fine-tuned DEV eval
python pipelines/11_test_oneshot.py          # DEV dry-run of the one-shot harness
```

**⚠ The held-out TEST split is scored exactly once** (pre-registration §B5/§D)
and **has already been run**. `pipelines/11_test_oneshot.py --split test` is
guarded: it refuses to re-score while `results/oneshot_test_scores.csv` exists.
Use `--split test --no-refresh` to re-print the analysis from the recorded
scores. Committed results are scores-only
(`results/oneshot_test_scores_metrics.csv`, `oneshot_test_correlation.csv`);
the full artefacts stay local because they embed transcripts and gold notes.

## Demo — Clarion
A **FastAPI + Next.js** app in `app/`: pick a consultation (or upload audio) and
watch the transcript stream in, the SOAP note draft itself, and a hairline sweep
down the note flagging every claim the recording does not support.

```
app/
  api/    FastAPI over s2n.service — /api/samples, /api/analyze (SSE)
  web/    Next.js 15 + TypeScript + Tailwind 4, Framer Motion / GSAP / Lenis
```

### Run the demo
```bash
# one time: cache the sample consultations (runs the real pipeline; slow)
make samples          # -> app/web/public/samples/*.json   (GITIGNORED)

make demo             # FastAPI on :8000 + Next.js on :3000
```
Then open **http://localhost:3000**. Needs Ollama running with `medgemma:4b` and
`llama3.1:8b`, plus Node 20+ (`make demo` picks up `~/.local/node/bin`).

Selecting a **sample** replays a cached result instantly — no models run, so the
demo is reproducible on camera. **Uploading audio** runs the real local pipeline
and streams each stage over SSE into the same animation, with a live per-stage
elapsed counter so a slow stage never looks frozen.

**The demo's ASR is not the research ASR.** RQ2 is sealed on
`transcription.model: large-v3-turbo`, which takes minutes per consultation on
CPU. The live-upload path alone swaps in `demo.transcription_model: base`
(config), which transcribes a one-minute clip in under two seconds — a ~55s clip
goes from upload to verdict in about **13 seconds**. Cached samples and every
research number still come from the sealed model; nothing in `demo:` can move a
result. The API also warms Whisper and both Ollama models at startup (~10-17s),
so the first upload is not the slow one.

> **Data safety:** `app/web/public/samples/` holds transcript and note text and is
> gitignored. Regenerate it locally; never commit it.

### Using the pipeline directly
`s2n.service.run_pipeline()` is UI-agnostic — audio (or a transcript) in, a plain
JSON-serialisable dict out:

```python
from s2n.service import run_pipeline
result = run_pipeline(audio_path="results/mixed_audio_dev/day1_consultation02.wav")
print(result["alarm"]["verdict"], "—", result["alarm"]["score_note"])
```
`run_pipeline_staged()` yields the same work one stage at a time, which is what
the SSE endpoint streams.

## Layout
```
config/        one control panel (config.yaml) — every setting lives here
src/s2n/       the code library (the engine)
  data/          load PriMock57
  transcription/ Whisper wrapper                 (RQ2)
  generation/    transcript + prompt → note
  evaluation/    metrics + LLM-judge + highlights + correlation  ← the heart
  alarm/         reliability flagger             ← the star
  llm/           backend-agnostic client (swap freely, stay free)
prompts/       versioned prompts (v1.0, v1.1 …)
pipelines/     runnable scripts (01→02→03)
experiments/   notebooks + experiment configs
results/       generated outputs (gitignored)
app/           demo front-end (the stage)
tests/         unit tests
docs/          architecture + decision records (ADRs)
primock57/     dataset
A1_plan/ A2_progress/ A3_final/   coursework deliverables
```

## Principles
1. **Separation of concerns** — library vs pipelines vs app vs deliverables.
2. **Reproducibility** — pinned deps, config-driven, versioned prompts, logged results.
3. **Evaluation-first** — the evaluation harness is central, not bolted on.

## Cost
Runs **entirely free**: Whisper is open-source and local; metrics are open-source;
the LLM uses a free backend (course API or local Ollama). No paid APIs required.

## Running on Windows (cross-platform)

The Python code is OS-agnostic (`pathlib` throughout; audio mixing is pure Python). To run on
Windows — e.g. an RTX 5060 lab laptop — install Python 3.10+, Node 20+, **Ollama for Windows**, and
**ffmpeg** on `PATH` (Whisper needs it: `winget install Gyan.FFmpeg`), then:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
ollama pull medgemma:4b ; ollama pull llama3.1:8b ; ollama pull atla/selene-mini
pytest
python scripts/dev.py     # demo: FastAPI :8000 + Next.js :3000 (cross-platform; replaces `make demo`)
```

**Note:** the LoRA fine-tuning pipelines (`10`, `12_finetune_train`) use Apple **MLX** and run on
Apple-Silicon Macs only; that experiment is complete and its results are in `results/finetune/`.
Everything else — transcription, generation, the alarm, and the demo — runs on any OS.
