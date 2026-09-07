# Speech-to-Clinical-Note (s2n)

Turn a GP consultation recording into a SOAP note — then **check every claim in
that note against what was actually said, and flag the notes that are
unreliable.** The reliability flag is the star; the app is just the stage.

**Module:** ECS-8060 AI Engineering · **Author:** Trupal Chauhan · Dataset: PriMock57

This repo holds two things: the **research** (the reliability method, validated on
PriMock57 human ratings) and the **product** built around it — *Clarion*, a local
web app that runs the pipeline and, around it, an **audit trail** (every run is
stored with the exact models and prompt versions that produced it), a **reviewer
dashboard** at `/audit`, request-id tracing and per-stage latency **metrics** for
observability, and **one-command packaging** (`make run`, or Docker). The product
is clinical-only. To run it, see **[DEPLOY.md](DEPLOY.md)**.

---

## 🚀 Quick start (the easy way — no experience needed)

You do **not** need to understand the code to run this. One file does everything:
it installs what's missing, downloads the AI models, and opens the app in your
browser.

**On Windows** (e.g. the lab laptops)
1. Double-click **`SETUP_WINDOWS.bat`** (it's in this folder).
2. A black window opens and works through 6 steps on its own. The first run
   downloads ~8 GB of AI models, so give it 20–40 minutes on a fast connection. ☕
3. A second small window titled **"Ollama Engine"** pops up — **leave it open.**
   That's the brain running in the background.
4. When it's done, your browser is ready at **http://localhost:3000**.

**On Mac**
1. Double-click **`setup_mac.command`** (or in a terminal: `./setup_mac.command`).
2. Same idea — it installs anything missing and launches the app.
3. Open **http://localhost:3000**.

That's the whole thing. Everything below is for people who want to look under the
hood or reproduce the research results.

> 💡 **University lab laptops wipe themselves every night.** Nothing you install
> survives to tomorrow. That's normal — just double-click the setup file again at
> the start of each session and it rebuilds everything from scratch.

---

## 🧰 What you need before you start

The setup file handles almost everything, but two base tools must already be on
the machine (they usually are — the lab PCs and most Macs ship with them):

| Tool | Check it's there | If missing |
|---|---|---|
| **Python 3.10+** | run `python --version` | Install from [python.org](https://www.python.org/downloads/) — tick *"Add to PATH"* |
| **Node.js 20+** | run `node --version` | Install from [nodejs.org](https://nodejs.org) (LTS version) |

The setup file installs the rest for you, no admin rights needed:

- **ffmpeg** — lets the app read audio files (downloaded into this folder).
- **Ollama** — runs the AI models locally (portable version, no installer).
- **AI models (Ollama)** — the app needs two: `medgemma:4b` (writes the note) and
  `llama3.1:8b` (the claim verifier / reliability check). A third, `atla/selene-mini`,
  is pulled only to *reproduce the research* (the scoring judge) — not needed to run
  the app. ~8 GB total for all three.
- **Python libraries** — everything in `requirements.txt`.

Everything runs **100% free and offline** once downloaded — no paid APIs, no
cloud, no API keys. Your audio never leaves the machine.

---

## 🛟 If something goes wrong

These are the exact problems we hit and how to fix them. Most are one line.

| What you see | What it means | Fix |
|---|---|---|
| `'ollama' is not recognized` | The window opened before Ollama was installed | Close it, open a **new** Command Prompt, try again — or just re-run `SETUP_WINDOWS.bat` |
| `No such file or directory: 'ffmpeg'` | ffmpeg isn't reachable | Re-run the setup file; it drops `ffmpeg.exe` into this folder |
| `Connection refused ... 11434` | The "Ollama Engine" window got closed | Reopen it: `SETUP_WINDOWS.bat` starts it again |
| `Address already in use ... 8000` | A previous run is still holding the port | Close old black windows, or restart the machine, then re-run |
| `winget is not recognized` | The lab PC has no installer helper | You don't need it — the setup file downloads tools directly |
| It pasted as one long broken line | You pasted many lines into Command Prompt at once | **Don't copy-paste steps** — just double-click `SETUP_WINDOWS.bat` |

Still stuck? The setup file is safe to run again as many times as you like — it
skips anything already done.

---

## 🔬 For the marker: research questions & pipeline

- **RQ1 (main):** How faithful are the AI-generated notes, and can we
  *automatically flag* the unreliable ones? Does an LLM-as-a-judge faithfulness
  score beat traditional metrics (ROUGE, BERTScore, Levenshtein) at flagging?
- **RQ2:** How much do speech-to-text (Whisper) errors degrade the notes?

```
audio ──Whisper──▶ transcript ──LLM+prompt──▶ SOAP note ──evaluation──▶ RELIABILITY FLAG 🚨
                                                   │
                        validated against PriMock57 human ratings (RQ1)
```

---

## Manual setup (if you'd rather not use the setup file)

Works the same on Mac, Linux, or Windows. Run these from this folder.

**1. Create an isolated Python environment** (so nothing clashes with the rest of
the machine):

```bash
# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
```

**2. Install the code and its libraries:**

```bash
pip install -r requirements.txt
pip install -e .
```

**3. Install the system tools** (once):

- **ffmpeg** — Mac: `brew install ffmpeg` · Windows: download the "essentials"
  build from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and put
  `ffmpeg.exe` on your PATH (or in this folder).
- **Ollama** — Mac: `brew install ollama` · Windows: the portable zip from
  [ollama.com/download](https://ollama.com/download). Then start it and pull the
  models:

```bash
ollama serve            # leave this running in its own window
ollama pull medgemma:4b
ollama pull llama3.1:8b
ollama pull atla/selene-mini
```

**4. (Optional) the third-party audio** — PriMock57's audio is licensed and not
included. Fetch it only if you want to reproduce the research:

```bash
cd primock57 && python3 download_audio.py && cd ..
```

---

## Run the demo (Clarion)

A **FastAPI + Next.js** app in `app/`: pick a consultation (or upload your own
audio) and watch the transcript stream in, the SOAP note draft itself, and a
hairline sweep down the note flagging every claim the recording does not support.

```bash
# one time: cache the sample consultations (runs the real pipeline; slow)
python scripts/build_samples.py        # or: make samples

# start the app (FastAPI on :8000 + Next.js on :3000)
python scripts/dev.py                   # or: make demo
```

Then open **http://localhost:3000**. Needs Ollama running with `medgemma:4b` and
`llama3.1:8b`, plus Node 20+.

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
result. The API also warms Whisper and both Ollama models at startup (~10–17s),
so the first upload is not the slow one.

> **Data safety:** `app/web/public/samples/` holds transcript and note text and is
> gitignored. Regenerate it locally; never commit it.

Every real (uploaded) run is written to a local **audit trail** — open **`/audit`**
in the app to browse past runs and open any one to see its full provenance (models,
prompt versions, verdict bands, flagged claims). The same data is served at
`GET /api/runs`, `/api/runs/{id}`, `/api/stats`, and `/api/metrics`. For running the
product as a service (a preflight launcher and a Docker image that talks to Ollama on
the host), see **[DEPLOY.md](DEPLOY.md)** — `make run`, `make check`, `make docker`.

---

## Reproduce the research pipeline (00 → 11)

Every stage is a standalone script in `pipelines/`, run in order with your
environment active and Ollama up (`medgemma:4b`, `llama3.1:8b`,
`atla/selene-mini`). All settings come from `config/config.yaml`.

```bash
python pipelines/00_dataset_report.py        # dataset inventory + integrity
python pipelines/01_transcribe.py            # Whisper over the audio          (RQ2)
python pipelines/02_generate_notes.py        # transcripts → SOAP notes
python pipelines/03_evaluate.py              # metrics + judge + reliability         (RQ1)
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

**Note:** the LoRA fine-tuning pipelines (`10`, `12_finetune_train`) use Apple
**MLX** and run on Apple-Silicon Macs only; that experiment is complete and its
results are in `results/finetune/`. Everything else — transcription, generation,
the reliability check, and the demo — runs on any OS.

---

## Using the pipeline directly

`s2n.service.run_pipeline()` is UI-agnostic — audio (or a transcript) in, a plain
JSON-serialisable dict out:

```python
from s2n.service import run_pipeline
result = run_pipeline(audio_path="results/mixed_audio_dev/day1_consultation02.wav")
print(result["reliability"]["verdict"], "—", result["reliability"]["score_note"])
```

`run_pipeline_staged()` yields the same work one stage at a time, which is what
the SSE endpoint streams.

---

## Layout

```
SETUP_WINDOWS.bat   one-double-click setup for Windows
setup_mac.command   one-double-click setup for Mac
config/             one control panel (config.yaml) — every setting lives here
src/s2n/            the code library (the engine)
  data/               load PriMock57
  transcription/      Whisper wrapper                 (RQ2)
  generation/         transcript + prompt → note
  evaluation/         metrics + LLM-judge + highlights + correlation  ← the heart
  reliability/        turns the claim counts into a verdict (bands)   ← the star
  llm/                backend-agnostic client (swap freely, stay free)
prompts/            versioned prompts (v1.0, v1.1 …)
pipelines/          runnable scripts (00 → 12)
experiments/        notebooks + experiment configs
results/            generated outputs (gitignored)
app/                demo front-end (the stage)
tests/              unit tests
docs/               architecture + decision records (ADRs)
primock57/          dataset (fetched via download script; licensed)
```

## Principles

1. **Separation of concerns** — library vs pipelines vs app vs deliverables.
2. **Reproducibility** — config-driven, versioned prompts, logged results, a
   one-command setup that works on a fresh machine.
3. **Evaluation-first** — the evaluation harness is central, not bolted on.
