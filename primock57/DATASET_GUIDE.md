# PriMock57 — Dataset Guide (plain English)

*Your cheat-sheet for understanding and defending the dataset. Written for the viva.*

## What it is, in one line
57 **mock** GP consultations (fake patients, real clinicians) recorded on purpose so
researchers can work on medical speech + note generation **without breaking patient
confidentiality**. That confidentiality problem is *why* public medical datasets are
rare — a great line for your viva.

- Made by **Babylon Health**; published at **ACL 2022** (two papers).
- 7 clinicians + 57 employees acting as patients, over 5 days, using scripted "case cards".

## What's in the folder

| Folder / file | What it holds | Why you care |
|---|---|---|
| `audio/` | 114 `.wav` files = 57 consults × 2 speakers (doctor + patient recorded separately) | Input to Whisper (RQ2). **Currently pointers — run `download_audio.py`.** |
| `transcripts/` | 114 `.TextGrid` files = the **human** (gold) transcript, utterance-by-utterance with timings | Your "correct answer" for transcription. Compare Whisper against these for RQ2. |
| `notes/` | 57 `.json` files: `note` (the SOAP-style clinician note) + `highlights` (key facts the clinician flagged) | The **reference note** your LLM tries to match. `highlights` = ready-made key-fact checklist for spotting omissions. |
| `human_eval_data/` | The gold mine (see below) | Your **answer key** for RQ1 — proves your auto-metric agrees with humans. |
| `consultation_checklists/` | Case cards / checklists per clinician | Context on what *should* be covered. |
| `scripts/` | Babylon's helper scripts (TextGrid → transcript, audio mixing) | `textgrid_to_transcript.py` turns the messy TextGrid into plain text for you. |

## The `human_eval_data/` folder — read this twice
This is what makes PriMock57 special and why your proposal picked it.

- **`results.csv`** — humans (5 evaluators) reviewed notes from **10 different models** + the
  real doctor's note. For each, they marked **Incorrect Statements** (= hallucinations) and
  **Omissions**, and even split them into *critical* (`!`) vs *non-critical* (`-`).
  → This is direct human-labelled ground truth for "hallucination vs omission".
- **`metric-scores.json`** — pre-computed scores for **18 automatic metrics** (incl. ROUGE,
  BERTScore, etc.) on those same notes.
  → You can validate your LLM-judge against humans **and** benchmark it against 18 existing
  metrics without recomputing them. This is the backbone of RQ1.
- **`other-issues-taxonomy-examples.csv`** — categorised qualitative feedback.

**Viva soundbite:** *"PriMock57 doesn't just give me data — it gives me human judgements of
note quality, so I can prove my automatic flag actually tracks what a clinician would flag."*

## File formats you'll meet
- **`.TextGrid`** = a Praat annotation format. Plain text; each utterance is an "interval"
  with `xmin` (start sec), `xmax` (end sec), `text`. Tags: `<UNSURE>` = transcriber unsure,
  `<UNIN/>` = unintelligible. Use the provided `scripts/textgrid_to_transcript.py`.
- **`.json` note** = fields `day`, `consultation`, `presenting_complaint`, `note`, `highlights`.

## How to get the audio (~2 GB) — do this on your Mac
The audio is stored with **Git LFS** (big files kept outside the normal repo as pointers).
Two options — pick one:

**Option A — the script (no extra installs):**
```bash
cd "primock57"
python3 download_audio.py
```

**Option B — the standard Git LFS way (what the README teaches):**
```bash
brew install git-lfs      # one time
git lfs install           # one time
cd "primock57"
git lfs pull              # note: the .git here was set up in the cloud;
                          # if this errors, just use Option A, or re-clone fresh.
```

## What is / isn't done
- [x] Repo cloned; all text data present (notes, transcripts, human_eval, scripts, checklists)
- [x] `download_audio.py` ready
- [ ] **Audio not downloaded yet** — run the script on your Mac
- [ ] (Later, not today) Whisper transcription, note generation, the judge

## Citation (for your report)
Papadopoulos Korfiatis et al., *PriMock57: A Dataset of Primary Care Mock Consultations*, ACL 2022.
Moramarco et al., *Human Evaluation and Correlation with Automatic Metrics in Consultation Note Generation*, ACL 2022.
