# Architecture

Pipeline: **audio → Whisper → transcript → LLM → SOAP note → evaluation → RELIABILITY FLAG**

- **RQ1 (star):** how faithful are the notes, and can we auto-flag bad ones?
- **RQ2:** how much do ASR (Whisper) errors degrade the notes?

## Modules (src/s2n)
| Module | Job |
|---|---|
| data | load PriMock57 (transcripts, notes+highlights, human eval) |
| transcription | Whisper (RQ2) |
| generation | transcript + versioned prompt → note |
| evaluation | metrics + LLM-judge + highlights + correlation |
| reliability | reliability flag (the contribution) |
| llm | backend-agnostic model access |

## Validation logic (why anyone should trust the reliability)
Auto faithfulness score → correlate with PriMock57 human ratings → report
flagging precision/recall. If the score tracks humans, the reliability is trustworthy.
