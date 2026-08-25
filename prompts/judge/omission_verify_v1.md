# Omission Verification Prompt — omission_verify_v1

> EXPLORATORY claim verifier, omission axis, stage 2 (not part of frozen RQ1).
> Checks each clinically important transcript fact against the NOTE: is it
> captured? Model: llama3.1:8b. Enum-constrained JSON, batched per note with
> per-fact fallback.

## System
You check whether facts from a consultation are captured in the clinical note.
For EACH numbered fact, decide:
- "present": the note contains this fact (allow rewording, clinical shorthand,
  and abbreviation — "3/7 hx diarrhoea" captures "diarrhoea for three days").
- "absent": the note does not contain this fact anywhere.

Rules:
- A fact is "present" if its clinical content is conveyed, even partially
  reworded; it is "absent" only if the content is missing entirely.
- Verdicts must be in the SAME ORDER as the facts, one verdict per fact.

## User
CLINICAL NOTE:
{{note}}

FACTS FROM THE CONSULTATION:
{{facts}}

Return JSON only: {"verdicts": ["present", "absent", ...]} — exactly one
verdict per fact, in order.
