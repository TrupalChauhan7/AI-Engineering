# Omission Verification Prompt — omission_verify_v2

> Domain-general, STRICTER successor to omission_verify_v1. v1 (clinical, lenient:
> "present if conveyed even partially reworded; absent only if missing entirely")
> biased toward "present" — on the TofuEval-MeetingBank key-points probe it caught
> only 57% of removed facts while wrongly flagging just 2.6% of present ones (huge
> headroom). v2 removes clinical wording and tightens the "present" bar: a fact is
> present only if its SPECIFIC content is actually stated, not merely the topic.
> Same JSON contract as v1. Model: llama3.1:8b. Enum-constrained JSON.

## System
You check whether each numbered fact is actually stated in the summary.

For EACH fact decide:
- "present": the summary conveys THIS SPECIFIC fact — the same detail, not just
  the same general topic. Rewording is fine, but the actual content of the fact
  must appear in the summary.
- "absent": the summary does not state this fact — including when it only touches
  the general topic without the specific detail, or states only part of it.

Rules:
- Judge each fact on its OWN specific content. Related information about the same
  subject does NOT make a distinct fact "present."
- If a fact has several parts (e.g. a date AND a place AND a purpose), it is
  "present" only if ALL its parts appear in the summary; if any part is missing,
  mark it "absent".
- Do not use outside knowledge or assume the summary implies something it does
  not actually say.
- When unsure whether the specific fact is really stated, choose "absent".
- Verdicts must be in the SAME ORDER as the facts, one verdict per fact.

## User
SUMMARY:
{{note}}

FACTS THAT A COMPLETE SUMMARY SHOULD CONTAIN:
{{facts}}

Return JSON only: {"verdicts": ["present", "absent", ...]} — exactly one verdict
per fact, in order.
