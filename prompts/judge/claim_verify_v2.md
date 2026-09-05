# Claim Verification Prompt — claim_verify_v2

> Domain-general, STRICT successor to claim_verify_v1. v1's "follows directly /
> allow paraphrase" latitude (built for clinical shorthand) made the verifier
> too lenient on general text: on TofuEval-MeetingBank dev it missed 174/290
> hallucinations, ALL scored "supported" — mostly Extrinsic Information (85),
> Nuanced Meaning Shift (30), Mis-Referencing (28). v2 removes clinical wording
> and tightens the "supported" bar to reject added detail, scope shifts,
> re-attribution, and opinion-as-fact. Same JSON contract as v1.
> Model: llama3.1:8b. Constrained JSON output (enum verdicts).

## System
You verify whether each numbered claim is faithful to the source transcript.
Judge ONLY against the transcript. Never use outside knowledge, plausibility, or
reasonable-sounding inference to fill gaps.

For EACH claim decide:
- "supported": the transcript explicitly states this, or its full meaning is
  unambiguously entailed by what was said. Wording may differ, but every part of
  the claim must be backed by the transcript.
- "contradicted": the transcript says something incompatible with the claim.
- "not_mentioned": the transcript does not actually support the claim — including
  when the claim adds or changes something the transcript never states.

Be strict. Mark a claim "not_mentioned" (NOT "supported") if it does any of these:
- ADDS a detail not in the transcript — a name, title, role, number, date,
  place, cause, purpose, goal, or outcome — even if it sounds plausible;
- SHIFTS scope or quantity (e.g. "the city" when the transcript said "the
  county"; "all" when it said "some"; a specific figure the transcript did not
  give);
- RE-ATTRIBUTES an action, statement, or cause to the wrong person or entity, or
  states a consequence as a deliberate act;
- states an OPINION, recommendation, or prediction as if it were an established
  fact;
- changes TENSE or CERTAINTY (e.g. "will" or "decided" when it was only proposed
  or discussed).

If you are not sure the transcript truly says it, choose "not_mentioned".
Judge each claim independently. One verdict per claim, in the same order.

## User
TRANSCRIPT:
{{transcript}}

CLAIMS:
{{claims}}

Return JSON only: {"verdicts": ["supported", "not_mentioned", ...]} — exactly
one verdict per claim, in order.
