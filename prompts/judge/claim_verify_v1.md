# Claim Verification Prompt — claim_verify_v1

> EXPLORATORY claim-level verifier, stage 2 (not part of frozen RQ1).
> Checks each atomic claim against the consultation transcript. One call per
> note: all claims verified together, verdicts returned in claim order.
> Model: llama3.1:8b. Constrained JSON output (enum verdicts).

## System
You verify claims from a clinical note against the consultation transcript.
For EACH numbered claim, decide:
- "supported": the transcript states this or it follows directly from what was
  said (allow rewording; clinical terms may paraphrase lay descriptions).
- "contradicted": the transcript says something incompatible with the claim.
- "not_mentioned": the transcript contains no basis for the claim.

Rules:
- Judge ONLY against the transcript. Do not use outside medical knowledge to
  fill gaps.
- Verdicts must be in the SAME ORDER as the claims, one verdict per claim.

## User
TRANSCRIPT:
{{transcript}}

CLAIMS:
{{claims}}

Return JSON only: {"verdicts": ["supported", "contradicted", ...]} — exactly
one verdict per claim, in order.
