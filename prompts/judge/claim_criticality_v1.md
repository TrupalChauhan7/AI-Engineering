# Claim Criticality Prompt — claim_criticality_v1

> EXPLORATORY claim verifier, SECONDARY column. Rates only the claims already
> judged unsupported: would this error matter clinically? Mirrors PriMock57's
> human "!" (critical) vs "-" (non-critical) scheme. Separate pass so the
> locked verify path is untouched. Model: llama3.1:8b.

## System
Each numbered claim below appears in a clinical note but is NOT supported by
the consultation transcript — it is an error. For EACH claim, rate the error:
- "critical": could affect patient safety, diagnosis, or management if acted
  on (wrong medication/dose, wrong symptom or side, invented finding or
  history, wrong follow-up interval, fabricated diagnosis).
- "minor": unlikely to change care (stylistic detail, minor demographic or
  phrasing detail, redundant filler).

Labels must be in the SAME ORDER as the claims, one label per claim.

## User
TRANSCRIPT (what was actually said):
{{transcript}}

UNSUPPORTED CLAIMS FROM THE NOTE:
{{claims}}

Return JSON only: {"labels": ["critical", "minor", ...]} — exactly one label
per claim, in order.
