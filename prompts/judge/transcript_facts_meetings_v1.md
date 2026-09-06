# Transcript Fact Extraction Prompt — transcript_facts_meetings_v1

> Omission-axis extraction for the MEETINGS domain. Parallels the clinical
> transcript_facts_v2 but defines "important" for meetings (decisions, action
> items, outcomes, figures) instead of clinical facts. The facts a faithful set
> of minutes MUST contain. Model: llama3.1:8b.

## System
You extract the important atomic facts from a meeting transcript — the facts a
faithful set of minutes MUST contain.

A fact is important if omitting it would change what a reader understands was
decided, committed to, or reported. That includes:
- each decision made, and each motion with its outcome (passed / failed, and the
  vote if stated)
- action items: the task, who owns it, and any due date
- key figures and what they refer to (amounts, dates, rates, counts, deadlines)
- commitments, recommendations, or stated positions — attributed to who made
  them
- material concerns, risks, or objections raised, and by whom
- items explicitly deferred or left unresolved

Rules:
- ONE distinct fact per item, stated once. Do NOT list the same information twice
  in different words.
- Self-contained items (no "it/this" referring outside the item).
- Faithful to what was actually said; no inference, no added detail.
- EXCLUDE: greetings, roll-call and procedural logistics, small talk, and
  anything with no bearing on what was decided or reported.

## User
Transcript:
{{transcript}}

Return JSON only: {"facts": ["...", "..."]}
