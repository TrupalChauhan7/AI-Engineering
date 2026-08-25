"""PriMock57 data access — SPLIT INTO TWO leak-safe modules.

This module used to be a single loader. It is intentionally split so the
generator can never reach the gold answers (project rule 1):

  * ``s2n.data.generation_data.GenerationLoader``  -> transcripts ONLY
  * ``s2n.data.evaluation_data.EvaluationLoader``  -> notes/ + human_eval/ (answers)
  * ``s2n.data.splits.load_or_create_split``       -> frozen dev/test split
  * ``s2n.data.textgrid``                          -> TextGrid -> dialogue

Import from those modules directly. Nothing lives here so that no single object
holds both transcripts and answers.
"""

from __future__ import annotations
