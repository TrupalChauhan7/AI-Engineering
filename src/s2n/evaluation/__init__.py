"""The evaluation harness — the heart of the project (evaluation-first design).

metrics.py     traditional baselines (ROUGE, BERTScore, Levenshtein)
judge.py       LLM-as-a-judge faithfulness scoring
highlights.py  key-fact coverage using the clinician 'highlights'
correlation.py validate judge vs PriMock57 human ratings
"""
