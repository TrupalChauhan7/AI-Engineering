"""Audit trail — persist every completed pipeline run, with full provenance.

WHY this module exists: ``s2n.service`` scores a note and returns a dict; the
demo streams that dict to the screen and then discards it. A *product* — above
all one used in a regulated setting like a clinic or a law firm — must instead
keep a durable record of what it decided, on what input, with which models and
prompts, and when. That record is the AUDIT TRAIL. It answers, months later,
"why was this note flagged?" and "could we reproduce that verdict today?".

WHAT IS STORED: the transcript and the note (the audit target), every flagged
claim and omitted fact (the *why*, not just a number), the verdict and counts,
the timings, and — the part that makes a run REPRODUCIBLE — its provenance:
the domain, the generator and verifier models, all four prompt versions, and
the reliability bands in force at the time. The audio itself is NOT stored:
that matches the pipeline's privacy posture (audio never persists) and keeps
the database small.

STORAGE: SQLite via the standard library — one file, queryable with SQL, no
server to run, and portable. Each operation opens its own short-lived
connection, so the store is safe to call from the API's worker thread and the
request thread alike without shared-connection surprises.

LEAK SAFETY (project rule 1): this module imports only the standard library
plus ``s2n.config.ROOT`` (for the default DB location). It has NO route to the
gold notes or human-eval answer key, so persisting a run cannot smuggle the
answer key into the served path.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from s2n.config import ROOT

DEFAULT_DB_PATH = ROOT / "results" / "runs.db"

# INTEGER columns; every other column is TEXT (JSON-encoded where it holds a
# list or dict — see _JSON_COLUMNS).
_INT_COLUMNS = frozenset(
    {"reliable_max", "unreliable_min", "n_unsupported", "n_omitted", "combined"}
)
_JSON_COLUMNS = frozenset({"unsupported_claims", "omitted_facts", "timings"})

# Column order is the table's contract; _row_to_dict and the INSERT both rely on it.
_COLUMNS = (
    "run_id",
    "request_id",
    "created_at",
    "domain",
    "source",
    "source_name",
    "generator_model",
    "verifier_model",
    "generation_prompt",
    "verify_prompt",
    "omission_verify_prompt",
    "transcript_facts_prompt",
    "reliable_max",
    "unreliable_min",
    "transcript",
    "note",
    "n_unsupported",
    "n_omitted",
    "combined",
    "verdict",
    "score_note",
    "unsupported_claims",
    "omitted_facts",
    "timings",
)

# Light summary columns for list views — never the transcript or note, so a
# history page stays cheap even with thousands of rows.
_SUMMARY_COLUMNS = (
    "run_id",
    "request_id",
    "created_at",
    "domain",
    "source",
    "source_name",
    "generator_model",
    "n_unsupported",
    "n_omitted",
    "combined",
    "verdict",
)

# The per-stage timing keys the pipeline reports (s2n.service). latency_stats
# reads these out of the stored ``timings`` JSON.
_TIMING_KEYS = ("transcribe_s", "generate_s", "reliability_s")


def _column_ddl(name: str) -> str:
    return f"{name} INTEGER" if name in _INT_COLUMNS else f"{name} TEXT"


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS runs (\n  "
    + ",\n  ".join(_column_ddl(c) + (" PRIMARY KEY" if c == "run_id" else "") for c in _COLUMNS)
    + "\n);\n"
    "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at);\n"
    "CREATE INDEX IF NOT EXISTS idx_runs_domain ON runs (domain);\n"
)


class RunStore:
    """A durable, queryable record of completed pipeline runs (SQLite-backed)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- write ----------------------------------------------------------------

    def record(
        self,
        result: dict,
        *,
        cfg: dict,
        source: str,
        source_name: str | None = None,
        run_id: str | None = None,
        request_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        """Persist one completed run and return its ``run_id``.

        ``result`` is exactly what ``s2n.service.run_pipeline`` returns; ``cfg``
        is the resolved config it ran under (the source of provenance). ``source``
        is ``"audio"`` or ``"transcript"``; ``source_name`` an optional label
        (e.g. the uploaded filename). ``request_id`` correlates the row with the
        request's log lines (observability). A ``run_id`` / ``created_at`` may be
        passed for determinism in tests; otherwise a uuid4 and the current UTC
        time are used.
        """
        run_id = run_id or uuid.uuid4().hex
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        rel = result.get("reliability", {})
        cv = cfg.get("claim_verifier", {})
        bands = cfg.get("reliability", {})

        values = {
            "run_id": run_id,
            "request_id": request_id,
            "created_at": created_at,
            "domain": cfg.get("active_domain") or cfg.get("domain"),
            "source": source,
            "source_name": source_name,
            "generator_model": cfg.get("llm", {}).get("model"),
            "verifier_model": cv.get("model"),
            "generation_prompt": cfg.get("generation", {}).get("prompt_version"),
            "verify_prompt": cv.get("verify_prompt"),
            "omission_verify_prompt": cv.get("omission_verify_prompt"),
            "transcript_facts_prompt": cv.get("transcript_facts_prompt"),
            "reliable_max": bands.get("reliable_max"),
            "unreliable_min": bands.get("unreliable_min"),
            "transcript": result.get("transcript"),
            "note": result.get("note"),
            "n_unsupported": rel.get("n_unsupported"),
            "n_omitted": rel.get("n_omitted"),
            "combined": rel.get("combined"),
            "verdict": rel.get("verdict"),
            "score_note": rel.get("score_note"),
            "unsupported_claims": json.dumps(rel.get("unsupported_claims", [])),
            "omitted_facts": json.dumps(rel.get("omitted_facts", [])),
            "timings": json.dumps(result.get("timings", {})),
        }
        placeholders = ", ".join(f":{c}" for c in _COLUMNS)
        sql = f"INSERT INTO runs ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, values)
        return run_id

    # -- read -----------------------------------------------------------------

    def get(self, run_id: str) -> dict | None:
        """The full stored record for one run, or ``None`` if unknown."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list(self, limit: int = 50, domain: str | None = None) -> list[dict]:
        """Recent runs (summary fields only), newest first.

        Transcript and note are omitted deliberately — a history list must stay
        light. Fetch one run's full record with :meth:`get`.
        """
        cols = ", ".join(_SUMMARY_COLUMNS)
        sql = f"SELECT {cols} FROM runs"
        params: list = []
        if domain:
            sql += " WHERE domain = ?"
            params.append(domain)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Total number of stored runs."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def stats(self) -> dict:
        """Aggregate counts by verdict and by domain — a dashboard's raw material."""
        with self._connect() as conn:
            by_verdict = {
                r["verdict"]: r["n"]
                for r in conn.execute(
                    "SELECT verdict, COUNT(*) AS n FROM runs GROUP BY verdict"
                ).fetchall()
            }
            by_domain = {
                r["domain"]: r["n"]
                for r in conn.execute(
                    "SELECT domain, COUNT(*) AS n FROM runs GROUP BY domain"
                ).fetchall()
            }
        return {"total": self.count(), "by_verdict": by_verdict, "by_domain": by_domain}

    def latency_stats(self) -> dict:
        """Per-stage latency (mean / p50 / p90) over every recorded run.

        The operational companion to :meth:`stats`: turns the timings already
        stored with each run into a picture of how the pipeline performs. Stages
        are ``transcribe`` / ``generate`` / ``reliability`` (seconds).
        """
        samples: dict[str, list[float]] = {k: [] for k in _TIMING_KEYS}
        with self._connect() as conn:
            for (raw,) in conn.execute("SELECT timings FROM runs"):
                try:
                    timings = json.loads(raw) if raw else {}
                except (TypeError, json.JSONDecodeError):
                    continue
                for key in _TIMING_KEYS:
                    val = timings.get(key)
                    if isinstance(val, (int, float)):
                        samples[key].append(float(val))
        return {key: _summary(vals) for key, vals in samples.items()}


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in [0, 1]) of a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def _summary(values: list[float]) -> dict:
    """{n, mean, p50, p90} for a stage's latency samples (Nones when empty)."""
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(_percentile(values, 0.5), 3),
        "p90": round(_percentile(values, 0.9), 3),
    }


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Turn a stored row back into the shape callers expect (JSON columns decoded)."""
    out = dict(row)
    for col in _JSON_COLUMNS:
        if col in out and isinstance(out[col], str):
            out[col] = json.loads(out[col])
    return out
