/** Wire types — mirror the FastAPI contract in app/api/main.py exactly. */

export type Verdict = "reliable" | "review" | "unreliable";

export interface Reliability {
  unsupported_claims: string[];
  omitted_facts: string[];
  n_unsupported: number;
  n_omitted: number;
  combined: number;
  verdict: Verdict;
  score_note: string;
}

/** Best-effort character span of a flagged claim inside the note. -1 = no match. */
export interface Span {
  claim: string;
  start: number;
  end: number;
  line: number;
}

export interface Timings {
  transcribe_s: number;
  generate_s: number;
  reliability_s: number;
}

export interface AnalysisResult {
  id?: string;
  label?: string;
  durationLabel?: string;
  transcript: string;
  note: string;
  reliability: Reliability;
  timings: Timings;
  spans?: Span[];
}

export interface SampleMeta {
  id: string;
  label: string;
  durationLabel: string;
}

/** A light audit-trail row (GET /api/runs). */
export interface RunSummary {
  run_id: string;
  request_id: string | null;
  created_at: string;
  source: string;
  source_name: string | null;
  generator_model: string;
  n_unsupported: number;
  n_omitted: number;
  combined: number;
  verdict: Verdict;
}

/** A full audit record (GET /api/runs/{id}) — payload + provenance. */
export interface RunDetail extends RunSummary {
  verifier_model: string;
  generation_prompt: string;
  verify_prompt: string;
  omission_verify_prompt: string;
  transcript_facts_prompt: string;
  reliable_max: number;
  unreliable_min: number;
  transcript: string;
  note: string;
  score_note: string;
  unsupported_claims: string[];
  omitted_facts: string[];
  timings: Timings;
}

/** Per-stage latency (GET /api/metrics). */
export interface StageLatency {
  n: number;
  mean: number | null;
  p50: number | null;
  p90: number | null;
}

/** GET /api/metrics — operational counts + latency, from the audit trail. */
export interface Metrics {
  total: number;
  by_verdict: Record<string, number>;
  latency_s: Record<string, StageLatency>;
}

/** Which stage the UI is currently revealing. */
export type Stage = "idle" | "transcribe" | "generate" | "reliability" | "done" | "error";

export const VERDICT_COLOR: Record<Verdict, string> = {
  reliable: "var(--color-sig-green)",
  review: "var(--color-sig-amber)",
  unreliable: "var(--color-sig-red)",
};

export const VERDICT_WORD: Record<Verdict, string> = {
  reliable: "Reliable",
  review: "Review",
  unreliable: "Unreliable",
};
