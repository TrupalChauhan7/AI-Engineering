/** Wire types — mirror the FastAPI contract in app/api/main.py exactly. */

export type Verdict = "reliable" | "review" | "unreliable";

export interface Alarm {
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
  alarm_s: number;
}

export interface AnalysisResult {
  id?: string;
  label?: string;
  durationLabel?: string;
  transcript: string;
  note: string;
  alarm: Alarm;
  timings: Timings;
  spans?: Span[];
}

export interface SampleMeta {
  id: string;
  label: string;
  durationLabel: string;
}

/** Which stage the UI is currently revealing. */
export type Stage = "idle" | "transcribe" | "generate" | "alarm" | "done" | "error";

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
