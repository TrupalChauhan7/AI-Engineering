"use client";

import { useCallback, useRef, useState } from "react";
import type { Alarm, AnalysisResult, SampleMeta, Span, Stage, Timings } from "./types";

/**
 * ONE analysis controller, two event sources.
 *
 * Samples replay a cached result on a scripted cadence so the demo is instant
 * and identical every take. Uploads run the real local pipeline and are driven
 * by the SSE stream. Both push through the same state machine, so the animation
 * layer downstream never knows which it is watching.
 */

export interface AnalysisState {
  stage: Stage;
  transcript: string;
  note: string;
  alarm: Alarm | null;
  spans: Span[];
  timings: Timings | null;
  error: string | null;
  /** true while the real pipeline is running (upload only) */
  live: boolean;
  sampleId: string | null;
  /** epoch ms the current stage began — drives the live elapsed counter */
  stageStartedAt: number | null;
  /** true once the server has confirmed the stream is alive */
  streaming: boolean;
}

const EMPTY: AnalysisState = {
  stage: "idle",
  transcript: "",
  note: "",
  alarm: null,
  spans: [],
  timings: null,
  error: null,
  live: false,
  sampleId: null,
  stageStartedAt: null,
  streaming: false,
};

/** Scripted replay cadence (ms) — long enough to read, short enough to hold. */
const BEAT = { transcribe: 1100, generate: 1000, alarm: 1500 };

const prefersReduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>(EMPTY);
  const abortRef = useRef<AbortController | null>(null);
  const runIdRef = useRef(0);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    runIdRef.current += 1;
    setState(EMPTY);
  }, []);

  /** Replay a cached sample with scripted timing (no models run). */
  const runSample = useCallback(async (meta: SampleMeta) => {
    abortRef.current?.abort();
    const run = ++runIdRef.current;
    const alive = () => runIdRef.current === run;

    setState({ ...EMPTY, stage: "transcribe", sampleId: meta.id, stageStartedAt: Date.now() });

    let data: AnalysisResult;
    try {
      const res = await fetch(`/api/samples/${meta.id}`);
      if (!res.ok) throw new Error(`Sample unavailable (${res.status})`);
      data = await res.json();
    } catch (e) {
      if (alive()) {
        setState({ ...EMPTY, stage: "error", error: (e as Error).message });
      }
      return;
    }
    if (!alive()) return;

    const instant = prefersReduced();
    const beat = (ms: number) => (instant ? 0 : ms);

    setState((s) => ({ ...s, transcript: data.transcript }));
    await wait(beat(BEAT.transcribe));
    if (!alive()) return;

    setState((s) => ({ ...s, stage: "generate", note: data.note, stageStartedAt: Date.now() }));
    await wait(beat(BEAT.generate));
    if (!alive()) return;

    setState((s) => ({
      ...s,
      stage: "alarm",
      alarm: data.alarm,
      spans: data.spans ?? [],
      stageStartedAt: Date.now(),
    }));
    await wait(beat(BEAT.alarm));
    if (!alive()) return;

    setState((s) => ({ ...s, stage: "done", timings: data.timings, stageStartedAt: null }));
  }, []);

  /** Run the real pipeline on an uploaded file, driven by the SSE stream. */
  const runUpload = useCallback(async (file: File) => {
    abortRef.current?.abort();
    const run = ++runIdRef.current;
    const alive = () => runIdRef.current === run;
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState({ ...EMPTY, stage: "transcribe", live: true, stageStartedAt: Date.now() });

    const body = new FormData();
    body.append("audio", file);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body, signal: ctrl.signal });
      if (!res.ok || !res.body) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line
        let split: number;
        while ((split = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          if (!alive()) return;

          let event = "message";
          const dataLines: string[] = [];
          for (const line of frame.split("\n")) {
            if (line.startsWith("event: ")) event = line.slice(7).trim();
            else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
          }
          if (!dataLines.length) continue;

          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(dataLines.join("\n"));
          } catch {
            continue; // a malformed frame must not kill the stream
          }
          applyEvent(event, payload, setState);
        }
      }
      if (alive()) setState((s) => ({ ...s, live: false, stageStartedAt: null }));
    } catch (e) {
      if ((e as Error).name === "AbortError" || !alive()) return;
      setState((s) => ({
        ...s,
        stage: "error",
        live: false,
        stageStartedAt: null,
        error: (e as Error).message,
      }));
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    runIdRef.current += 1; // orphan any in-flight handler
    setState((s) => ({
      ...s,
      stage: "idle",
      live: false,
      streaming: false,
      stageStartedAt: null,
      error: null,
    }));
  }, []);

  return { state, runSample, runUpload, reset, cancel };
}

function applyEvent(
  event: string,
  p: Record<string, unknown>,
  setState: React.Dispatch<React.SetStateAction<AnalysisState>>,
) {
  if (event === "heartbeat") {
    // proof the stream is open while a long stage runs
    setState((s) => (s.streaming ? s : { ...s, streaming: true }));
    return;
  }
  if (event === "error") {
    setState((s) => ({
      ...s,
      stage: "error",
      live: false,
      stageStartedAt: null,
      error: String(p.message ?? "failed"),
    }));
    return;
  }
  if (event === "done") {
    setState((s) => ({
      ...s,
      stage: "done",
      live: false,
      stageStartedAt: null,
      timings: p.timings as Timings,
    }));
    return;
  }
  // event === "stage"
  const stage = p.stage as Stage;
  if (p.status === "running") {
    // new stage -> restart the elapsed clock
    setState((s) => ({ ...s, stage, streaming: true, stageStartedAt: Date.now() }));
    return;
  }
  setState((s) => ({
    ...s,
    stage,
    transcript: (p.transcript as string) ?? s.transcript,
    note: (p.note as string) ?? s.note,
    alarm: (p.alarm as Alarm) ?? s.alarm,
    spans: (p.spans as Span[]) ?? s.spans,
  }));
}
