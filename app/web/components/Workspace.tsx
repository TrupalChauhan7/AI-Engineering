"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { EASE, fadeUp, sectionReveal } from "@/lib/motion";
import { useAnalysis } from "@/lib/useAnalysis";
import type { SampleMeta } from "@/lib/types";
import AlarmPanel from "./analysis/AlarmPanel";
import NotePanel from "./analysis/NotePanel";
import TranscriptPanel from "./analysis/TranscriptPanel";
import { Elapsed, LivePulse } from "./StageProgress";
import Waveform from "./Waveform";

const STAGE_LABEL: Record<string, string> = {
  idle: "Ready",
  transcribe: "Transcribing…",
  generate: "Drafting the note…",
  alarm: "Verifying every claim…",
  done: "Complete",
  error: "Failed",
};

export default function Workspace() {
  const { state, runSample, runUpload, cancel } = useAnalysis();
  const [samples, setSamples] = useState<SampleMeta[]>([]);
  const [samplesLoaded, setSamplesLoaded] = useState(false);
  const [linkedFlag, setLinkedFlag] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [unreliableMin, setUnreliableMin] = useState(12);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // the ring scales against the real config band, not a magic number
    fetch("/api/health")
      .then((r) => (r.ok ? r.json() : null))
      .then((h) => h?.alarm?.unreliable_min && setUnreliableMin(h.alarm.unreliable_min))
      .catch(() => {});
    fetch("/api/samples")
      .then((r) => (r.ok ? r.json() : []))
      .then(setSamples)
      .catch(() => setSamples([]))
      .finally(() => setSamplesLoaded(true));
  }, []);

  const copyNote = useCallback(async () => {
    if (!state.note) return;
    await navigator.clipboard.writeText(state.note);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, [state.note]);

  const downloadNote = useCallback(() => {
    if (!state.note) return;
    const blob = new Blob([state.note], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.sampleId ?? "consultation"}-soap-note.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [state.note, state.sampleId]);

  const busy = state.stage !== "idle" && state.stage !== "done" && state.stage !== "error";
  // Reserve the tall fixed columns only once there is content; otherwise the
  // empty state opens a 544px black void above "How it works".
  const hasContent = Boolean(state.transcript || state.note || state.alarm);
  const colClass = hasContent ? "lg:h-[34rem] lg:overflow-hidden" : "";

  return (
    <section id="workspace" className="px-6 py-24 md:px-[60px] md:py-32">
      <motion.div {...sectionReveal}>
        <p className="t-label mb-6 text-[var(--color-ink-mute)]">002 — The workspace</p>
        <h2 className="t-section max-w-[20ch] text-[var(--color-ink)]">
          Watch it check its <em className="font-normal italic">own</em> work.
        </h2>
      </motion.div>

      {/* controls */}
      <motion.div
        {...sectionReveal}
        className="mt-12 flex flex-wrap items-center gap-x-8 gap-y-5 border-y border-[var(--color-hairline)] py-6"
      >
        <span className="t-label text-[var(--color-ink-mute)]">Consultation</span>

        {!samplesLoaded && (
          <span className="t-label text-[var(--color-ink-faint)]">loading…</span>
        )}

        {samplesLoaded && samples.length === 0 && (
          <span className="t-label text-[var(--color-ink-faint)]">
            no samples cached — run scripts/build_samples.py
          </span>
        )}

        <div className="flex flex-wrap gap-x-6 gap-y-3">
          {samples.map((s) => {
            const active = state.sampleId === s.id;
            return (
              <button
                key={s.id}
                onClick={() => runSample(s)}
                disabled={busy}
                className="group relative pb-1 text-[0.8125rem] transition-colors duration-300 disabled:cursor-not-allowed"
                style={{ color: active ? "var(--color-ink)" : "var(--color-ink-mute)" }}
              >
                {s.label}
                {s.durationLabel && (
                  <span className="ml-2 text-[var(--color-ink-faint)]">{s.durationLabel}</span>
                )}
                <motion.span
                  className="absolute inset-x-0 bottom-0 h-px bg-[var(--color-ink)]"
                  initial={false}
                  animate={{ scaleX: active ? 1 : 0 }}
                  transition={{ duration: 0.4, ease: EASE }}
                  style={{ transformOrigin: "left" }}
                />
              </button>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-4">
          <input
            ref={fileRef}
            type="file"
            aria-label="Upload consultation audio"
            accept=".wav,.mp3,.m4a,.flac,.ogg,.webm,.mp4,audio/*"
            className="sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) runUpload(f);
              e.target.value = "";
            }}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={busy}
            className="btn-outline"
          >
            Upload audio
          </button>
          {busy && (
            <button onClick={cancel} className="t-label text-[var(--color-ink-mute)] underline">
              cancel
            </button>
          )}
        </div>
      </motion.div>

      {/* status strip — must never look frozen while a stage runs */}
      <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2">
        <span
          className="t-label flex items-center gap-2"
          style={{ color: state.stage === "error" ? "var(--color-sig-red)" : "var(--color-ink-2)" }}
        >
          {busy && <LivePulse />}
          {STAGE_LABEL[state.stage] ?? state.stage}
          {busy && state.stageStartedAt && (
            <span className="tabular-nums text-[var(--color-ink)]">
              <Elapsed since={state.stageStartedAt} />
            </span>
          )}
        </span>
        {state.live && (
          <span className="t-label text-[var(--color-ink-faint)]">
            running the real pipeline on this machine — around a minute for a short clip
          </span>
        )}
        {state.stage === "done" && state.timings && (
          <span className="t-label whitespace-nowrap text-[var(--color-ink-faint)]">
            {state.timings.transcribe_s}s · {state.timings.generate_s}s ·{" "}
            {state.timings.alarm_s}s
          </span>
        )}
        <div className="ml-auto hidden w-40 sm:block">
          <Waveform active={state.stage === "transcribe"} className="h-8" />
        </div>
      </div>

      <AnimatePresence>
        {state.error && (
          <motion.p
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 text-[0.8125rem] text-[var(--color-sig-red)]"
          >
            {state.error}
          </motion.p>
        )}
      </AnimatePresence>

      {/* the three columns */}
      <div className="mt-10 grid grid-cols-1 gap-10 lg:grid-cols-3 lg:gap-8">
        <div className={colClass}>
          <TranscriptPanel transcript={state.transcript} stage={state.stage} />
        </div>
        <div className={colClass}>
          <NotePanel
            note={state.note}
            spans={state.spans}
            stage={state.stage}
            linkedFlag={linkedFlag}
            onHoverFlag={setLinkedFlag}
          />
        </div>
        <div className={colClass}>
          <AlarmPanel
            alarm={state.alarm}
            unreliableMin={unreliableMin}
            stage={state.stage}
            linkedFlag={linkedFlag}
            onHoverFlag={setLinkedFlag}
          />
        </div>
      </div>

      {/* note actions */}
      <AnimatePresence>
        {state.note && (
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="show"
            exit={{ opacity: 0 }}
            className="mt-8 flex items-center gap-6 border-t border-[var(--color-hairline)] pt-6"
          >
            <button onClick={copyNote} className="btn-outline">
              {copied ? "Copied" : "Copy note"}
            </button>
            <button onClick={downloadNote} className="btn-outline">
              Download .txt
            </button>
            {state.alarm && (
              <span className="t-label text-[var(--color-ink-faint)]">
                {state.alarm.score_note}
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
