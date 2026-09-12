"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";
import { EASE } from "@/lib/motion";
import type { Stage } from "@/lib/types";
import { StageDot } from "../StageProgress";

/** Split a transcript into speaker turns so the stream reads like dialogue. */
function turns(transcript: string) {
  if (!transcript) return [];
  return transcript
    .split(/(?=\b(?:Doctor|Patient)\s*:)/g)
    .map((t) => t.trim())
    .filter(Boolean)
    .map((line) => {
      const m = line.match(/^(Doctor|Patient)\s*:\s*([\s\S]*)$/);
      return m ? { who: m[1], text: m[2] } : { who: "", text: line };
    });
}

export default function TranscriptPanel({
  transcript,
  stage,
}: {
  transcript: string;
  stage: Stage;
}) {
  const lines = useMemo(() => turns(transcript), [transcript]);
  const running = stage === "transcribe" && !transcript;

  return (
    <div className="relative flex min-h-0 flex-col lg:h-full">
      <header className="flex items-baseline justify-between border-b border-[var(--color-hairline)] pb-3">
        <span className="t-label flex items-center gap-2 text-[var(--color-ink-2)]">
          <StageDot state={transcript ? "done" : stage === "transcribe" ? "active" : "idle"} />
          01 — Transcript
        </span>
        <span className="t-label text-[var(--color-ink-faint)]">Whisper</span>
      </header>

      <div
        data-lenis-prevent
        className="scroll-panel mt-5 min-h-0 flex-1 space-y-4 overflow-y-auto pr-2"
      >
        {running && (
          <p className="t-note text-[var(--color-ink-2)]">Listening…</p>
        )}
        {!running && !lines.length && (
          <p className="t-note text-[var(--color-ink-2)]">
            Choose a consultation or upload audio.
          </p>
        )}

        {lines.map((line, i) => (
          <motion.p
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: EASE, delay: Math.min(i * 0.06, 1.2) }}
            className="t-note text-[var(--color-ink-read)]"
            style={{ willChange: "transform, opacity" }}
          >
            {line.who && (
              <span className="mr-2 text-[var(--color-ink-faint)]">{line.who}:</span>
            )}
            {line.text}
          </motion.p>
        ))}
      </div>
        {/* hints there is more below — only when there's content that scrolls,
            so the fade never sits on top of the empty-state placeholder */}
        {lines.length > 0 && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 z-20 hidden h-12 lg:block"
            style={{
              background:
                "linear-gradient(to bottom, transparent, var(--color-canvas) 88%)",
            }}
          />
        )}
    </div>
  );
}
