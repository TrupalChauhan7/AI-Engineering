"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import { EASE } from "@/lib/motion";
import type { Span, Stage } from "@/lib/types";

const SCAN_MS = 1100;

interface Segment {
  text: string;
  flagIndex: number; // -1 = unflagged
}

/**
 * Render the generator's `**Heading:**` markers as real emphasis.
 *
 * Purely presentational: the underlying note string is never rewritten, so the
 * character offsets the spans are built from stay valid, and copy/download hand
 * over exactly what the model produced.
 */
function richText(text: string, keyBase: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={`${keyBase}-${i}`} className="font-medium text-[var(--color-ink)]">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={`${keyBase}-${i}`}>{part}</span>
    ),
  );
}

/** Split the note into flagged / unflagged segments from character spans. */
function segment(note: string, spans: Span[]): Segment[] {
  const valid = spans
    .map((s, i) => ({ ...s, i }))
    .filter((s) => s.start >= 0 && s.end > s.start && s.end <= note.length)
    .sort((a, b) => a.start - b.start);

  // merge overlaps: several claims can land on the same line
  const merged: { start: number; end: number; i: number }[] = [];
  for (const s of valid) {
    const last = merged[merged.length - 1];
    if (last && s.start < last.end) last.end = Math.max(last.end, s.end);
    else merged.push({ start: s.start, end: s.end, i: s.i });
  }

  const out: Segment[] = [];
  let cursor = 0;
  for (const m of merged) {
    if (m.start > cursor) out.push({ text: note.slice(cursor, m.start), flagIndex: -1 });
    out.push({ text: note.slice(m.start, m.end), flagIndex: m.i });
    cursor = m.end;
  }
  if (cursor < note.length) out.push({ text: note.slice(cursor), flagIndex: -1 });
  return out;
}

export default function NotePanel({
  note,
  spans,
  stage,
  linkedFlag,
  onHoverFlag,
}: {
  note: string;
  spans: Span[];
  stage: Stage;
  linkedFlag: number | null;
  onHoverFlag: (i: number | null) => void;
}) {
  const reduced = useReducedMotion();
  const bodyRef = useRef<HTMLDivElement>(null);
  const [drawn, setDrawn] = useState<Set<number>>(new Set());
  const [scanning, setScanning] = useState(false);

  const segments = useMemo(() => segment(note, spans), [note, spans]);
  const verifying = stage === "reliability" || stage === "done";

  /* The sweep: measure where each flagged span sits, then draw its underline
     at the moment the scan line actually crosses it. */
  useEffect(() => {
    if (!verifying || !note) return;
    if (reduced) {
      setDrawn(new Set(segments.map((s) => s.flagIndex).filter((i) => i >= 0)));
      return;
    }

    setScanning(true);
    setDrawn(new Set());
    const container = bodyRef.current;
    const timers: ReturnType<typeof setTimeout>[] = [];

    if (container) {
      const height = container.scrollHeight || 1;
      container.querySelectorAll<HTMLElement>("[data-flag]").forEach((el) => {
        const idx = Number(el.dataset.flag);
        const ratio = Math.min(1, (el.offsetTop + el.offsetHeight / 2) / height);
        timers.push(
          setTimeout(() => setDrawn((prev) => new Set(prev).add(idx)), ratio * SCAN_MS),
        );
      });
    }
    timers.push(setTimeout(() => setScanning(false), SCAN_MS + 120));

    return () => timers.forEach(clearTimeout);
  }, [verifying, note, segments, reduced]);

  return (
    <div className="relative flex min-h-0 flex-col lg:h-full">
      <header className="flex items-baseline justify-between border-b border-[var(--color-hairline)] pb-3">
        <span className="t-label text-[var(--color-ink-2)]">02 — Generated note</span>
        <span className="t-label text-[var(--color-ink-faint)]">MedGemma 4B</span>
      </header>

      <div
        ref={bodyRef}
        data-lenis-prevent
        className="scroll-panel relative mt-5 min-h-0 flex-1 overflow-y-auto pr-2"
      >
        {/* the verification sweep — 1px hairline, travels once */}
        {scanning && (
          <motion.div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 z-10 h-px bg-[var(--color-ink)]"
            style={{ willChange: "transform", opacity: 0.85 }}
            initial={{ top: 0 }}
            animate={{ top: "100%" }}
            transition={{ duration: SCAN_MS / 1000, ease: "linear" }}
          />
        )}

        {!note ? (
          <p className="t-note text-[var(--color-ink-faint)]">Awaiting transcript…</p>
        ) : (
          <motion.p
            className="t-note whitespace-pre-wrap text-[var(--color-ink)]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, ease: EASE }}
          >
            {segments.map((seg, i) =>
              seg.flagIndex >= 0 ? (
                <span
                  key={i}
                  data-flag={seg.flagIndex}
                  data-drawn={drawn.has(seg.flagIndex)}
                  data-linked={linkedFlag === seg.flagIndex}
                  className="flagged"
                  style={{ color: drawn.has(seg.flagIndex) ? undefined : "var(--color-ink)" }}
                  onMouseEnter={() => onHoverFlag(seg.flagIndex)}
                  onMouseLeave={() => onHoverFlag(null)}
                >
                  {richText(seg.text, `f${i}`)}
                </span>
              ) : (
                <span key={i}>{richText(seg.text, `p${i}`)}</span>
              ),
            )}
          </motion.p>
        )}
      </div>
        {/* hints there is more below — only where the panel scrolls */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 z-20 hidden h-12 lg:block"
          style={{
            background:
              "linear-gradient(to bottom, transparent, var(--color-canvas) 88%)",
          }}
        />
    </div>
  );
}
