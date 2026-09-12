"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";

/**
 * The anti-frozen indicator.
 *
 * A local pipeline stage can legitimately take a minute, and a static label for
 * a minute reads as a crash. This ticks a real elapsed counter every second and
 * pulses a dot, so the screen always proves something is still happening.
 */

function fmt(ms: number) {
  const total = Math.floor(ms / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [since]);
  return <span suppressHydrationWarning>{fmt(Math.max(0, now - since))}</span>;
}

export function LivePulse() {
  const reduced = useReducedMotion();
  return (
    <motion.span
      aria-hidden
      className="inline-block h-[5px] w-[5px] rounded-full bg-[var(--color-ink-2)]"
      animate={reduced ? { opacity: 0.8 } : { opacity: [0.25, 1, 0.25] }}
      transition={reduced ? undefined : { duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
    />
  );
}

export type StageState = "idle" | "active" | "done";

/**
 * A per-stage node for a panel header, in the same node vocabulary as the
 * section spine and the pipeline rail. It makes the pipeline's progress visible
 * spatially: hollow (waiting) → pulsing (running) → filled (done). As a run
 * proceeds, the three panel nodes light left-to-right across the columns.
 */
export function StageDot({ state }: { state: StageState }) {
  const reduced = useReducedMotion();
  if (state === "active") {
    return (
      <motion.span
        aria-hidden
        className="inline-block h-[6px] w-[6px] rounded-full bg-[var(--color-ink-2)]"
        animate={reduced ? { opacity: 0.8 } : { opacity: [0.25, 1, 0.25] }}
        transition={reduced ? undefined : { duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
      />
    );
  }
  return (
    <span
      aria-hidden
      className="inline-block h-[6px] w-[6px] rounded-full border transition-colors duration-500"
      style={{
        borderColor: state === "done" ? "var(--color-ink)" : "var(--color-border-strong)",
        background: state === "done" ? "var(--color-ink)" : "transparent",
      }}
    />
  );
}
