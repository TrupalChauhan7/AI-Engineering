"use client";

import { motion } from "framer-motion";
import { EASE } from "@/lib/motion";

/** Deterministic pseudo-random bar heights — same shape every render/build. */
function bars(count: number) {
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const a = Math.sin(i * 0.7) * 0.5 + 0.5;
    const b = Math.sin(i * 0.17 + 1.3) * 0.5 + 0.5;
    const envelope = Math.sin((i / count) * Math.PI); // quiet at both ends
    // round: full-precision floats serialise differently on server vs client
    out.push(Number((0.12 + (a * 0.55 + b * 0.45) * envelope * 0.88).toFixed(4)));
  }
  return out;
}

const HEIGHTS = bars(96);

export default function Waveform({
  active = false,
  delay = 0,
  className = "",
}: {
  /** pulses while the transcribe stage is running */
  active?: boolean;
  delay?: number;
  className?: string;
}) {
  return (
    <div
      className={`flex h-16 w-full items-center gap-[3px] ${className}`}
      aria-hidden="true"
    >
      {HEIGHTS.map((h, i) => (
        <motion.span
          key={i}
          className="flex-1 origin-bottom bg-[var(--color-ink-faint)]"
          style={{ height: `${h * 100}%`, willChange: "transform, opacity" }}
          initial={{ scaleY: 0, opacity: 0 }}
          animate={
            active
              ? {
                  scaleY: [1, 1 + h * 0.5, 1],
                  opacity: [0.55, 1, 0.55],
                  transition: {
                    duration: 1.15,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: (i % 12) * 0.045,
                  },
                }
              : { scaleY: 1, opacity: 0.55 }
          }
          transition={{ duration: 0.7, ease: EASE, delay: delay + i * 0.006 }}
          whileInView={{ scaleY: 1, opacity: 0.55 }}
          viewport={{ once: true }}
        />
      ))}
    </div>
  );
}
