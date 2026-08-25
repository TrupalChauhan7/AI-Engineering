"use client";

import { motion } from "framer-motion";
import { sectionReveal } from "@/lib/motion";

export default function Footer() {
  return (
    <motion.footer
      {...sectionReveal}
      className="border-t border-[var(--color-hairline)] px-6 py-14 md:px-[60px]"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-y-6">
        <div>
          <p className="t-label text-[var(--color-ink)]">Clarion</p>
          <p className="t-label mt-3 text-[var(--color-ink-mute)]">
            Runs entirely on-device · no API keys · no data leaves the machine
          </p>
        </div>
        <div className="text-right">
          <p className="t-label text-[var(--color-ink-mute)]">
            Whisper · MedGemma 4B · Llama 3.1 8B · FastAPI · Next.js
          </p>
          <p className="t-label mt-3 text-[var(--color-ink-faint)]">
            Trupal Chauhan · ECS-8060 · PriMock57
          </p>
        </div>
      </div>
    </motion.footer>
  );
}
