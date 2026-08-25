"use client";

import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { EASE, clipUp, fadeUp, stagger } from "@/lib/motion";
import IridescenceHero from "./IridescenceHero";
import Waveform from "./Waveform";

/** One masked line of the display headline. */
function Line({ i, children }: { i: number; children: React.ReactNode }) {
  return (
    <span className="block overflow-hidden pb-[0.08em]">
      <motion.span className="block" custom={i} variants={clipUp}>
        {children}
      </motion.span>
    </span>
  );
}

export default function Hero({ onStart }: { onStart: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();

  // the single restrained parallax: the statement drifts slower than scroll
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const drift = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : 64]);

  return (
    <section
      ref={ref}
      className="relative isolate flex min-h-svh flex-col justify-between px-6 pb-10 pt-8 md:px-[60px] md:pb-14"
    >
      {/* animated liquid iridescence + its vignette (replaces the achromatic radial) */}
      <IridescenceHero />

      <motion.div initial="hidden" animate="show" variants={stagger(0.09)}>
        {/* nav */}
        <motion.nav
          variants={fadeUp}
          className="flex items-baseline justify-between border-b border-[var(--color-hairline)] pb-6"
        >
          <span className="t-label text-[var(--color-ink)]">Clarion</span>
          <span className="t-label hidden text-[var(--color-ink-mute)] sm:block">
            Speech → Clinical Note → Reliability
          </span>
          <span className="t-label text-[var(--color-ink-mute)]">Local · Free</span>
        </motion.nav>

        {/* marginalia + statement */}
        <motion.div style={{ y: drift }} className="mt-16 md:mt-24">
          <motion.p variants={fadeUp} className="t-label mb-8 text-[var(--color-ink-mute)]">
            001 — The alarm, not the autocomplete
          </motion.p>

          <h1 className="t-hero max-w-[16ch] text-[var(--color-ink)]">
            <Line i={0}>Every note</Line>
            <Line i={1}>
              it writes, it <em className="font-normal italic">doubts</em>
            </Line>
            <Line i={2}>out loud.</Line>
          </h1>

          <motion.p
            variants={fadeUp}
            className="t-body mt-10 max-w-[52ch] text-[var(--color-ink-2)]"
          >
            Clarion turns a consultation recording into a SOAP note — then checks every
            claim in that note against what was actually said, and tells you which notes
            deserve a second look.
          </motion.p>

          <motion.div variants={fadeUp} className="mt-12 flex flex-wrap items-center gap-6">
            <button onClick={onStart} className="btn-outline">
              Run an analysis
            </button>
            <span className="t-label text-[var(--color-ink-faint)]">
              Whisper · MedGemma · claim verifier — all on-device
            </span>
          </motion.div>
        </motion.div>
      </motion.div>

      {/* waveform + footer hairline */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9, ease: EASE, delay: 0.7 }}
        className="mt-16"
      >
        <Waveform delay={0.75} />
        <div className="mt-6 flex items-center justify-between border-t border-[var(--color-hairline)] pt-5">
          <span className="t-label text-[var(--color-ink-faint)]">PriMock57 · 57 consultations</span>
          <span className="t-label text-[var(--color-ink-faint)]">Scroll</span>
        </div>
      </motion.div>
    </section>
  );
}
