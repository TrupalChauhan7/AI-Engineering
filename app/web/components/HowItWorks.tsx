"use client";

import { motion } from "framer-motion";
import { drawX, drawY, fadeUp, sectionReveal, stagger } from "@/lib/motion";

/** The pipeline, as stations on a measured rail. */
const STEPS = [
  { n: "01", label: "Audio", note: "Consultation recording, mixed to one track" },
  { n: "02", label: "Whisper", note: "Local ASR → transcript" },
  { n: "03", label: "MedGemma 4B", note: "Transcript → SOAP note" },
  { n: "04", label: "Claim verifier", note: "Every claim checked against the transcript" },
  { n: "05", label: "Reliability", note: "Unsupported + omitted → verdict" },
];

/**
 * A station node sitting on the rail. The last stage (Reliability) resolves into
 * a ring — a deliberate miniature of the verdict ring in the workspace, so the
 * eye recognises "this stage produces that". Ink only; the signal colour stays
 * reserved for a real verdict.
 */
function Node({ terminal = false }: { terminal?: boolean }) {
  if (terminal) {
    return (
      <span className="relative block h-[15px] w-[15px] rounded-full border border-[var(--color-border-strong)] bg-[var(--color-canvas)] transition-all duration-300 group-hover:scale-110 group-hover:border-[var(--color-ink)]">
        <span className="absolute inset-[3px] rounded-full border border-[var(--color-ink-mute)]" />
      </span>
    );
  }
  return (
    <span className="block h-[9px] w-[9px] rounded-full border border-[var(--color-border-strong)] bg-[var(--color-canvas)] transition-all duration-300 group-hover:scale-125 group-hover:border-[var(--color-ink)]" />
  );
}

function StepText({ n, label, note }: { n: string; label: string; note: string }) {
  return (
    <>
      <span className="t-label tabular-nums text-[var(--color-ink-faint)] transition-colors duration-300 group-hover:text-[var(--color-ink-2)]">
        {n}
      </span>
      <p className="mt-3 text-[0.9375rem] text-[var(--color-ink)]">{label}</p>
      <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--color-ink-mute)]">{note}</p>
    </>
  );
}

export default function HowItWorks() {
  const inView = {
    initial: "hidden",
    whileInView: "show",
    viewport: { once: true, margin: "-15% 0px" },
  } as const;

  return (
    <section
      id="how"
      className="border-t border-[var(--color-hairline)] px-6 py-24 md:px-[60px] md:py-32"
    >
      <motion.div {...sectionReveal}>
        <p className="t-label mb-6 text-[var(--color-ink-mute)]">003 — How it works</p>
        <h2 className="t-section max-w-[24ch] text-[var(--color-ink)]">
          It <em className="font-normal italic">never</em> sees the answer.
        </h2>
        <p className="t-body mt-8 max-w-[62ch] text-[var(--color-ink-2)]">
          The verifier grades the note against the transcript that produced it — not against
          a reference note written by a clinician. That is the whole point: there is no gold
          answer at the bedside, so a metric that needs one cannot be deployed. This one can.
        </p>
      </motion.div>

      {/* The pipeline as a rail that draws itself, tracing audio → reliability. */}
      <div className="mt-20">
        {/* desktop: horizontal rail */}
        <div className="relative hidden lg:block">
          <motion.div
            {...inView}
            variants={drawX}
            className="absolute inset-x-0 top-2 h-px bg-[var(--color-border-strong)]"
            style={{ transformOrigin: "left" }}
          />
          <motion.ol {...inView} variants={stagger(0.14)} className="grid grid-cols-5 gap-8">
            {STEPS.map((s, i) => (
              <motion.li key={s.n} variants={fadeUp} className="group relative">
                <div className="flex h-4 items-center">
                  <Node terminal={i === STEPS.length - 1} />
                </div>
                <div className="mt-6">
                  <StepText n={s.n} label={s.label} note={s.note} />
                </div>
              </motion.li>
            ))}
          </motion.ol>
        </div>

        {/* mobile: vertical rail (the flow survives on a phone) */}
        <div className="relative lg:hidden">
          <motion.div
            {...inView}
            variants={drawY}
            className="absolute bottom-2 left-[5.5px] top-2 w-px bg-[var(--color-border-strong)]"
            style={{ transformOrigin: "top" }}
          />
          <motion.ol {...inView} variants={stagger(0.12)} className="space-y-9">
            {STEPS.map((s, i) => (
              <motion.li key={s.n} variants={fadeUp} className="group relative pl-10">
                <div className="absolute left-0 top-0.5 flex w-3 justify-center">
                  <Node terminal={i === STEPS.length - 1} />
                </div>
                <StepText n={s.n} label={s.label} note={s.note} />
              </motion.li>
            ))}
          </motion.ol>
        </div>
      </div>

      <motion.div
        {...sectionReveal}
        className="mt-20 grid gap-10 border-t border-[var(--color-hairline)] pt-10 sm:grid-cols-2"
      >
        <p className="t-body text-[var(--color-ink-2)]">
          Validated on 47 held-out consultations scored once. Against human error counts the
          verifier ranked first of six metrics — ahead of BERTScore, which did not survive the
          move off the development set.
        </p>
        <p className="t-body text-[var(--color-ink-mute)]">
          Honest limit: no metric separated distinguishably from another, and all sit well
          below the agreement ceiling between human raters. Clarion tells you where to look.
          It does not tell you a note is safe.
        </p>
      </motion.div>
    </section>
  );
}
