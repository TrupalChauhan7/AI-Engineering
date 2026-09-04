"use client";

import { motion } from "framer-motion";
import { fadeUp, sectionReveal, stagger } from "@/lib/motion";

const STEPS = [
  { n: "01", label: "Audio", note: "Consultation recording, mixed to one track" },
  { n: "02", label: "Whisper", note: "Local ASR → transcript" },
  { n: "03", label: "MedGemma 4B", note: "Transcript → SOAP note" },
  { n: "04", label: "Claim verifier", note: "Every claim checked against the transcript" },
  { n: "05", label: "Reliability", note: "Unsupported + omitted → verdict" },
];

export default function HowItWorks() {
  return (
    <section className="border-t border-[var(--color-hairline)] px-6 py-24 md:px-[60px] md:py-32">
      <motion.div {...sectionReveal}>
        <p className="t-label mb-6 text-[var(--color-ink-mute)]">003 — How it works</p>
        <h2 className="t-section max-w-[24ch] text-[var(--color-ink)]">
          It never sees the answer.
        </h2>
        <p className="t-body mt-8 max-w-[62ch] text-[var(--color-ink-2)]">
          The verifier grades the note against the transcript that produced it — not against
          a reference note written by a clinician. That is the whole point: there is no gold
          answer at the bedside, so a metric that needs one cannot be deployed. This one can.
        </p>
      </motion.div>

      <motion.ol
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-10% 0px" }}
        variants={stagger(0.08)}
        className="mt-16 grid grid-cols-1 border-t border-[var(--color-hairline)] sm:grid-cols-2 lg:grid-cols-5"
      >
        {STEPS.map((s) => (
          <motion.li
            key={s.n}
            variants={fadeUp}
            className="border-b border-[var(--color-hairline)] px-0 py-8 lg:border-r lg:px-6 lg:first:pl-0 lg:last:border-r-0"
          >
            <span className="t-label text-[var(--color-ink-faint)]">{s.n}</span>
            <p className="mt-4 text-[0.9375rem] text-[var(--color-ink)]">{s.label}</p>
            <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--color-ink-mute)]">
              {s.note}
            </p>
          </motion.li>
        ))}
      </motion.ol>

      <motion.div
        {...sectionReveal}
        className="mt-16 grid gap-10 border-t border-[var(--color-hairline)] pt-10 sm:grid-cols-2"
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
