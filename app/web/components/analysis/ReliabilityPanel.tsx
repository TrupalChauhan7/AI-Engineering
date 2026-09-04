"use client";

import { animate, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { EASE, fadeUp, stagger } from "@/lib/motion";
import { VERDICT_COLOR, VERDICT_WORD, type Reliability, type Stage } from "@/lib/types";

/** Counts tick up rather than snapping — the numbers should feel *counted*. */
function Count({ to, delay = 0 }: { to: number; delay?: number }) {
  const [n, setN] = useState(0);
  const reduced = useReducedMotion();
  useEffect(() => {
    if (reduced) return setN(to);
    setN(0);
    const controls = animate(0, to, {
      duration: 0.9,
      delay,
      ease: EASE,
      onUpdate: (v) => setN(Math.round(v)),
    });
    return () => controls.stop();
  }, [to, delay, reduced]);
  return <>{n}</>;
}

/**
 * The ring draws itself, and its fill encodes the reliability verdict — not error count.
 *
 * Filling by error count made a clean note show a nearly empty ring, which
 * reads as "40% reliable" and undersells the result. Inverted: a note with few
 * flags is a fuller, greener ring; an unreliable one is a red sliver. So
 * "fuller and greener = more trustworthy" holds at a glance, which is the only
 * thing a viewer reads in the first second.
 */
function VerdictRing({ reliability, show, unreliableMin = 12 }: {
  reliability: Reliability;
  show: boolean;
  unreliableMin?: number;
}) {
  const reduced = useReducedMotion();
  const R = 54;
  const C = 2 * Math.PI * R;
  const frac = Math.min(1, Math.max(0.05, 1 - reliability.combined / unreliableMin));
  const color = VERDICT_COLOR[reliability.verdict];

  return (
    <svg viewBox="0 0 128 128" className="h-32 w-32" aria-hidden>
      <circle cx="64" cy="64" r={R} fill="none" stroke="var(--color-hairline)" strokeWidth="1" />
      <motion.circle
        cx="64"
        cy="64"
        r={R}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        transform="rotate(-90 64 64)"
        strokeDasharray={C}
        initial={{ strokeDashoffset: C }}
        animate={{ strokeDashoffset: show ? C * (1 - frac) : C }}
        transition={{ duration: reduced ? 0 : 1.2, ease: EASE, delay: reduced ? 0 : 0.2 }}
      />
    </svg>
  );
}

export default function ReliabilityPanel({
  reliability,
  stage,
  linkedFlag,
  onHoverFlag,
  unreliableMin = 12,
}: {
  reliability: Reliability | null;
  stage: Stage;
  unreliableMin?: number;
  linkedFlag: number | null;
  onHoverFlag: (i: number | null) => void;
}) {
  const show = Boolean(reliability) && (stage === "reliability" || stage === "done");
  const railRef = useRef<HTMLDivElement>(null);

  return (
    <div className="flex min-h-0 flex-col lg:h-full">
      <header className="flex items-baseline justify-between border-b border-[var(--color-hairline)] pb-3">
        <span className="t-label text-[var(--color-ink-2)]">03 — Reliability</span>
        <span className="t-label text-[var(--color-ink-faint)]">Claim verifier</span>
      </header>

      {!reliability ? (
        <p className="t-note mt-5 text-[var(--color-ink-faint)]">
          {stage === "generate" ? "Note drafted — verifying next…" : "Awaiting note…"}
        </p>
      ) : (
        <motion.div
          ref={railRef}
          initial="hidden"
          animate={show ? "show" : "hidden"}
          variants={stagger(0.08, 0.15)}
          data-lenis-prevent
          className="scroll-panel mt-5 min-h-0 flex-1 overflow-y-auto pr-2"
        >
          {/* verdict */}
          <div className="flex items-center gap-6">
            <VerdictRing reliability={reliability} show={show} unreliableMin={unreliableMin} />
            <div>
              <motion.p
                variants={fadeUp}
                className="t-verdict"
                style={{ color: VERDICT_COLOR[reliability.verdict] }}
              >
                {VERDICT_WORD[reliability.verdict]}
              </motion.p>
              {/* The COMBINED total leads: it is the one signal the research
                  validated. The breakdown stays underneath as its provenance. */}
              <motion.p
                variants={fadeUp}
                className="t-label mt-3"
                style={{ color: VERDICT_COLOR[reliability.verdict] }}
              >
                <Count to={reliability.combined} delay={0.25} /> combined flags
              </motion.p>
              <motion.p variants={fadeUp} className="t-label mt-2 text-[var(--color-ink-mute)]">
                <Count to={reliability.n_unsupported} delay={0.3} /> unsupported ·{" "}
                <Count to={reliability.n_omitted} delay={0.4} /> omitted
              </motion.p>
            </div>
          </div>

          {/* unsupported claims */}
          <motion.div variants={fadeUp} className="mt-9">
            <p className="t-label mb-3 text-[var(--color-ink-mute)]">
              Claims to verify against the transcript
            </p>
            <ul className="space-y-0">
              {reliability.unsupported_claims.length === 0 && (
                <li className="t-note text-[var(--color-ink-faint)]">
                  None — every claim traced to the transcript.
                </li>
              )}
              {reliability.unsupported_claims.map((c, i) => (
                <motion.li
                  key={i}
                  variants={fadeUp}
                  custom={i}
                  onMouseEnter={() => onHoverFlag(i)}
                  onMouseLeave={() => onHoverFlag(null)}
                  className="cursor-default border-b border-[var(--color-hairline)] py-3 text-[0.8125rem] leading-relaxed transition-colors duration-200"
                  style={{
                    color:
                      linkedFlag === i ? "var(--color-sig-red)" : "var(--color-ink-2)",
                  }}
                >
                  <span className="t-label mr-3 text-[var(--color-sig-red)]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {c}
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* omitted facts */}
          <motion.div variants={fadeUp} className="mt-9">
            <p className="t-label mb-3 text-[var(--color-ink-mute)]">
              Details possibly said, not in the note
            </p>
            <ul className="space-y-0">
              {reliability.omitted_facts.length === 0 && (
                <li className="t-note text-[var(--color-ink-faint)]">None — nothing dropped.</li>
              )}
              {reliability.omitted_facts.map((f, i) => (
                <motion.li
                  key={i}
                  variants={fadeUp}
                  custom={i}
                  className="border-b border-[var(--color-hairline)] py-3 text-[0.8125rem] leading-relaxed text-[var(--color-ink-2)]"
                >
                  <span className="t-label mr-3 text-[var(--color-ink-faint)]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {f}
                </motion.li>
              ))}
            </ul>
          </motion.div>

          <motion.div variants={fadeUp} className="mt-8 space-y-3">
            {/* The research validated the note-level COMBINED COUNT, not the
                individual verdicts. The UI must not imply otherwise. */}
            <p className="t-label text-[var(--color-ink-faint)]">
              The checker&rsquo;s flags, for a clinician to confirm — individual flags can be
              imperfect; the validated signal is the overall count.
            </p>
            <p className="t-label text-[var(--color-ink-faint)]">
              Prototype bands, DEV-calibrated. Flags notes worth a second look — it does not
              certify one as safe.
            </p>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}
