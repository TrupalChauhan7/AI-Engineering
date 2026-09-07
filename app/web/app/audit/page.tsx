"use client";

/**
 * /audit — the reviewer view over the audit trail.
 *
 * Read-only. It consults what the pipeline already recorded (GET /api/runs,
 * /api/runs/{id}, /api/metrics): every run, its verdict and flags, and the
 * provenance that makes it reproducible — which models and prompt versions
 * produced it. The visible face of the Stage-2 audit trail + observability,
 * dressed in the same editorial language as the landing page (motion, the
 * serif/mono interplay, value-depth surfaces, the verdict ring).
 */

import { animate, AnimatePresence, motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { EASE, fadeUp, sectionReveal, stagger } from "@/lib/motion";
import type { Metrics, RunDetail, RunSummary, Verdict } from "@/lib/types";
import { VERDICT_COLOR, VERDICT_WORD } from "@/lib/types";

const VERDICTS: Verdict[] = ["reliable", "review", "unreliable"];
const STAGES = [
  ["transcribe_s", "Transcribe"],
  ["generate_s", "Generate"],
  ["reliability_s", "Reliability"],
] as const;

function fmtTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function secs(v: number | null): string {
  return v == null ? "—" : `${v.toFixed(1)}s`;
}

/** Counts tick up rather than snapping — the numbers should feel counted. */
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

/** Compact verdict ring — fuller + greener = more trustworthy (mirrors the app). */
function Ring({ combined, verdict, unreliableMin }: {
  combined: number;
  verdict: Verdict;
  unreliableMin: number;
}) {
  const reduced = useReducedMotion();
  const R = 40;
  const C = 2 * Math.PI * R;
  const frac = Math.min(1, Math.max(0.05, 1 - combined / unreliableMin));
  const color = VERDICT_COLOR[verdict];
  return (
    <svg viewBox="0 0 96 96" className="h-24 w-24 shrink-0" aria-hidden>
      <circle cx="48" cy="48" r={R} fill="none" stroke="var(--color-hairline)" strokeWidth="1" />
      <motion.circle
        cx="48"
        cy="48"
        r={R}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        transform="rotate(-90 48 48)"
        strokeDasharray={C}
        initial={{ strokeDashoffset: C }}
        animate={{ strokeDashoffset: C * (1 - frac) }}
        transition={{ duration: reduced ? 0 : 1.1, ease: EASE, delay: reduced ? 0 : 0.15 }}
      />
    </svg>
  );
}

export default function AuditPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rRuns, rMetrics] = await Promise.all([fetch("/api/runs"), fetch("/api/metrics")]);
      if (rRuns.status === 503 || rMetrics.status === 503) {
        throw new Error("The audit store is not initialised — start the API (make run).");
      }
      if (!rRuns.ok) throw new Error(`Could not load runs (${rRuns.status}).`);
      setRuns(await rRuns.json());
      setMetrics(rMetrics.ok ? await rMetrics.json() : null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const openRun = useCallback(async (id: string) => {
    try {
      const r = await fetch(`/api/runs/${id}`);
      if (r.ok) setSelected(await r.json());
    } catch {
      /* ignore — the list stays usable */
    }
  }, []);

  const verdictTotal = metrics
    ? VERDICTS.reduce((s, v) => s + (metrics.by_verdict[v] ?? 0), 0)
    : 0;

  return (
    <main className="relative min-h-svh px-6 py-10 md:px-[60px] md:py-14">
      {/* nav — echoes the hero's top bar */}
      <motion.nav
        initial="hidden"
        animate="show"
        variants={fadeUp}
        className="flex items-baseline justify-between border-b border-[var(--color-hairline)] pb-6"
      >
        <Link href="/" className="t-label text-[var(--color-ink)]">
          Clarion
        </Link>
        <span className="t-label hidden text-[var(--color-ink-mute)] sm:block">
          Speech → Clinical Note → Reliability
        </span>
        <div className="flex items-center gap-6">
          <button
            onClick={load}
            className="t-label text-[var(--color-ink-mute)] transition-colors hover:text-[var(--color-ink)]"
          >
            Refresh
          </button>
          <Link
            href="/"
            className="t-label text-[var(--color-ink-mute)] transition-colors hover:text-[var(--color-ink)]"
          >
            ← Clarion
          </Link>
        </div>
      </motion.nav>

      {/* title */}
      <motion.div initial="hidden" animate="show" variants={stagger(0.09, 0.1)} className="mt-16">
        <motion.p variants={fadeUp} className="t-label mb-6 text-[var(--color-ink-mute)]">
          003 — The audit trail
        </motion.p>
        <motion.h1 variants={fadeUp} className="t-section max-w-[18ch] text-[var(--color-ink)]">
          Every run, <em className="font-normal italic">on the record.</em>
        </motion.h1>
        <motion.p variants={fadeUp} className="t-body mt-8 max-w-[56ch] text-[var(--color-ink-2)]">
          Each analysis is stored with the exact models and prompt versions that produced it.
          Nothing here re-runs anything — it is the ledger you can point to and say
          <em className="font-normal italic"> this is what happened, and why.</em>
        </motion.p>
      </motion.div>

      {error && <p className="mt-10 text-[0.8125rem] text-[var(--color-sig-red)]">{error}</p>}

      {/* metrics — value-depth surface cards */}
      {metrics && (
        <motion.section
          {...sectionReveal}
          className="mt-14 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
        >
          {/* total */}
          <div className="bg-[var(--color-surface)] p-7">
            <p className="t-label text-[var(--color-ink-mute)]">Total runs</p>
            <p
              className="mt-4 font-light leading-none text-[var(--color-ink)]"
              style={{ fontFamily: "var(--font-display)", fontSize: "3.5rem" }}
            >
              <Count to={metrics.total} />
            </p>
          </div>

          {/* verdict distribution */}
          <div className="bg-[var(--color-surface)] p-7">
            <p className="t-label text-[var(--color-ink-mute)]">By verdict</p>
            <div className="mt-4 flex h-2 w-full overflow-hidden rounded-full bg-[var(--color-hairline)]">
              {verdictTotal > 0 &&
                VERDICTS.map((v) => {
                  const n = metrics.by_verdict[v] ?? 0;
                  if (!n) return null;
                  return (
                    <motion.span
                      key={v}
                      initial={{ width: 0 }}
                      animate={{ width: `${(n / verdictTotal) * 100}%` }}
                      transition={{ duration: 0.9, ease: EASE }}
                      style={{ background: VERDICT_COLOR[v] }}
                    />
                  );
                })}
            </div>
            <div className="mt-4 flex flex-col gap-1.5">
              {VERDICTS.map((v) => (
                <span key={v} className="flex items-center gap-2 text-[0.8125rem]">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ background: VERDICT_COLOR[v] }}
                  />
                  <span className="text-[var(--color-ink-2)]">{VERDICT_WORD[v]}</span>
                  <span className="ml-auto tabular-nums text-[var(--color-ink)]">
                    {metrics.by_verdict[v] ?? 0}
                  </span>
                </span>
              ))}
            </div>
          </div>

          {/* latency */}
          <div className="bg-[var(--color-surface)] p-7">
            <p className="t-label text-[var(--color-ink-mute)]">Latency · p50 / p90</p>
            <div className="mt-4 flex flex-col gap-2">
              {STAGES.map(([key, label]) => (
                <span key={key} className="flex items-center gap-2 text-[0.8125rem]">
                  <span className="text-[var(--color-ink-2)]">{label}</span>
                  <span className="ml-auto tabular-nums text-[var(--color-ink)]">
                    {secs(metrics.latency_s?.[key]?.p50 ?? null)}
                    <span className="text-[var(--color-ink-faint)]">
                      {" "}
                      / {secs(metrics.latency_s?.[key]?.p90 ?? null)}
                    </span>
                  </span>
                </span>
              ))}
            </div>
          </div>
        </motion.section>
      )}

      {/* runs */}
      <section className="mt-16">
        <p className="t-label mb-5 text-[var(--color-ink-mute)]">
          Recent runs {runs.length > 0 && `· ${runs.length}`}
        </p>

        {loading && <p className="t-label text-[var(--color-ink-faint)]">loading…</p>}

        {!loading && !error && runs.length === 0 && (
          <p className="text-[0.8125rem] text-[var(--color-ink-faint)]">
            No runs yet. Analyze a consultation on the{" "}
            <Link href="/" className="underline underline-offset-4">
              main page
            </Link>{" "}
            and it will appear here.
          </p>
        )}

        {runs.length > 0 && (
          <div className="overflow-x-auto">
            {/* header row */}
            <div className="grid grid-cols-[1.4fr_0.8fr_1.4fr_0.9fr_1fr] gap-4 border-b border-[var(--color-hairline)] pb-3">
              {["Time", "Domain", "Source", "Flags (u+o)", "Verdict"].map((h, i) => (
                <span
                  key={h}
                  className={`t-label text-[var(--color-ink-mute)] ${i === 3 ? "text-right" : ""}`}
                >
                  {h}
                </span>
              ))}
            </div>
            <motion.div initial="hidden" animate="show" variants={stagger(0.05)}>
              {runs.map((r) => (
                <motion.button
                  key={r.run_id}
                  variants={fadeUp}
                  onClick={() => openRun(r.run_id)}
                  className="group grid w-full grid-cols-[1.4fr_0.8fr_1.4fr_0.9fr_1fr] items-center gap-4 border-b border-[var(--color-hairline)] py-4 text-left text-[0.8125rem] transition-colors hover:bg-[var(--color-surface)]"
                >
                  <span className="tabular-nums text-[var(--color-ink-mute)] transition-colors group-hover:text-[var(--color-ink-2)]">
                    {fmtTime(r.created_at)}
                  </span>
                  <span className="capitalize text-[var(--color-ink-2)]">{r.domain}</span>
                  <span className="truncate text-[var(--color-ink-mute)]">
                    {r.source_name ?? r.source}
                  </span>
                  <span className="text-right tabular-nums text-[var(--color-ink)]">
                    {r.combined}{" "}
                    <span className="text-[var(--color-ink-faint)]">
                      ({r.n_unsupported}+{r.n_omitted})
                    </span>
                  </span>
                  <span
                    className="flex items-center gap-2"
                    style={{ color: VERDICT_COLOR[r.verdict] }}
                  >
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ background: VERDICT_COLOR[r.verdict] }}
                    />
                    {VERDICT_WORD[r.verdict]}
                  </span>
                </motion.button>
              ))}
            </motion.div>
          </div>
        )}
      </section>

      <AnimatePresence>
        {selected && <RunDetailPanel run={selected} onClose={() => setSelected(null)} />}
      </AnimatePresence>
    </main>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="t-label text-[var(--color-ink-mute)]">{label}</p>
      <p className="mt-1 break-words text-[0.8125rem] text-[var(--color-ink-2)]">{value}</p>
    </div>
  );
}

function RunDetailPanel({ run, onClose }: { run: RunDetail; onClose: () => void }) {
  const reduced = useReducedMotion();

  // Esc closes, and the back button is always reachable (sticky header).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex justify-end bg-black/60"
      onClick={onClose}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      role="dialog"
      aria-modal="true"
    >
      <motion.div
        className="scroll-panel relative h-full w-full max-w-2xl overflow-y-auto border-l border-[var(--color-border-strong)] bg-[var(--color-canvas)]"
        data-lenis-prevent
        onClick={(e) => e.stopPropagation()}
        initial={{ x: reduced ? 0 : "100%" }}
        animate={{ x: 0 }}
        exit={{ x: reduced ? 0 : "100%" }}
        transition={{ duration: reduced ? 0 : 0.5, ease: EASE }}
      >
        {/* sticky bar — the back button stays put however far you scroll */}
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-[var(--color-hairline)] bg-[var(--color-canvas)]/95 px-8 py-4 backdrop-blur">
          <button
            onClick={onClose}
            className="t-label text-[var(--color-ink)] transition-colors hover:text-[var(--color-ink-mute)]"
          >
            ← Back to runs
          </button>
          <span className="t-label text-[var(--color-ink-faint)]">Esc</span>
        </div>

        <div className="px-8 py-8">
          {/* verdict head — the ring ties it to the app */}
          <p className="t-label mb-4 text-[var(--color-ink-mute)]">
            Run · <span className="capitalize">{run.domain}</span>
          </p>
          <div className="flex items-center gap-6">
            <Ring combined={run.combined} verdict={run.verdict} unreliableMin={run.unreliable_min} />
            <div>
              <span className="t-verdict" style={{ color: VERDICT_COLOR[run.verdict] }}>
                {VERDICT_WORD[run.verdict]}
              </span>
              <p
                className="mt-3 t-label"
                style={{ color: VERDICT_COLOR[run.verdict] }}
              >
                <Count to={run.combined} delay={0.2} /> combined flags
              </p>
              <p className="mt-2 t-label text-[var(--color-ink-mute)]">
                {run.n_unsupported} unsupported · {run.n_omitted} omitted · bands{" "}
                {run.reliable_max}/{run.unreliable_min}
              </p>
            </div>
          </div>

          {/* provenance */}
          <div className="mt-9 grid grid-cols-2 gap-x-6 gap-y-5 border-y border-[var(--color-hairline)] py-6">
            <Field label="Recorded" value={fmtTime(run.created_at)} />
            <Field label="Request id" value={run.request_id ?? "—"} />
            <Field label="Source" value={run.source_name ?? run.source} />
            <Field label="Generator" value={run.generator_model} />
            <Field label="Verifier" value={run.verifier_model} />
            <Field label="Gen prompt" value={run.generation_prompt} />
            <Field label="Verify prompt" value={run.verify_prompt} />
            <Field label="Omission prompt" value={run.omission_verify_prompt} />
            <Field label="Facts prompt" value={run.transcript_facts_prompt} />
            <Field
              label="Timings"
              value={`${run.timings.transcribe_s}s · ${run.timings.generate_s}s · ${run.timings.reliability_s}s`}
            />
          </div>

          {run.unsupported_claims.length > 0 && (
            <div className="mt-8">
              <p className="t-label mb-3 text-[var(--color-sig-red)]">
                Unsupported claims ({run.unsupported_claims.length})
              </p>
              <ul className="flex flex-col">
                {run.unsupported_claims.map((c, i) => (
                  <li
                    key={i}
                    className="border-b border-[var(--color-hairline)] py-3 text-[0.8125rem] leading-relaxed text-[var(--color-ink-2)]"
                  >
                    <span className="t-label mr-3 text-[var(--color-sig-red)]">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {run.omitted_facts.length > 0 && (
            <div className="mt-8">
              <p className="t-label mb-3 text-[var(--color-sig-amber)]">
                Omitted facts ({run.omitted_facts.length})
              </p>
              <ul className="flex flex-col">
                {run.omitted_facts.map((f, i) => (
                  <li
                    key={i}
                    className="border-b border-[var(--color-hairline)] py-3 text-[0.8125rem] leading-relaxed text-[var(--color-ink-2)]"
                  >
                    <span className="t-label mr-3 text-[var(--color-ink-faint)]">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="mt-8">
            <p className="t-label mb-3 text-[var(--color-ink-mute)]">Generated note</p>
            <pre className="t-note whitespace-pre-wrap text-[var(--color-ink-read)]">{run.note}</pre>
          </div>

          <div className="mt-8">
            <p className="t-label mb-3 text-[var(--color-ink-mute)]">Transcript</p>
            <pre className="t-note whitespace-pre-wrap text-[var(--color-ink-read)]">
              {run.transcript}
            </pre>
          </div>

          {/* a second, plain back affordance at the very end of a long read */}
          <button
            onClick={onClose}
            className="mt-10 t-label text-[var(--color-ink-mute)] underline underline-offset-4 transition-colors hover:text-[var(--color-ink)]"
          >
            ← Back to runs
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
