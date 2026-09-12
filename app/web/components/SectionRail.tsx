"use client";

/**
 * SectionRail — the page's measured spine.
 *
 * A quiet vertical rail in the left margin: a hairline track whose fill tracks
 * scroll progress, with a tick per section that marks where you are and jumps
 * there on click. It extends the "measured instrument" motif (the same rail as
 * the pipeline in How-it-works) into site-wide orientation — the one thing a
 * long single-scroll page lacked. Desktop only (it lives in margin space);
 * hidden where there is no margin to spare. Purely an aid: labels stay hidden
 * until hover so it never competes with the content.
 */

import { motion, useReducedMotion, useScroll, useSpring } from "framer-motion";
import { useEffect, useState } from "react";

const SECTIONS = [
  { id: "top", n: "001", label: "Overview" },
  { id: "workspace", n: "002", label: "Workspace" },
  { id: "how", n: "003", label: "How it works" },
];

export default function SectionRail() {
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll();
  const fill = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.4 });
  const [active, setActive] = useState("top");

  useEffect(() => {
    const els = SECTIONS.map((s) => document.getElementById(s.id)).filter(Boolean) as HTMLElement[];
    if (!els.length) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id);
        });
      },
      // fire when a section crosses the vertical middle of the viewport
      { rootMargin: "-45% 0px -45% 0px", threshold: 0 },
    );
    els.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  const go = (id: string) =>
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <nav
      aria-label="Sections"
      className="fixed left-6 top-1/2 z-40 hidden -translate-y-1/2 lg:block"
    >
      <div className="relative flex flex-col gap-7 py-1">
        {/* track */}
        <span
          aria-hidden
          className="absolute left-[3px] bottom-1 top-1 w-px bg-[var(--color-hairline)]"
        />
        {/* progress fill — grows downward as the page scrolls */}
        <motion.span
          aria-hidden
          className="absolute left-[3px] bottom-1 top-1 w-px origin-top bg-[var(--color-border-strong)]"
          style={{ scaleY: reduced ? 1 : fill }}
        />
        {SECTIONS.map((s) => {
          const on = active === s.id;
          return (
            <button
              key={s.id}
              onClick={() => go(s.id)}
              aria-current={on ? "true" : undefined}
              className="group relative flex items-center gap-3"
            >
              <span
                aria-hidden
                className="relative z-10 block h-[7px] w-[7px] rounded-full border bg-[var(--color-canvas)] transition-transform duration-300"
                style={{
                  borderColor: on ? "var(--color-ink)" : "var(--color-border-strong)",
                  transform: on ? "scale(1.35)" : "scale(1)",
                }}
              />
              <span
                className="t-label -translate-x-1 whitespace-nowrap opacity-0 transition-all duration-300 group-hover:translate-x-0 group-hover:opacity-100"
                style={{ color: on ? "var(--color-ink)" : "var(--color-ink-mute)" }}
              >
                {s.n} · {s.label}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
