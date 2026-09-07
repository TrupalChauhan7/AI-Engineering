"use client";

/**
 * Tether — the signature "grounding" gesture.
 *
 * When a flagged claim is hovered (in the note or the reliability rail), a red
 * hairline arcs from the flagged phrase in the NOTE across to its entry in the
 * RELIABILITY rail — drawing, on screen, the one idea the whole project is about:
 * every claim is checked against the source, and here is the link.
 *
 * It reads the two DOM anchors by data attribute (`data-flag` in the note,
 * `data-railflag` in the rail), both descendants of `containerRef`, and recomputes
 * on scroll/resize so the line tracks. lg-only (the columns are side by side);
 * on stacked mobile it renders nothing. Purely presentational, pointer-events off.
 */

import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { EASE } from "@/lib/motion";

interface Geom {
  d: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export default function Tether({
  containerRef,
  active,
}: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  active: number | null;
}) {
  const [geom, setGeom] = useState<Geom | null>(null);

  useEffect(() => {
    const cont = containerRef.current;
    if (active == null || !cont) {
      setGeom(null);
      return;
    }

    const compute = () => {
      const c = cont.getBoundingClientRect();
      const noteEl = cont.querySelector<HTMLElement>(`[data-flag="${active}"]`);
      const railEl = cont.querySelector<HTMLElement>(`[data-railflag="${active}"]`);
      if (!noteEl || !railEl) {
        setGeom(null);
        return;
      }
      const n = noteEl.getBoundingClientRect();
      const r = railEl.getBoundingClientRect();
      // right edge of the note phrase -> left edge of the rail entry
      const x1 = n.right - c.left;
      const y1 = n.top + n.height / 2 - c.top;
      const x2 = r.left - c.left;
      const y2 = r.top + r.height / 2 - c.top;
      const mx = (x1 + x2) / 2;
      setGeom({
        d: `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`,
        x1,
        y1,
        x2,
        y2,
      });
    };

    compute();
    const raf = requestAnimationFrame(compute);
    // capture:true catches scrolling inside the note/rail panels, not just window
    window.addEventListener("scroll", compute, true);
    window.addEventListener("resize", compute);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", compute, true);
      window.removeEventListener("resize", compute);
    };
  }, [active, containerRef]);

  if (!geom) return null;

  return (
    <svg
      className="pointer-events-none absolute inset-0 z-30 hidden h-full w-full overflow-visible lg:block"
      aria-hidden
    >
      <motion.path
        key={active}
        d={geom.d}
        fill="none"
        stroke="var(--color-sig-red)"
        strokeWidth="1"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 0.85 }}
        transition={{ duration: 0.45, ease: EASE }}
      />
      <circle cx={geom.x1} cy={geom.y1} r="2.5" fill="var(--color-sig-red)" />
      <circle cx={geom.x2} cy={geom.y2} r="2.5" fill="var(--color-sig-red)" />
    </svg>
  );
}
