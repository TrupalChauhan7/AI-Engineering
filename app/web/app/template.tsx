"use client";

/**
 * Route transition. Next's App Router re-mounts template.tsx on every
 * navigation, so this gives each route a composed entrance instead of an
 * instant swap — the connective tissue that makes "/" and "/audit" feel like
 * one instrument rather than two pages.
 *
 * Deliberately OPACITY-ONLY: a transform here would create a containing block
 * and break the fixed section spine (SectionRail) for the duration of the
 * animation. A fade needs no transform, so fixed positioning stays intact.
 * Honours reduced-motion.
 */

import { motion, useReducedMotion } from "framer-motion";
import { EASE } from "@/lib/motion";

export default function Template({ children }: { children: React.ReactNode }) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      initial={reduced ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}
