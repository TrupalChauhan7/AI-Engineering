import type { Variants } from "framer-motion";

/** The one easing curve the whole product moves on — patient, not snappy. */
export const EASE = [0.19, 1, 0.22, 1] as const;

/** A line that clip-reveals upward from behind a mask. */
export const clipUp: Variants = {
  hidden: { y: "110%" },
  show: (i = 0) => ({
    y: "0%",
    transition: { duration: 0.9, ease: EASE, delay: 0.09 * i },
  }),
};

/** Quiet fade + small rise. Keep the offset small so it reads as a fade. */
export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.7, ease: EASE, delay: 0.08 * i },
  }),
};

/** Container that staggers its children. */
export const stagger = (step = 0.06, delay = 0): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: step, delayChildren: delay } },
});

/** Section reveal used on scroll — once, never re-triggering. */
export const sectionReveal = {
  initial: "hidden",
  whileInView: "show",
  viewport: { once: true, margin: "-12% 0px -12% 0px" },
  variants: fadeUp,
} as const;
