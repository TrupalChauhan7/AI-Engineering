import type { Metadata, Viewport } from "next";
import { Cormorant_Garamond, Hanken_Grotesk, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

/* Self-hosted at build time by next/font — no runtime fetch, no layout shift. */
const cormorant = Cormorant_Garamond({
  variable: "--font-cormorant",
  subsets: ["latin"],
  weight: ["300", "400"],
  style: ["normal", "italic"],
  display: "swap",
});

const hanken = Hanken_Grotesk({
  variable: "--font-hanken",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Clarion — reliability for AI clinical notes",
  description:
    "Turns a consultation recording into a SOAP note, then verifies every claim against the transcript and raises an reliability on the notes that need a second look.",
};

export const viewport: Viewport = {
  themeColor: "#0A0A0B",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    /* The font variables must sit on <html>: globals.css declares
       --font-display: var(--font-cormorant) inside @theme at :root, and a
       custom property's var() resolves in the scope it is DECLARED in. On
       <body> they would be out of scope and every face would fall back. */
    <html
      lang="en"
      className={`${cormorant.variable} ${hanken.variable} ${plexMono.variable}`}
    >
      <body>{children}</body>
    </html>
  );
}
