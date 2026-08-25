"use client";

import { useCallback } from "react";
import Footer from "@/components/Footer";
import Hero from "@/components/Hero";
import HowItWorks from "@/components/HowItWorks";
import SmoothScroll from "@/components/SmoothScroll";
import Workspace from "@/components/Workspace";

export default function Page() {
  const scrollToWorkspace = useCallback(() => {
    document.getElementById("workspace")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  return (
    <>
      <SmoothScroll />
      <main>
        <Hero onStart={scrollToWorkspace} />
        <Workspace />
        <HowItWorks />
      </main>
      <Footer />
    </>
  );
}
