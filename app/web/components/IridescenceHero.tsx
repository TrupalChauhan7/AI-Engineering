"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Animated "liquid iridescence" for the HOME HERO BACKGROUND only.
 *
 * WHY raw WebGL2 and not a library: this is one full-screen quad and one
 * fragment shader. `ogl`/`three` would add a dependency and a bundle for
 * something that is ~40 lines of GL setup.
 *
 * WHAT it draws: domain-warped fbm (Iñigo Quilez's warp — fbm of fbm of fbm)
 * mapped through a three-stop ramp over a near-black base. Domain warping is
 * what makes it read as flowing liquid rather than a moving gradient: the noise
 * field displaces its own lookup coordinates, so bands stretch and fold like
 * oil on water instead of sliding.
 *
 * NOTE ON THE DESIGN SYSTEM: globals.css reserves colour for the reliability. This
 * hero background is a deliberate, scoped exception; nothing else in the app
 * gains colour from it.
 */

const FIELD_SCALE = 0.5;

/** Bottom dissolve: opaque through the upper hero, eased to nothing at the edge. */
const FADE_OUT =
  "linear-gradient(to bottom," +
  " #000 0%, #000 56%," +
  " rgba(0,0,0,0.95) 66%," +
  " rgba(0,0,0,0.82) 74%," +
  " rgba(0,0,0,0.62) 81%," +
  " rgba(0,0,0,0.38) 88%," +
  " rgba(0,0,0,0.18) 93%," +
  " rgba(0,0,0,0.05) 97%," +
  " transparent 100%)";

const VERT = `#version 300 es
// Fullscreen triangle from gl_VertexID — no buffers, no attributes.
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
out vec4 fragColor;
uniform vec2  uRes;
uniform float uTime;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);          // smoothstep interpolation
  return mix(mix(hash(i),               hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  mat2 m = mat2(1.6, 1.2, -1.2, 1.6);        // rotate+scale between octaves
  for (int i = 0; i < 4; i++) { v += a * noise(p); p = m * p; a *= 0.5; }
  return v;
}

void main() {
  // aspect-correct, centred; dividing by .y keeps the flow undistorted on resize
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
  float t = uTime * 0.055;                   // slow, patient — no shimmer
  vec2 p = uv * 2.1;

  // --- domain warp: three chained fbm lookups -----------------------------
  vec2 q = vec2(fbm(p + vec2(0.0, t)),
                fbm(p + vec2(5.2, 1.3) - t * 0.7));
  vec2 r = vec2(fbm(p + 3.0 * q + vec2(1.7, 9.2) + t * 0.30),
                fbm(p + 3.0 * q + vec2(8.3, 2.8) - t * 0.25));
  float f = fbm(p + 3.4 * r);

  // --- ramp: green -> amber -> oxblood over near-black --------------------
  const vec3 GREEN   = vec3(160.0, 224.0, 171.0) / 255.0;
  const vec3 AMBER   = vec3(255.0, 172.0,  46.0) / 255.0;
  const vec3 OXBLOOD = vec3(165.0,  45.0,  37.0) / 255.0;
  const vec3 BASE    = vec3(10.0 / 255.0);

  // fbm concentrates around ~0.48, so a plain scale lands the whole image in
  // the middle of the ramp and everything reads amber. Spread the ramp across
  // the range the field actually occupies, then shift hue by the warp vector
  // itself — that second axis is what interleaves the bands and reads as
  // iridescence rather than one tinted cloud.
  float k = smoothstep(0.26, 0.74, f);
  k = clamp(k + (q.x - 0.5) * 0.55 + (r.y - 0.5) * 0.25, 0.0, 1.0);
  vec3 c = mix(GREEN, AMBER,   smoothstep(0.00, 0.50, k));
  c      = mix(c,     OXBLOOD, smoothstep(0.50, 1.00, k));

  // CONTRAST. A gentle pow() over the whole field gives a uniform dim wash.
  // What reads as molten glass is a hard tone curve: genuinely black shadows,
  // a fast shoulder, and pale specular cores where the field peaks. smoothstep
  // supplies the S-curve; the separate spec term is the bright cream centre of
  // each vein, which is what the eye reads as "lit from inside".
  // NB: no backticks in this file's shader comments -- FRAG is a template
  // literal, so one would truncate the shader source mid-string.
  const vec3 CREAM = vec3(240.0, 228.0, 201.0) / 255.0;
  float veins = smoothstep(0.05, 0.80, dot(r, r) * 1.6 + f * 0.55);
  float lum  = smoothstep(0.24, 0.78, f) * veins;
  float spec = pow(smoothstep(0.44, 0.88, f), 1.8) * veins;
  spec *= 1.0 - smoothstep(0.66, 0.98, k);  // cream lifts green/amber, never oxblood
  vec3 col = BASE + c * lum * 0.92 + CREAM * spec * 0.20;

  // ordered-ish dither: kills 8-bit banding across these very low-contrast ramps
  col += (hash(gl_FragCoord.xy * 0.7 + fract(uTime)) - 0.5) / 255.0;

  fragColor = vec4(col, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error("IridescenceHero shader:", gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

export default function IridescenceHero() {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;

    // Shader compile + link is SYNCHRONOUS main-thread work (~300ms under a 4x
    // CPU throttle) and this is decoration, not content. Defer it past first
    // paint so it lands outside the page's blocking window; the hero shows its
    // dark base until then, so nothing pops or shifts.
    let disposed = false;
    let teardown: (() => void) | null = null;
    const idle: number = (window.requestIdleCallback ?? ((f: () => void) => window.setTimeout(f, 1)))(
      () => { if (!disposed) teardown = init(canvas, host) ?? null; },
      { timeout: 1500 },
    );

    function init(canvas: HTMLCanvasElement, host: HTMLDivElement) {

    const gl = canvas.getContext("webgl2", {
      antialias: false,          // full-screen quad: nothing to antialias
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: "low-power",
    });
    if (!gl) {
      setFailed(true);           // -> static CSS gradient below
      return;
    }

    // ---- program --------------------------------------------------------
    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    const prog = vs && fs ? gl.createProgram() : null;
    if (!vs || !fs || !prog) {
      setFailed(true);
      return;
    }
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("IridescenceHero link:", gl.getProgramInfoLog(prog));
      setFailed(true);
      return;
    }
    gl.useProgram(prog);
    const uRes = gl.getUniformLocation(prog, "uRes");
    const uTime = gl.getUniformLocation(prog, "uTime");
    const vao = gl.createVertexArray();       // required in WebGL2 even with 0 attribs
    gl.bindVertexArray(vao);

    // ---- state ----------------------------------------------------------
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let clock = 0;          // accumulated, NOT wall time — pausing must not jump
    let last = 0;
    let visible = true;     // tab visible
    let onScreen = true;    // hero in viewport

    const draw = () => {
      gl.uniform2f(uRes, canvas.width, canvas.height);
      gl.uniform1f(uTime, clock);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };

    const resize = () => {
      // DPR capped at 2: beyond that a full-screen fragment shader costs
      // 2-3x the fill rate for no perceptible gain on this kind of soft field.
      // DPR ceiling of 2, then a field-resolution scale on top. This is a soft,
      // blurry noise field with no edges to preserve, so rendering it below
      // display resolution and letting the compositor upscale is visually
      // indistinguishable while cutting fragment work ~3x — which is what keeps
      // it smooth on weak GPUs and software rasterisers, not just this laptop.
      const dpr = Math.min(window.devicePixelRatio || 1, 2) * FIELD_SCALE;
      const w = Math.max(1, Math.round(host.clientWidth * dpr));
      const h = Math.max(1, Math.round(host.clientHeight * dpr));
      if (canvas.width === w && canvas.height === h) return;
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
      if (reduced || !raf) draw();            // repaint immediately when paused
    };

    const frame = (now: number) => {
      if (last) clock += Math.min((now - last) / 1000, 0.05); // clamp: no jump after a stall
      last = now;
      draw();
      raf = requestAnimationFrame(frame);
    };

    const start = () => {
      if (reduced || raf || !visible || !onScreen) return;
      last = 0;
      raf = requestAnimationFrame(frame);
    };
    const stop = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    };

    // ---- observers + listeners -----------------------------------------
    const ro = new ResizeObserver(resize);
    ro.observe(host);

    const io = new IntersectionObserver(
      ([e]) => {
        onScreen = e.isIntersecting;
        onScreen ? start() : stop();
      },
      { threshold: 0 },
    );
    io.observe(host);

    const onVisibility = () => {
      visible = document.visibilityState === "visible";
      visible ? start() : stop();
    };
    document.addEventListener("visibilitychange", onVisibility);

    // A lost context (GPU reset, tab backgrounded for a long time) would
    // otherwise leave a black rectangle behind the headline forever.
    const onLost = (e: Event) => {
      e.preventDefault();
      stop();
      setFailed(true);
    };
    canvas.addEventListener("webglcontextlost", onLost);

    resize();
    if (reduced) draw();        // reduced motion: exactly one frame, no loop
    else start();

    // ---- teardown -------------------------------------------------------
    return () => {
      stop();
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("webglcontextlost", onLost);
      gl.deleteProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteVertexArray(vao);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
    }

    return () => {
      disposed = true;
      (window.cancelIdleCallback ?? window.clearTimeout)(idle);
      teardown?.();
    };
  }, []);

  return (
    <div
      ref={hostRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 -z-10 overflow-hidden"
      // The layer is clipped by the hero section, which put a hard horizontal
      // edge wherever the field was still bright at the boundary. Masking the
      // WHOLE wrapper -- field and both scrims together -- dissolves the
      // composite to transparent before it reaches that edge, so what shows
      // through is the page background it was already sitting on. Done in CSS
      // rather than tied to scroll: a spatial fade has no listener to run,
      // cannot desync from Lenis, and costs nothing per frame. The extra stops
      // approximate an ease-out; a two-stop linear ramp leaves a visible seam
      // where the gradient starts.
      style={{ WebkitMaskImage: FADE_OUT, maskImage: FADE_OUT }}
    >
      {failed ? (
        // No WebGL (or context lost): the hero must never break.
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(90deg, rgb(160,224,171), rgb(255,172,46) 50%, rgb(165,45,37))",
            opacity: 0.28,
          }}
        />
      ) : (
        <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full" />
      )}
      {/* Scrim. Deliberately LEFT-WEIGHTED rather than a symmetric vignette: all
          hero copy lives in the left ~45% of the section, so darkening that band
          hard buys the right-hand side the headroom to reach its bright specular
          cores. A uniform vignette would have to crush the whole field to protect
          the text, which is exactly what flattened the earlier version. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(100deg, rgba(10,10,11,0.90) 0%, rgba(10,10,11,0.78) 26%," +
            " rgba(10,10,11,0.64) 52%, rgba(10,10,11,0.46) 78%, rgba(10,10,11,0.42) 100%)",
        }}
      />
      {/* The nav row and the footer hairline row run the full width, so the
          left-weighted scrim above cannot protect their right-hand labels.
          These two bands darken just those strips and leave the middle to blaze. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(10,10,11,0.86) 0%, rgba(10,10,11,0.34) 11%," +
            " rgba(10,10,11,0) 26%, rgba(10,10,11,0) 70%," +
            " rgba(10,10,11,0.42) 87%, rgba(10,10,11,0.84) 100%)",
        }}
      />
    </div>
  );
}
