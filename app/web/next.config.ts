import type { NextConfig } from "next";

const API = process.env.CLARION_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // The FastAPI backend runs separately (uvicorn on :8000). Proxying keeps the
  // browser same-origin, so SSE needs no CORS preflight and no absolute URLs
  // leak into the client bundle.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
