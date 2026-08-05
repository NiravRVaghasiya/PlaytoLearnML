import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // GameML has no ML backend by design (CLAUDE.md: "Never add a server-side ML
  // inference endpoint"). Everything below exists only to make the heavy
  // client-side ML/graphics libraries bundle cleanly.
  experimental: {
    // Keep TF.js / Three.js / Phaser out of the server graph where possible.
    optimizePackageImports: ["lucide-react", "d3"],
  },
};

export default nextConfig;
