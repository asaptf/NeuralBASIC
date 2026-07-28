import type { NextConfig } from "next";

/**
 * GitHub Pages serves project sites under a repo subpath
 * (https://<user>.github.io/<repo>/). Set BASE_PATH to that subpath
 * (e.g. `/NeuralBASIC`) only for the Pages production build.
 *
 * Leave unset for local `npm run dev` and a plain `npm run build` so both
 * stay rooted at `/`. The deploy workflow sets BASE_PATH; CI builds do not.
 */
const basePath = process.env.BASE_PATH?.replace(/\/$/, "") || "";

const nextConfig: NextConfig = {
  output: "export",
  // Static export cannot optimize images at request time.
  images: { unoptimized: true },
  ...(basePath
    ? {
        basePath,
        // Keep asset URLs under the same subpath Pages serves.
        assetPrefix: basePath,
      }
    : {}),
};

export default nextConfig;
