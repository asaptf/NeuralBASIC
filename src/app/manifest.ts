import type { MetadataRoute } from "next";

/**
 * Web app manifest, generated rather than a static file in `public/`.
 *
 * It has to be generated because every path inside it — `start_url`, `scope`,
 * and each icon `src` — needs the deployment's base path, and a static JSON file
 * cannot interpolate one. On a project GitHub Pages site the app lives under
 * `/NeuralBASIC/`, and a manifest claiming `start_url: "/"` would install a
 * shortcut to the root of `<user>.github.io` — somebody else's site.
 */
const BASE_PATH = process.env.BASE_PATH?.replace(/\/$/, "") || "";

/**
 * Required: a generated manifest is a route, and `output: "export"` refuses to
 * build one that hasn't declared itself static. Without this the whole build
 * fails rather than degrading — which is the right behaviour, but only if you
 * find out at build time.
 */
export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "NeuralBASIC — Educational Neural Network IDE",
    // Home screens truncate at roughly 12 characters, so this is the name that
    // actually gets read.
    short_name: "NeuralBASIC",
    description:
      "Build neural networks and watch them learn. Five chapters, live decision boundaries, and a curriculum that unlocks when you can explain what you saw.",
    start_url: `${BASE_PATH}/`,
    scope: `${BASE_PATH}/`,
    display: "standalone",
    // Matches the top of the header gradient, so the Android status bar sits
    // flush against the app chrome instead of drawing a seam above it.
    theme_color: "#101a2a",
    // The app background, so the launch screen doesn't flash a colour the app
    // never uses.
    background_color: "#070b12",
    // Deliberately no `orientation`: this is a tablet-first IDE that works in
    // both, and pinning one would only take a choice away from the reader.
    icons: [
      {
        src: `${BASE_PATH}/icon-192.png`,
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: `${BASE_PATH}/icon-512.png`,
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
      {
        // Separate artwork, not the same file relabelled: Android crops maskable
        // icons to a circle of 80% diameter, and the mark at full size runs
        // almost edge to edge, so its ends would be sliced off. This variant is
        // full-bleed with the curve scaled into the safe zone — measured at
        // 185px from centre against a 205px safe radius.
        src: `${BASE_PATH}/icon-maskable-512.png`,
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
