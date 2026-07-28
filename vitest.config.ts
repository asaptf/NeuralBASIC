import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.spec.ts"],
    // Attention / conv multi-run aggregates train by finite differences and are
    // intentionally heavy — do not shrink run counts to "fix" timeouts.
    // Sized for a CI runner ~4× slower than a typical laptop, with headroom so
    // a slightly slower host does not flake. Per-test overrides below this
    // value are a trap (they reintroduce the old ceiling); only raise above it.
    //
    // Measured: the slowest single test is ~43s locally, and the whole
    // negation file took 289s on a GitHub runner. At 300s that left the slowest
    // test only ~1.7× headroom — too thin for a limit that gates the Pages
    // deploy. A timeout exists to catch a hang, not to police slowness, so it
    // costs nothing to be generous: a genuine hang still trips 600s, while a
    // merely slow runner no longer blocks a deploy.
    testTimeout: 600_000,
    hookTimeout: 60_000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
