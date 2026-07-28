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
    testTimeout: 300_000,
    hookTimeout: 60_000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
