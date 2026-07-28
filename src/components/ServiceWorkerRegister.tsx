"use client";

import { useEffect } from "react";

/**
 * Registers the production service worker once on the client.
 *
 * Only runs in production builds: in dev a worker fights HMR and surfaces
 * confusing stale-module behaviour. Failures are logged and never thrown —
 * a broken worker must not break the reader experience.
 *
 * `basePath` is passed from the server layout so it matches the same
 * `BASE_PATH` used by the manifest and icon metadata (custom env vars are not
 * automatically available in client components).
 */
export function ServiceWorkerRegister({ basePath }: { basePath: string }) {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") return;
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    const prefix = basePath.replace(/\/$/, "");
    navigator.serviceWorker
      .register(`${prefix}/sw.js`, { scope: `${prefix}/` })
      .catch((err) => {
        console.warn("[sw] registration failed:", err);
      });
  }, [basePath]);

  return null;
}
