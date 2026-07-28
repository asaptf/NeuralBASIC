/**
 * Service worker for NeuralBASIC.
 *
 * Caching policy is deliberate: staleness is worse than a cache miss. Lessons
 * quote measured figures that must match the running engine, so a deploy must
 * land on the next load — not after some cache expiry.
 *
 * Routing:
 *   - navigation / HTML  → network-first (offline shell fallback)
 *   - /_next/static/**   → cache-first (content-hashed, immutable)
 *   - icons + manifest   → stale-while-revalidate
 *   - cross-origin       → not intercepted (Monaco / fonts stay on the network)
 *
 * Install precaches:
 *   1. the stable app shell (icons, manifest, …)
 *   2. every same-origin `/_next/static/**` URL referenced by the shell HTML
 *   3. one hop of dynamic chunks those JS files reference (e.g. next/dynamic
 *      Monaco wrapper as `static/chunks/….js`) so a failed chunk load does not
 *      tear down the hydrated tree offline
 *
 * Without (2), the first visit never sees CSS/JS (the worker is still
 * installing while the page loads them from the network), so a cold offline
 * start after one online visit serves unstyled HTML.
 *
 * Monaco itself still comes from a CDN and stays offline-unavailable — that is
 * expected. Precaching only keeps its same-origin wrapper from crashing React.
 *
 * BASE_PATH is derived from this script's own URL so the same file works at
 * `/sw.js` and at `/NeuralBASIC/sw.js` without a build step.
 */

const CACHE_VERSION = "v2";
const CACHE_NAME = `neuralbasic-${CACHE_VERSION}`;

const BASE_PATH = self.location.pathname.replace(/\/sw\.js$/, "");

/** App shell + stable static assets for a cold offline start. */
const PRECACHE_URLS = [
  `${BASE_PATH}/`,
  `${BASE_PATH}/index.html`,
  `${BASE_PATH}/manifest.webmanifest`,
  `${BASE_PATH}/icon-192.png`,
  `${BASE_PATH}/icon-512.png`,
  `${BASE_PATH}/icon-maskable-512.png`,
  `${BASE_PATH}/icon.svg`,
  `${BASE_PATH}/favicon.ico`,
  `${BASE_PATH}/apple-icon.png`,
];

/**
 * Pull same-origin `/_next/static/**` paths out of text (HTML or JS).
 *
 * DOMParser is not available in service workers; a regex over the text is
 * intentional here, not an oversight. The static export shell lists every CSS,
 * JS chunk, and font the cold start needs in href/src attributes (and again in
 * the RSC flight payload as JSON-escaped strings). Matching path-absolute
 * `/…/_next/static/…` tokens covers both.
 *
 * @param {string} text
 * @returns {string[]}
 */
function extractNextStaticUrls(text) {
  const found = new Set();
  // Path-absolute URLs only. Stop before quotes, backslashes (JSON escapes in
  // the RSC payload), whitespace, and tag delimiters.
  const re = /\/(?:[^"'\\\s<>/]+\/)*_next\/static\/[^"'\\\s<>]+/g;
  let match;
  while ((match = re.exec(text)) !== null) {
    found.add(match[0]);
  }
  return [...found];
}

/**
 * Turbopack/webpack dynamic imports often look like `"static/chunks/abc.js"`
 * (relative to `/_next/`), not a path-absolute `/_next/static/…` URL. Resolve
 * those against BASE_PATH so next/dynamic chunks are also precached.
 *
 * @param {string} jsText
 * @returns {string[]}
 */
function extractRelativeNextStaticUrls(jsText) {
  const found = new Set();
  const re = /["']static\/(?:chunks|media|css)\/[^"']+["']/g;
  let match;
  while ((match = re.exec(jsText)) !== null) {
    const rel = match[0].slice(1, -1);
    found.add(`${BASE_PATH}/_next/${rel}`);
  }
  return [...found];
}

/**
 * Fetch one URL with cache:reload and store it. Returns the body text for JS
 * so callers can mine dynamic-import paths. Failures are swallowed so a single
 * miss cannot abort install.
 *
 * @param {Cache} cache
 * @param {string} url
 * @returns {Promise<string|null>}
 */
async function precacheOne(cache, url) {
  try {
    const response = await fetch(url, { cache: "reload" });
    if (!response.ok) return null;
    // Clone before reading so cache.put still has a body stream.
    const forCache = response.clone();
    const text = url.endsWith(".js") || url.endsWith(".css") || url.endsWith(".html")
      ? await response.text()
      : null;
    await cache.put(url, forCache);
    return text;
  } catch {
    // Individual precache misses must not fail install.
    return null;
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);

      // Shell first so we can parse it for content-hashed build assets.
      // Cache under both `/` and `/index.html` — navigations may use either.
      const shellUrl = `${BASE_PATH}/`;
      const indexUrl = `${BASE_PATH}/index.html`;
      let shellHtml = "";
      try {
        const shellResponse = await fetch(shellUrl, { cache: "reload" });
        if (shellResponse.ok) {
          // Clone before reading the body so cache.put still has a stream.
          // Two clones: cache.put consumes the body of each Response it is given.
          const forSlash = shellResponse.clone();
          const forIndex = shellResponse.clone();
          shellHtml = await shellResponse.text();
          await cache.put(shellUrl, forSlash);
          await cache.put(indexUrl, forIndex);
        }
      } catch {
        // Shell miss — still attempt the rest of PRECACHE_URLS below.
      }

      const shellStaticUrls = shellHtml ? extractNextStaticUrls(shellHtml) : [];

      // Remaining stable shell assets (icons, manifest, …). Skip shell URLs we
      // already stored above so we do not double-fetch.
      const already = new Set([shellUrl, indexUrl]);
      const stableUrls = PRECACHE_URLS.filter((url) => !already.has(url));

      // Precache shell static assets; keep JS bodies to discover dynamic chunks.
      const jsBodies = [];
      await Promise.all(
        [...stableUrls, ...shellStaticUrls].map(async (url) => {
          already.add(url);
          const body = await precacheOne(cache, url);
          if (body != null && url.endsWith(".js")) jsBodies.push(body);
        }),
      );

      // One hop: dynamic import targets referenced from shell JS (next/dynamic).
      const dynamicUrls = new Set();
      for (const body of jsBodies) {
        for (const u of extractNextStaticUrls(body)) dynamicUrls.add(u);
        for (const u of extractRelativeNextStaticUrls(body)) dynamicUrls.add(u);
      }
      await Promise.all(
        [...dynamicUrls]
          .filter((url) => !already.has(url))
          .map((url) => precacheOne(cache, url)),
      );

      // Take over on the next load rather than waiting for every tab to close.
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key)),
      );
      await self.clients.claim();
    })(),
  );
});

/**
 * @param {Request} request
 * @param {Response} response
 */
async function putInCache(request, response) {
  if (!response || !response.ok) return;
  // Only cache basic (same-origin) responses — never opaque cross-origin ones.
  if (response.type !== "basic" && response.type !== "default") return;
  try {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(request, response.clone());
  } catch {
    // Quota or abort — ignore.
  }
}

/** Resolve the offline app shell from either slash or index.html keys. */
async function matchShell() {
  return (
    (await caches.match(`${BASE_PATH}/`)) ||
    (await caches.match(`${BASE_PATH}/index.html`))
  );
}

/**
 * Network first; on failure serve cache (navigation falls back to shell).
 *
 * Freshness stays intact for OK responses: a successful network hit is what
 * gets cached and returned. Non-OK / thrown fetches fall through to cache so
 * an offline cold start still gets the shell rather than a bare error page.
 */
async function networkFirst(request, { shellFallback = false } = {}) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      await putInCache(request, response);
      return response;
    }
    // Non-OK network reply: prefer a cached copy (and shell for navigations).
    const cached = await caches.match(request);
    if (cached) return cached;
    if (shellFallback) {
      const shell = await matchShell();
      if (shell) return shell;
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    if (shellFallback) {
      const shell = await matchShell();
      if (shell) return shell;
    }
    // Re-throw so the browser surfaces a network error rather than an empty reply.
    throw new Error("Network failed and no cache match");
  }
}

/** Cache first; fetch and store on miss. */
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      await putInCache(request, response);
    }
    return response;
  } catch (err) {
    // Offline miss: surface the failure rather than an empty reply.
    throw err;
  }
}

/**
 * Stale-while-revalidate: return cache immediately if present, refresh in
 * background. Icons and the manifest change rarely and are not versioned by
 * content hash, so SWR keeps them snappy without pinning a forever-stale copy.
 */
async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  const networkPromise = fetch(request)
    .then(async (response) => {
      if (response.ok) {
        await cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => undefined);

  if (cached) {
    // Kick off revalidation; don't block the response on it.
    void networkPromise;
    return cached;
  }

  const networkResponse = await networkPromise;
  if (networkResponse) return networkResponse;
  throw new Error("Network failed and no cache match");
}

/**
 * @param {URL} url
 */
function isNextStatic(url) {
  return url.pathname.includes("/_next/static/");
}

/**
 * Icons, favicon, apple-touch, and the web app manifest.
 * @param {URL} url
 */
function isIconOrManifest(url) {
  const path = url.pathname;
  if (path.endsWith("/manifest.webmanifest")) return true;
  if (path.endsWith("/favicon.ico")) return true;
  if (path.endsWith("/apple-icon.png")) return true;
  if (path.endsWith("/icon.svg")) return true;
  if (/\/icon(?:-maskable)?-\d+\.png$/.test(path)) return true;
  return false;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only handle GET. Leave POST/etc alone.
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Cross-origin (Monaco on jsDelivr, fonts, etc.): do not intercept.
  // Opaque responses cannot be inspected; caching them risks poisoning the
  // cache with an error that then persists.
  if (url.origin !== self.location.origin) return;

  // Navigations and HTML documents: network-first so a deploy is visible on
  // the next load. Offline → cached shell.
  const acceptsHtml = request.headers.get("accept")?.includes("text/html");
  if (request.mode === "navigate" || acceptsHtml) {
    event.respondWith(networkFirst(request, { shellFallback: true }));
    return;
  }

  // Content-hashed build assets: safe to cache forever.
  if (isNextStatic(url)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Icons + manifest: rarely change, not content-hashed → SWR.
  if (isIconOrManifest(url)) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  // Anything else same-origin: network-first, no shell fallback.
  // Prefer freshness over a quiet cache miss for unknown assets.
  event.respondWith(networkFirst(request));
});
