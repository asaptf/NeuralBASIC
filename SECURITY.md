# Security Policy

## The short version of the threat model

NeuralBASIC has no backend. There is no server, no account, no database, no analytics, and no
telemetry. Every network is built, trained and evaluated in your browser by the TypeScript engine in
`src/engine`, and nothing you type or train is transmitted anywhere.

What that means in practice:

- **Your work stays local.** Save/Load uses `localStorage` under the `neuralbasic:*` keys. Export
  writes a file to your machine; Import reads one you choose. That is the whole data path.
- **The hosted build is a static bundle.** The GitHub Pages deployment is prerendered HTML, CSS, JS
  and fonts. It cannot receive data because there is nothing behind it to receive it.
- **There is nothing to leak.** No credentials, no tokens, no personal data are collected or stored,
  so a compromise of the site could not disclose any.

That reduces the realistic attack surface to two things: the dependency tree, and the one runtime
resource fetched from outside the bundle.

## The one third-party runtime dependency

`@monaco-editor/react` loads the Monaco editor from the jsDelivr CDN on first use. So the DSL editor
depends on a third party being reachable and serving what it claims to serve. Everything else in the
app is bundled from this repository.

Two consequences worth knowing:

- With the CDN blocked or offline, the editor does not load. The rest of the app — training,
  visualisation, lessons, tutor — keeps working.
- A compromise of that CDN would mean untrusted script in the page. If you need to eliminate that
  dependency, self-host Monaco by configuring `@monaco-editor/react`'s loader to point at a local
  copy; nothing else in the codebase assumes the CDN.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private reporting on this repository — *Security → Report a vulnerability* — which opens
a confidential advisory visible only to the maintainer. If that is unavailable to you, contact the
maintainer through their GitHub profile at [@asaptf](https://github.com/asaptf) and ask for a private
channel before sending details.

Please include what you found, how to reproduce it, and what an attacker could actually achieve. A
proof of concept is welcome.

Because this is a single-maintainer educational project, there is no formal response-time commitment.
You will get an acknowledgement as promptly as is reasonable, and credit in the fix unless you would
rather not be named.

## Supported versions

The `main` branch is the only supported version. Fixes land there and reach the hosted build on the
next deploy; there are no maintenance branches or backports.
