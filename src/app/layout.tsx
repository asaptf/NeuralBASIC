import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";

/**
 * Mirrors `basePath` in next.config.ts. Read at build time, so a static export
 * carries whichever prefix it was built with. Kept in sync by hand because the
 * config's value isn't importable here without pulling the config into the
 * client bundle.
 */
const BASE_PATH = process.env.BASE_PATH?.replace(/\/$/, "") || "";

/**
 * One type system, two families. IBM Plex pairs a humanist sans with a mono
 * that shares its skeleton, which suits an IDE whose chrome and prose sit
 * inches apart — and its lineage is the same computing era the Retro Blue
 * theme is quoting. Plex Mono's slashed zero also matters in a panel full of
 * losses and learning rates.
 */
const plexSans = IBM_Plex_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-mono-family",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "NeuralBASIC — Educational Neural Network IDE",
  description:
    "Immediate-mode educational IDE for learning neural networks. Socratic tutor, live visualization, curriculum. Spiritual successor to QuickBASIC & TabletBasic.",
  /**
   * Declared explicitly rather than left to the `app/favicon.ico` convention.
   * That convention serves the file from the domain root and drops its <link>
   * entirely once `basePath` is set — so on a project Pages site the browser
   * would fall back to probing `<user>.github.io/favicon.ico`, which belongs to
   * whatever else lives at that root. Prefixing it here keeps the icon pointing
   * at this app on both a root deploy and a subpath one.
   */
  icons: {
    icon: [
      // SVG first: browsers that support it get a mark that stays sharp at any
      // size. The .ico is the fallback for those that don't.
      { url: `${BASE_PATH}/icon.svg`, type: "image/svg+xml", sizes: "any" },
      { url: `${BASE_PATH}/favicon.ico`, type: "image/x-icon", sizes: "16x16 32x32 48x48" },
    ],
    apple: { url: `${BASE_PATH}/apple-icon.png`, sizes: "180x180", type: "image/png" },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // The font variables must live on <html>, not <body>: the theme tokens in
  // globals.css are declared on html/:root and reference them with var(). A
  // var() that can't resolve makes the whole custom property invalid, which
  // silently collapsed every --font-ui/--font-mono to the default serif.
  return (
    <html
      lang="en"
      data-theme="modern"
      className={`${plexSans.variable} ${plexMono.variable}`}
      suppressHydrationWarning
    >
      <body className="antialiased">{children}</body>
    </html>
  );
}
