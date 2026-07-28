import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";

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
