import type { Metadata } from "next";
import { Geist, Geist_Mono, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// The Retro Blue theme asks for IBM Plex Mono; without loading it the theme was
// silently falling back to Courier New.
const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "600"],
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
      className={`${geistSans.variable} ${geistMono.variable} ${plexMono.variable}`}
      suppressHydrationWarning
    >
      <body className="antialiased">{children}</body>
    </html>
  );
}
