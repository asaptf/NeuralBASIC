/**
 * The app mark: the same sigmoid as the favicon (`public/icon.svg`, with its
 * raster siblings `favicon.ico` and `apple-icon.png` beside it), but drawn from
 * theme tokens so Retro Blue gets its own colours instead of a hardcoded mint
 * tile. The path below is duplicated in those files — keep all of them in visual
 * sync if any changes.
 */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      className="logo-mark-svg"
      aria-hidden="true"
      focusable="false"
    >
      <rect width="32" height="32" rx="7" className="logo-tile" />
      <path
        d="M5 25C11 25 12.5 22.5 14.5 16.5C16.5 10.5 19.5 7 27 7"
        strokeWidth="4.6"
        strokeLinecap="round"
        fill="none"
        className="logo-curve"
      />
    </svg>
  );
}
