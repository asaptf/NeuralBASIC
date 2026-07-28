"use client";

import { useEffect, useRef, useState } from "react";

/**
 * True for `ms` after `value` flips from false to true — but never on the first
 * render.
 *
 * That exclusion is the whole point. Progress is restored from localStorage, so
 * a plain `value && animate` would replay every past win on every page load,
 * celebrating things the learner did yesterday. This only fires for changes that
 * happen while they're watching.
 */
export function useJustBecameTrue(value: boolean, ms = 1000): boolean {
  const [active, setActive] = useState(false);
  const previous = useRef<boolean | null>(null);

  useEffect(() => {
    const wasSeen = previous.current !== null;
    const rose = previous.current === false && value === true;
    previous.current = value;

    if (!wasSeen || !rose) return;

    setActive(true);
    const t = setTimeout(() => setActive(false), ms);
    return () => clearTimeout(t);
  }, [value, ms]);

  return active;
}
