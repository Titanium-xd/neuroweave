import { useEffect, useState } from "react";

/**
 * Counts up to `to` once on mount with an ease-out cubic animation.
 * Static (no animation) under prefers-reduced-motion.
 *
 * Previously used a `started` ref guard that broke under React Strict Mode:
 * - First effect run: setValue(0), starts RAF
 * - Cleanup: cancels RAF (value stuck at 0)
 * - Second effect run: guard bails out → value stays 0 permanently
 *
 * Fix: remove the guard. The [to, duration] dep array already prevents
 * spurious re-runs. Strict Mode's double-invocation now correctly restarts
 * the animation on the second run.
 */
export function CountUp({
  to,
  decimals = 1,
  duration = 900,
  className,
}: {
  to: number;
  decimals?: number;
  duration?: number;
  className?: string;
}) {
  const [value, setValue] = useState(to);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    setValue(0);
    const start = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(to * eased);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [to, duration]);

  return <span className={className}>{value.toFixed(decimals)}</span>;
}
