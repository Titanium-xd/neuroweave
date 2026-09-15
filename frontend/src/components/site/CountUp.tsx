import { useEffect, useRef, useState } from "react";

/** Counts up once on mount. Static under prefers-reduced-motion. */
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
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
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
