import { useEffect, useRef } from "react";
import { buildGraph, type Graph } from "@/data/connectome";

interface Pulse {
  edge: number;
  t: number;
  speed: number;
}

/**
 * Hero connectome field: sparse directed graph with occasional activity
 * pulses travelling along edges. Canvas, slow, no constant motion on nodes.
 */
export function ConnectomeField({ className = "" }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const graphRef = useRef<Graph>(null as unknown as Graph);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!graphRef.current) graphRef.current = buildGraph(150, 20250915);
    const graph = graphRef.current;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let raf = 0;
    const pulses: Pulse[] = [];

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const px = (n: { x: number; y: number }) => ({ x: n.x * width, y: n.y * height });

    const draw = (time: number) => {
      ctx.clearRect(0, 0, width, height);

      // edges
      for (const e of graph.edges) {
        const a = px(graph.nodes[e.source]!);
        const b = px(graph.nodes[e.target]!);
        ctx.strokeStyle = `rgba(190, 186, 175, ${0.045 + e.weight * 0.05})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      // pulses
      if (!reduced) {
        if (pulses.length < 26 && Math.random() < 0.09) {
          pulses.push({
            edge: Math.floor(Math.random() * graph.edges.length),
            t: 0,
            speed: 0.0022 + Math.random() * 0.0035,
          });
        }
        for (let i = pulses.length - 1; i >= 0; i--) {
          const p = pulses[i]!;
          p.t += p.speed * 16;
          if (p.t >= 1) {
            pulses.splice(i, 1);
            continue;
          }
          const e = graph.edges[p.edge]!;
          const a = px(graph.nodes[e.source]!);
          const b = px(graph.nodes[e.target]!);
          const fade = Math.sin(p.t * Math.PI);
          const cx = a.x + (b.x - a.x) * p.t;
          const cy = a.y + (b.y - a.y) * p.t;

          const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
          grad.addColorStop(Math.max(0, p.t - 0.12), "rgba(126, 176, 130, 0)");
          grad.addColorStop(p.t, `rgba(146, 194, 148, ${0.5 * fade})`);
          grad.addColorStop(Math.min(1, p.t + 0.001), "rgba(146, 194, 148, 0)");
          ctx.strokeStyle = grad;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();

          ctx.fillStyle = `rgba(168, 205, 168, ${0.7 * fade})`;
          ctx.beginPath();
          ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // nodes
      for (const n of graph.nodes) {
        const p = px(n);
        const breathe = reduced ? 0 : Math.sin(time / 2600 + n.id) * 0.07;
        const r = n.stimulated ? 2.6 : 1.5 + n.inDegree * 0.06;
        ctx.fillStyle = n.stimulated
          ? `rgba(160, 202, 162, ${0.85 + breathe})`
          : n.active
            ? `rgba(214, 210, 198, ${0.5 + breathe})`
            : "rgba(150, 147, 139, 0.28)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };

    resize();
    raf = requestAnimationFrame(draw);
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />;
}
