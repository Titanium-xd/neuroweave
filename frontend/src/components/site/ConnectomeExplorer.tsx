import { useMemo, useRef, useState } from "react";
import { buildGraph, REGION_LABELS, type Neuron } from "@/data/connectome";

const W = 1000;
const H = 620;

export function ConnectomeExplorer() {
  const graph = useMemo(() => buildGraph(150, 20250915), []);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const [hover, setHover] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const pos = (n: Neuron) => ({ x: n.x * W, y: n.y * H });

  const focus = selected ?? hover;
  const incident = useMemo(() => {
    if (focus === null) return null;
    return new Set(
      graph.edges.filter((e) => e.source === focus || e.target === focus).map((e) => e.id),
    );
  }, [focus, graph.edges]);

  const signalEdges = useMemo(() => graph.edges.filter((_, i) => i % 17 === 0).slice(0, 14), [graph.edges]);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const rect = svgRef.current!.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * W;
    const my = ((e.clientY - rect.top) / rect.height) * H;
    setView((v) => {
      const k = Math.min(6, Math.max(0.7, v.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      const scale = k / v.k;
      return { k, x: mx - (mx - v.x) * scale, y: my - (my - v.y) * scale };
    });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const rect = svgRef.current!.getBoundingClientRect();
    const dx = ((e.clientX - drag.current.x) / rect.width) * W;
    const dy = ((e.clientY - drag.current.y) / rect.height) * H;
    setView((v) => ({ ...v, x: drag.current!.vx + dx, y: drag.current!.vy + dy }));
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const detail = focus !== null ? (graph.nodes[focus] ?? null) : null;
  const detailEdges =
    focus === null
      ? []
      : graph.edges.filter((e) => e.source === focus || e.target === focus).slice(0, 6);

  return (
    <div className="grid gap-0 border border-border lg:grid-cols-[minmax(0,1fr)_300px]">
      <div className="relative border-b border-border lg:border-b-0 lg:border-r">
        <div className="flex items-center justify-between gap-4 border-b border-border px-4 py-2.5">
          <span className="label-tech">
            MaleCNS v1.0-derived prototype subgraph
            <span className="ml-2 opacity-60">&mdash; illustrative layout, not anatomical</span>
          </span>
          <div className="flex items-center gap-1">
            {[
              { l: "−", f: () => setView((v) => ({ ...v, k: Math.max(0.7, v.k / 1.25) })) },
              { l: "+", f: () => setView((v) => ({ ...v, k: Math.min(6, v.k * 1.25) })) },
              { l: "reset", f: () => setView({ x: 0, y: 0, k: 1 }) },
            ].map((b) => (
              <button
                key={b.l}
                type="button"
                onClick={b.f}
                className="label-tech border border-border px-2 py-1 transition-colors duration-200 hover:border-border-strong hover:text-foreground"
              >
                {b.l}
              </button>
            ))}
          </div>
        </div>

        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-none select-none"
          style={{ cursor: drag.current ? "grabbing" : "grab" }}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
        >
          <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
            {/* edges */}
            {graph.edges.map((e) => {
              const a = pos(graph.nodes[e.source]!);
              const b = pos(graph.nodes[e.target]!);
              const lit = incident?.has(e.id);
              return (
                <line
                  key={e.id}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={lit ? "var(--bio)" : "var(--border-strong)"}
                  strokeWidth={lit ? 1.1 : 0.5}
                  opacity={incident ? (lit ? 0.9 : 0.16) : 0.42}
                  vectorEffect="non-scaling-stroke"
                  style={{ transition: "opacity 200ms linear, stroke 200ms linear" }}
                />
              );
            })}

            {/* travelling signals */}
            {signalEdges.map((e, i) => {
              const a = pos(graph.nodes[e.source]!);
              const b = pos(graph.nodes[e.target]!);
              const len = Math.hypot(b.x - a.x, b.y - a.y);
              return (
                <line
                  key={`sig-${e.id}`}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke="var(--bio)"
                  strokeWidth={1.4}
                  vectorEffect="non-scaling-stroke"
                  className="abb-signal"
                  strokeDasharray={`${Math.max(10, len * 0.12)} ${len}`}
                  style={
                    {
                      "--abb-len": `${len * 1.12}`,
                      animationDelay: `${i * 0.42}s`,
                    } as React.CSSProperties
                  }
                />
              );
            })}

            {/* nodes */}
            {graph.nodes.map((n) => {
              const p = pos(n);
              const isFocus = focus === n.id;
              const dim = incident !== null && !isFocus;
              return (
                <g key={n.id}>
                  {n.stimulated && (
                    <rect
                      x={p.x - 5.5}
                      y={p.y - 5.5}
                      width={11}
                      height={11}
                      fill="none"
                      stroke="var(--caveat)"
                      strokeWidth={0.7}
                      opacity={dim ? 0.25 : 0.75}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={isFocus ? 5 : n.active ? 3 : 2.4}
                    fill={
                      n.stimulated
                        ? "var(--caveat)"
                        : n.active
                          ? "var(--bio)"
                          : "var(--muted-foreground)"
                    }
                    opacity={dim ? 0.3 : n.active || n.stimulated ? 0.95 : 0.5}
                    style={{ transition: "r 180ms ease, opacity 180ms linear" }}
                  />
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={9}
                    fill="transparent"
                    className="cursor-pointer"
                    onPointerEnter={() => setHover(n.id)}
                    onPointerLeave={() => setHover((h) => (h === n.id ? null : h))}
                    onClick={() => setSelected(selected === n.id ? null : n.id)}
                  />
                </g>
              );
            })}
          </g>
        </svg>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border px-4 py-3">
          {[
            ["var(--bio)", "Active neuron"],
            ["var(--muted-foreground)", "Inactive neuron"],
            ["var(--caveat)", "Stimulated (marked)"],
          ].map(([c, l]) => (
            <span key={l} className="flex items-center gap-2">
              <span className="inline-block h-[7px] w-[7px] rounded-full" style={{ background: c }} />
              <span className="label-tech">{l}</span>
            </span>
          ))}
        </div>
      </div>

      {/* metadata panel */}
      <aside className="flex flex-col">
        <div className="border-b border-border px-4 py-2.5">
          <span className="label-tech">Graph statistics</span>
          <span className="ml-2 font-mono text-[10px] text-muted-foreground">real data</span>
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-4 border-b border-border px-4 py-5">
          {[
            ["Nodes", graph.stats.nodes.toString()],
            ["Directed edges", graph.stats.edges.toString()],
            ["Density", graph.stats.density.toFixed(4)],
            ["Mean degree", graph.stats.meanDegree.toFixed(2)],
            ["Active", graph.stats.active.toString()],
            ["Stimulated", graph.stats.stimulated.toString()],
          ].map(([k, v]) => (
            <div key={k}>
              <dt className="label-tech">{k}</dt>
              <dd className="tnum mt-1.5 font-mono text-[13px]">{v}</dd>
            </div>
          ))}
        </dl>

        <div className="border-b border-border px-4 py-2.5">
          <span className="label-tech">{detail ? "Neuron" : "No selection"}</span>
        </div>
        <div className="px-4 py-5">
          {detail ? (
            <div className="space-y-5">
              <div>
                <div className="tnum font-mono text-lg">{detail.name}</div>
                <div className="mt-1 text-sm text-muted-foreground">
                  {REGION_LABELS[detail.region]}
                </div>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-4">
                {[
                  ["In-degree", detail.inDegree.toString()],
                  ["Out-degree", detail.outDegree.toString()],
                  ["State", detail.active ? "active" : "inactive"],
                  ["Stimulated", detail.stimulated ? "yes" : "no"],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="label-tech">{k}</dt>
                    <dd className="tnum mt-1.5 font-mono text-[13px]">{v}</dd>
                  </div>
                ))}
              </dl>
              <div>
                <div className="label-tech">Incident edges</div>
                <ul className="mt-2 space-y-1">
                  {detailEdges.map((e) => (
                    <li key={e.id} className="tnum font-mono text-[12px] text-muted-foreground">
                      {graph.nodes[e.source]!.name} → {graph.nodes[e.target]!.name}
                      <span className="ml-2 opacity-60">w {e.weight.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <p className="text-sm leading-relaxed text-muted-foreground">
              Hover a neuron to highlight its incident edges. Click to pin it. Scroll to zoom, drag
              to pan.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}
