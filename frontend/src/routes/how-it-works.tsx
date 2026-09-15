import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { T001, T002 } from "@/data/benchmark";

export const Route = createFileRoute("/how-it-works")({
  head: () => ({
    meta: [
      { title: "How It Works | ABB — Animal Brain Benchmark" },
      {
        name: "description",
        content:
          "A visual story explaining the Animal Brain Benchmark — from real connectome data to a controlled AI comparison. No background required.",
      },
    ],
  }),
  component: HowItWorks,
});

// ── Scroll-reveal hook ───────────────────────────────────────────────────────
function useInView(threshold = 0.12) {
  const ref = useRef<HTMLElement | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    // SSR / environments without IntersectionObserver: show immediately
    if (typeof IntersectionObserver === "undefined") { setInView(true); return; }
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setInView(true); obs.disconnect(); } },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return { ref, inView };
}

// ── Chapter shell ─────────────────────────────────────────────────────────────
function Chapter({ n, children }: { n: string; children: ReactNode }) {
  const { ref, inView } = useInView();
  return (
    <section
      ref={ref as React.RefObject<HTMLElement>}
      className="rule-t py-16 md:py-24"
      style={{
        opacity:    inView ? 1 : 0,
        transform:  inView ? "none" : "translateY(16px)",
        transition: "opacity 440ms ease, transform 440ms ease",
      }}
    >
      <div className="grid gap-10 md:grid-cols-[180px_minmax(0,1fr)] md:gap-16">
        <aside>
          <div className="label-tech text-muted-foreground md:sticky md:top-24">
            Chapter {n}
          </div>
        </aside>
        <div className="min-w-0">{children}</div>
      </div>
    </section>
  );
}

function H({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-3xl font-light leading-tight tracking-[-0.03em] md:text-[2.6rem]">
      {children}
    </h2>
  );
}
function P({ children }: { children: ReactNode }) {
  return (
    <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
      {children}
    </p>
  );
}

// ── SVG diagrams ─────────────────────────────────────────────────────────────

/** Chapter 01 — animated neuron ring */
function NeuronRing() {
  const nodes: [number, number][] = [
    [100, 18], [162, 50], [180, 128], [142, 192], [58, 192], [18, 128], [36, 50],
  ];
  const edges: [number, number][] = [
    [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,0],[1,3],[0,4],[2,5],
  ];
  return (
    <svg
      viewBox="0 0 200 210"
      className="w-full max-w-[200px] shrink-0"
      role="img"
      aria-label="Illustrative neuron connectivity — positions are schematic, not anatomical"
    >
      <defs>
        <marker id="hiw-arr" markerWidth={5} markerHeight={5} refX={4} refY={2.5} orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="var(--border-strong)" />
        </marker>
      </defs>
      {edges.map(([s, t], i) => {
        const [x1,y1] = nodes[s]!, [x2,y2] = nodes[t]!;
        return (
          <g key={i}>
            <line x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="var(--border-strong)" strokeWidth={0.8}
              markerEnd="url(#hiw-arr)" />
            {/* Travelling signal on every 3rd edge */}
            {i % 3 === 0 && (
              <circle r={2.5} fill="var(--bio)" opacity={0.85}>
                <animateMotion
                  dur={`${1.3 + (i % 4) * 0.45}s`}
                  repeatCount="indefinite"
                  path={`M${x1},${y1} L${x2},${y2}`}
                />
              </circle>
            )}
          </g>
        );
      })}
      {nodes.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={9}
          fill="var(--surface-2)" stroke="var(--border-strong)" strokeWidth={0.8} />
      ))}
    </svg>
  );
}

/** Chapter 02 — full connectome → subgraph shrink */
function ShrinkDiagram() {
  return (
    <svg viewBox="0 0 280 120" className="w-full max-w-[340px]"
      aria-label="Full connectome shrunk to 150-neuron subgraph">
      <defs>
        <marker id="hiw-arr2" markerWidth={5} markerHeight={5} refX={4} refY={2.5} orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="var(--border-strong)" />
        </marker>
      </defs>
      {/* Full connectome box */}
      <rect x={2} y={20} width={100} height={80} fill="none" stroke="var(--border-strong)" strokeWidth={0.8}/>
      <text x={52} y={14} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7.5}
        fill="var(--muted-foreground)" letterSpacing={0.8}>FULL CONNECTOME</text>
      <text x={52} y={60} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={11}
        fill="var(--foreground)">~21k</text>
      <text x={52} y={76} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7}
        fill="var(--muted-foreground)">neurons</text>
      {/* Arrow */}
      <line x1={108} y1={60} x2={156} y2={60}
        stroke="var(--border-strong)" strokeWidth={0.8} markerEnd="url(#hiw-arr2)" />
      <text x={132} y={52} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7}
        fill="var(--muted-foreground)">extract</text>
      {/* Subgraph box */}
      <rect x={160} y={38} width={60} height={44}
        fill="none" stroke="var(--bio)" strokeWidth={0.8}/>
      <text x={190} y={33} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7.5}
        fill="var(--bio)" letterSpacing={0.8}>SUBGRAPH</text>
      <text x={190} y={62} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={11}
        fill="var(--foreground)">150</text>
      <text x={190} y={78} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7}
        fill="var(--muted-foreground)">neurons</text>
    </svg>
  );
}

/** Chapter 03 — input → neurons → output flow */
function IOFlow() {
  const rows = ["INPUT", "NEURONS", "CONNECTIONS", "OUTPUT"];
  const h = 28, gap = 12, y0 = 8;
  return (
    <svg viewBox="0 0 180 200" className="w-full max-w-[200px]"
      aria-label="Computation flow: input through neurons to output">
      {rows.map((label, i) => (
        <g key={label}>
          <rect x={16} y={y0 + i*(h+gap)} width={148} height={h}
            fill="none" stroke="var(--border-strong)" strokeWidth={0.8}/>
          <text x={90} y={y0 + i*(h+gap) + 18} textAnchor="middle"
            fontFamily="var(--font-mono)" fontSize={9} letterSpacing={0.8}
            fill={i === 0 || i === 3 ? "var(--foreground)" : "var(--muted-foreground)"}>
            {label}
          </text>
          {i < rows.length - 1 && (
            <line
              x1={90} y1={y0 + i*(h+gap) + h}
              x2={90} y2={y0 + (i+1)*(h+gap)}
              stroke="var(--border-strong)" strokeWidth={0.8} strokeDasharray="3 3"/>
          )}
        </g>
      ))}
    </svg>
  );
}

/** Chapter 04 — fork/join fair comparison */
function FairComparisonDiagram() {
  return (
    <svg viewBox="0 0 320 210" className="w-full max-w-[400px]"
      aria-label="Same task split to two architectures, same evaluation">
      <defs>
        <marker id="hiw-arr3" markerWidth={5} markerHeight={5} refX={4} refY={2.5} orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="var(--border-strong)" />
        </marker>
      </defs>
      {/* SAME TASK */}
      <rect x={100} y={4} width={120} height={28} fill="none" stroke="var(--border-strong)" strokeWidth={0.8}/>
      <text x={160} y={23} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={9}
        fill="var(--foreground)" letterSpacing={0.8}>SAME TASK</text>
      {/* Fork */}
      <line x1={160} y1={32} x2={160} y2={54} stroke="var(--border-strong)" strokeWidth={0.8}/>
      <line x1={72}  y1={54} x2={248} y2={54} stroke="var(--border-strong)" strokeWidth={0.8}/>
      <line x1={72}  y1={54} x2={72}  y2={76} stroke="var(--border-strong)" strokeWidth={0.8} markerEnd="url(#hiw-arr3)"/>
      <line x1={248} y1={54} x2={248} y2={76} stroke="var(--border-strong)" strokeWidth={0.8} markerEnd="url(#hiw-arr3)"/>
      {/* Bio box */}
      <rect x={14} y={76} width={116} height={36} fill="none" stroke="var(--bio)" strokeWidth={0.8}/>
      <text x={72} y={92} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={8}
        fill="var(--bio)" letterSpacing={0.8}>MALECNS-DERIVED</text>
      <text x={72} y={105} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7}
        fill="var(--muted-foreground)">connectome-constrained</text>
      {/* AI box */}
      <rect x={190} y={76} width={116} height={36} fill="none" stroke="var(--ai)" strokeWidth={0.8}/>
      <text x={248} y={92} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={8}
        fill="var(--ai)" letterSpacing={0.8}>AI MODELS</text>
      <text x={248} y={105} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={7}
        fill="var(--muted-foreground)">conventional baselines</text>
      {/* Join */}
      <line x1={72}  y1={112} x2={72}  y2={150} stroke="var(--border-strong)" strokeWidth={0.8}/>
      <line x1={248} y1={112} x2={248} y2={150} stroke="var(--border-strong)" strokeWidth={0.8}/>
      <line x1={72}  y1={150} x2={248} y2={150} stroke="var(--border-strong)" strokeWidth={0.8}/>
      <line x1={160} y1={150} x2={160} y2={172} stroke="var(--border-strong)" strokeWidth={0.8} markerEnd="url(#hiw-arr3)"/>
      {/* SAME EVALUATION */}
      <rect x={90} y={172} width={140} height={28} fill="none" stroke="var(--border-strong)" strokeWidth={0.8}/>
      <text x={160} y={191} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={9}
        fill="var(--foreground)" letterSpacing={0.8}>SAME EVALUATION</text>
    </svg>
  );
}

/** Chapter 05 — T-001 bar chart (real data from benchmark.ts) */
function T001Bars() {
  const SHOW = ["A0-RANDOM","A1-BIO","A3-ER","A7-MLP","A8-LSTM"];
  const LABELS: Record<string,string> = {
    "A0-RANDOM": "Random",
    "A1-BIO":    "MaleCNS-derived",
    "A3-ER":     "Random graph (ER)",
    "A7-MLP":    "MLP",
    "A8-LSTM":   "LSTM",
  };
  const COLORS: Record<string,string> = {
    "A0-RANDOM": "var(--border-strong)",
    "A1-BIO":    "var(--bio)",
    "A3-ER":     "var(--control)",
    "A7-MLP":    "var(--ai)",
    "A8-LSTM":   "var(--ai)",
  };
  const rows = SHOW.map(id => T001.results.find(r => r.architectureId === id)).filter(Boolean);
  return (
    <div className="w-full max-w-md space-y-2.5">
      {rows.map(r => r && (
        <div key={r.architectureId} className="flex items-center gap-3">
          <div className="w-36 shrink-0 label-tech text-right text-muted-foreground leading-tight">
            {LABELS[r.architectureId]}
          </div>
          <div className="relative h-5 flex-1" style={{ background: "var(--surface)" }}>
            <div
              className="h-5"
              style={{ width: `${r.mean}%`, background: COLORS[r.architectureId], opacity: 0.8 }}
            />
          </div>
          <div className="w-12 shrink-0 tnum font-mono text-[12px]">
            {r.mean.toFixed(0)}%
          </div>
        </div>
      ))}
      <p className="label-tech mt-3 text-muted-foreground">
        Real data — T-001 confirmed results. All approach ceiling.
      </p>
    </div>
  );
}

/** Chapter 06 — temporal memory timeline */
function TemporalTimeline() {
  return (
    <svg viewBox="0 0 160 260" className="w-full max-w-[180px]"
      aria-label="Temporal sequence: stimulus, gap steps, recall">
      <defs>
        <marker id="hiw-arr4" markerWidth={5} markerHeight={5} refX={4} refY={2.5} orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="var(--border-strong)" />
        </marker>
      </defs>
      {/* STIMULUS */}
      <rect x={24} y={4} width={112} height={28}
        fill="var(--bio)" fillOpacity={0.12} stroke="var(--bio)" strokeWidth={0.8}/>
      <text x={80} y={23} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={9}
        fill="var(--bio)" letterSpacing={0.8}>STIMULUS</text>
      {/* 5 gap steps */}
      {Array.from({ length: 5 }).map((_, i) => (
        <g key={i}>
          <line x1={80} y1={32 + i*34} x2={80} y2={32 + i*34 + 24}
            stroke="var(--border-strong)" strokeWidth={0.8} strokeDasharray="3 3"
            markerEnd={i === 4 ? "url(#hiw-arr4)" : undefined} />
          <circle cx={80} cy={46 + i*34} r={5}
            fill="none" stroke="var(--border-strong)" strokeWidth={0.8}/>
          <text x={94} y={50 + i*34} fontFamily="var(--font-mono)" fontSize={6.5}
            fill="var(--muted-foreground)">step {i+1}</text>
        </g>
      ))}
      {/* RECALL */}
      <rect x={24} y={200} width={112} height={28}
        fill="none" stroke="var(--border-strong)" strokeWidth={0.8}/>
      <text x={80} y={219} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={9}
        fill="var(--foreground)" letterSpacing={0.8}>RECALL</text>
    </svg>
  );
}

/** Chapter 07 — T-002 result bars (real data) */
function T002Bars() {
  const bioResult    = T002.results.find(r => r.architectureId === "A1-BIO"    && r.variant?.startsWith("SA-010"));
  const frozenResult = T002.results.find(r => r.architectureId === "A1-FROZEN" && r.variant?.startsWith("SA-010"));
  const lstmResult   = T002.results.find(r => r.architectureId === "A8-LSTM");
  const randResult   = T002.results.find(r => r.architectureId === "A0-RANDOM");
  const rows = [
    { label: "MaleCNS-derived\n+ SA-010",         r: bioResult,    color: "var(--bio)" },
    { label: "MaleCNS-derived\n(frozen) + SA-010", r: frozenResult, color: "var(--bio)" },
    { label: "LSTM",                               r: lstmResult,   color: "var(--ai)" },
    { label: "Random baseline",                    r: randResult,   color: "var(--border-strong)" },
  ];
  return (
    <div className="w-full max-w-md space-y-3">
      {rows.map(({ label, r, color }) => (
        <div key={label} className="flex items-center gap-3">
          <div className="w-40 shrink-0 label-tech whitespace-pre-line text-right leading-tight text-muted-foreground">
            {label}
          </div>
          <div className="relative h-5 flex-1" style={{ background: "var(--surface)" }}>
            <div
              className="h-5 transition-all duration-700"
              style={{ width: r?.mean !== undefined ? `${r.mean}%` : "0%", background: color, opacity: 0.85 }}
            />
          </div>
          <div className="w-12 shrink-0 tnum font-mono text-[12px]">
            {r?.mean !== undefined ? `${r.mean.toFixed(1)}%` : "—"}
          </div>
        </div>
      ))}
      <p className="label-tech mt-2 text-muted-foreground">
        Real data — T-002 confirmation (5 seeds · held-out test set).
      </p>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────
function HowItWorks() {
  return (
    <div className="page-x mx-auto max-w-[1600px]">
      {/* Hero */}
      <header className="py-16 md:py-28">
        <div className="label-tech text-muted-foreground">How It Works</div>
        <h1 className="mt-6 text-5xl font-light leading-[0.94] tracking-[-0.04em] md:text-[5.5rem]">
          <span className="block">Science</span>
          <span className="block font-serif font-normal italic" style={{ color: "var(--muted-foreground)" }}>
            explained simply.
          </span>
        </h1>
        <p className="mt-8 max-w-xl text-[16px] leading-relaxed text-muted-foreground">
          A visual story. No neuroscience background required.
          Nine chapters from a real nervous system to a controlled AI experiment.
        </p>
      </header>

      {/* ── Chapter 01 ─────────────────────────────────────────────────── */}
      <Chapter n="01">
        <H>Start with the wiring.</H>
        <P>
          Scientists painstakingly mapped every neuron and every connection in
          the nervous system of an adult male fruit fly (<em>Drosophila melanogaster</em>).
          MaleCNS v1.0 contains roughly 21,000 neurons and tens of millions of
          synaptic connections — a real wiring diagram for a real nervous system.
        </P>
        <div className="mt-10 flex flex-wrap items-start gap-12">
          <NeuronRing />
          <div className="space-y-4">
            <div>
              <div className="label-tech text-muted-foreground">Organism</div>
              <div className="mt-1 text-sm text-muted-foreground italic">Drosophila melanogaster</div>
            </div>
            <div>
              <div className="label-tech text-muted-foreground">Dataset</div>
              <div className="tnum mt-1 text-2xl font-light">~21,000 <span className="text-sm">neurons</span></div>
            </div>
            <div>
              <div className="label-tech text-muted-foreground">Illustration note</div>
              <p className="mt-1 max-w-[220px] text-[12px] leading-relaxed text-muted-foreground"
                style={{ borderLeft: "2px solid var(--border-strong)", paddingLeft: "0.75rem" }}>
                Node positions are schematic — not anatomical coordinates.
                Real anatomical data is not used in the current prototype.
              </p>
            </div>
          </div>
        </div>
      </Chapter>

      {/* ── Chapter 02 ─────────────────────────────────────────────────── */}
      <Chapter n="02">
        <H>We can&apos;t run the whole thing in one experiment.</H>
        <P>
          21,000 neurons is computationally demanding for an initial benchmark.
          We extract a smaller <em>prototype subgraph</em> — a connected subset of
          150 neurons that preserves the wiring structure of the full connectome
          in a manageable form.
        </P>
        <div className="mt-10">
          <ShrinkDiagram />
        </div>
        <div className="mt-8 grid max-w-lg gap-4 sm:grid-cols-3">
          {[
            { label: "Prototype size",  value: "150",   unit: "neurons" },
            { label: "Connections",     value: "3,029", unit: "directed edges" },
            { label: "Graph density",   value: "0.136", unit: "" },
          ].map(({ label, value, unit }) => (
            <div key={label}>
              <div className="label-tech text-muted-foreground">{label}</div>
              <div className="tnum mt-1.5 text-2xl font-light">
                {value} {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-5 max-w-sm text-[13px] leading-relaxed text-muted-foreground"
          style={{ borderLeft: "2px solid var(--border)", paddingLeft: "0.75rem" }}>
          Graph statistics are real — extracted directly from the MaleCNS parquet data.
        </p>
      </Chapter>

      {/* ── Chapter 03 ─────────────────────────────────────────────────── */}
      <Chapter n="03">
        <H>Now give the wiring something to do.</H>
        <P>
          We use the connectome&apos;s wiring as a computational architecture — a layer
          in a neural network where only biologically real connections are allowed
          to carry signals. Inputs propagate through the connectome&apos;s topology
          and produce an output.
        </P>
        <div className="mt-10 flex flex-wrap items-start gap-12">
          <IOFlow />
          <div className="max-w-sm space-y-4 text-sm leading-relaxed text-muted-foreground">
            <p>
              <strong className="text-foreground">This is a computational model</strong>,
              not a simulation of biology. No neurons fire. No membrane potentials
              are computed. It is a mathematical graph with learned parameters,
              constrained to match real connectivity.
            </p>
            <p>
              Think of it as: <em>"what if a neural network&apos;s wiring
              was forced to look like this specific nervous system?"</em>
            </p>
          </div>
        </div>
      </Chapter>

      {/* ── Chapter 04 ─────────────────────────────────────────────────── */}
      <Chapter n="04">
        <H>Same problem. Same rules.</H>
        <P>
          To test whether biological wiring provides any computational benefit,
          we compare it against conventional AI architectures under identical
          conditions. Same data. Same task. Same evaluation. Multiple random seeds
          to prevent cherry-picking.
        </P>
        <div className="mt-10">
          <FairComparisonDiagram />
        </div>
        <div className="mt-8 grid max-w-xl gap-4 sm:grid-cols-2 text-sm leading-relaxed text-muted-foreground">
          <div>
            <div className="label-tech text-foreground mb-2">What is controlled</div>
            <ul className="space-y-1.5 list-disc list-inside">
              <li>Identical training data per seed</li>
              <li>Identical held-out test split</li>
              <li>Same input dimension (D_obs = 16)</li>
              <li>5 seeds per architecture</li>
              <li>Same evaluation metric</li>
            </ul>
          </div>
          <div>
            <div className="label-tech text-foreground mb-2">What varies</div>
            <ul className="space-y-1.5 list-disc list-inside">
              <li>Connectivity structure</li>
              <li>Trainable parameter count</li>
              <li>Temporal dynamics assumptions</li>
            </ul>
          </div>
        </div>
      </Chapter>

      {/* ── Chapter 05 ─────────────────────────────────────────────────── */}
      <Chapter n="05">
        <H>The easy test wasn&apos;t enough.</H>
        <P>
          The first task, T-001, asks each architecture to classify a static
          binary pattern presented once. No memory required — just a
          single feedforward decision.
        </P>
        <div className="mt-10">
          <T001Bars />
        </div>
        <P>
          Almost every architecture approaches perfect accuracy. Dense networks,
          random graphs, MLP, LSTM, and the MaleCNS-derived model all converge
          near the ceiling. This validates that our setup works, but it does not
          reveal whether biological wiring provides any unique advantage.
          T-001 is too easy.
        </P>
      </Chapter>

      {/* ── Chapter 06 ─────────────────────────────────────────────────── */}
      <Chapter n="06">
        <H>So we made time matter.</H>
        <P>
          T-002 introduces a memory gap. The model sees a stimulus, then must
          ignore several intervening steps, and only then recall which pattern
          it saw at the beginning. You cannot solve this with a single
          feedforward pass — you must carry information across time.
        </P>
        <div className="mt-10 flex flex-wrap items-start gap-12">
          <TemporalTimeline />
          <div className="max-w-sm space-y-4 text-sm leading-relaxed text-muted-foreground">
            <p>
              This mirrors a simple working-memory test: see something, wait,
              recall it. The gap length is configurable — longer gaps require
              more persistent memory.
            </p>
            <p>
              An LSTM has explicit gating mechanisms designed for this.
              The connectome-derived model relies on its recurrent connectivity
              and explicit leaky-decay temporal dynamics (SA-010 assumptions).
            </p>
          </div>
        </div>
      </Chapter>

      {/* ── Chapter 07 ─────────────────────────────────────────────────── */}
      <Chapter n="07">
        <H>Then something changed.</H>
        <P>
          On T-002, the MaleCNS-derived model with SA-010 temporal dynamics
          significantly outperforms the LSTM baseline — despite using far fewer
          trainable parameters.
        </P>
        <div className="mt-10">
          <T002Bars />
        </div>
        {/* Mandatory methodological caveat */}
        <div
          className="mt-8 max-w-2xl border-l pl-5 text-[13px] leading-relaxed"
          style={{ borderColor: "var(--caveat)", color: "var(--caveat)" }}
        >
          <strong>Methodological note:</strong> This is a task-specific
          computational result. The temporal configuration (decay 0.05 / leak 0.02)
          was selected via exploratory sensitivity analysis. Confidence intervals are
          wide (95% CI ≈ ±6%). The result establishes a finding worth further
          investigation — not a definitive biological claim.
        </div>
      </Chapter>

      {/* ── Chapter 08 ─────────────────────────────────────────────────── */}
      <Chapter n="08">
        <H>Not every test has a winner.</H>
        <P>
          The goal of ABB is not to prove that biological brains are universally
          better. It is to discover precisely where — and under what conditions —
          biological structure may provide useful computational properties.
        </P>
        <div className="mt-10 w-full max-w-2xl border border-border">
          {[
            {
              task:    "T-001",
              name:    "Binary Pattern Discrimination",
              finding: "MaleCNS ≈ random graph ≈ dense network. All near ceiling. No architectural advantage.",
              verdict: "No advantage found",
              color:   "var(--muted-foreground)",
            },
            {
              task:    "T-002",
              name:    "Temporal Sequence Memory",
              finding: `MaleCNS-derived + SA-010: 94.8%. LSTM: 57.6%. Strong divergence on the temporal task.`,
              verdict: "Advantage found",
              color:   "var(--bio)",
            },
          ].map(({ task, name, finding, verdict, color }) => (
            <div
              key={task}
              className="grid grid-cols-[72px_1fr_auto] items-start gap-4 border-b border-border p-5 last:border-b-0"
            >
              <div className="label-tech text-muted-foreground pt-0.5">{task}</div>
              <div>
                <div className="text-sm font-medium">{name}</div>
                <div className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">{finding}</div>
              </div>
              <div className="label-tech shrink-0 text-right" style={{ color }}>{verdict}</div>
            </div>
          ))}
        </div>
      </Chapter>

      {/* ── Chapter 09 ─────────────────────────────────────────────────── */}
      <Chapter n="09">
        <H>Now explore it.</H>
        <P>
          You&apos;ve seen the full story — from a real connectome to a controlled
          benchmark result. Explore the actual numbers, or run the T-002
          demonstration yourself.
        </P>
        <div className="mt-10 flex flex-wrap gap-4">
          <Link
            id="hiw-cta-benchmark"
            to="/benchmark"
            className="label-tech border border-foreground/80 bg-foreground px-6 py-3.5 !text-background transition-colors hover:bg-foreground/85"
          >
            Explore the benchmark →
          </Link>
          <Link
            id="hiw-cta-arena"
            to="/arena"
            className="label-tech border border-border-strong px-6 py-3.5 transition-colors hover:text-foreground"
          >
            Enter the live arena →
          </Link>
        </div>
      </Chapter>
    </div>
  );
}
