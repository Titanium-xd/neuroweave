import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { T002 } from "@/data/benchmark";

export const Route = createFileRoute("/arena")({
  head: () => ({
    meta: [
      { title: "Live Arena — Fruit Fly Brain vs AI | NeuroWeave" },
      {
        name: "description",
        content:
          "Interactive benchmark replay: MaleCNS-derived connectome-constrained model vs LSTM on the T-002 delayed-recall task.",
      },
    ],
  }),
  component: ArenaPage,
});

// ── Real accuracy from confirmed 5-seed T-002 experiment ─────────────────────
const BIO_ACC  = 0.948; // A1-BIO + SA-010 (decay 0.05 / leak 0.02)
const LSTM_ACC = 0.576; // A8-LSTM

type Phase = "idle" | "stimulus" | "gap" | "deciding" | "revealed";
type DelayN = 3 | 6 | 10;

interface Trial {
  stimulusIdx:  0 | 1;
  bioAnswer:    0 | 1;
  lstmAnswer:   0 | 1;
  bioCorrect:   boolean;
  lstmCorrect:  boolean;
}

// Binary patterns for the delayed-recall task
const PATTERNS = [
  { id: "A", label: "ALPHA", glyph: "●" },
  { id: "B", label: "BETA",  glyph: "◆" },
] as const;

// ── Seeded PRNG (same mulberry32 as connectome.ts) ───────────────────────────
function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a single T-002 delayed-recall trial.
 *
 * The stimulus is a binary choice (ALPHA / BETA). Each model independently
 * either recalls correctly or incorrectly, with probabilities drawn from the
 * real confirmed benchmark results (5-seed test accuracy).
 *
 * This faithfully mirrors the actual T-002 task structure:
 *   stimulus_t → gap (N steps) → binary recall decision
 */
function generateTrial(seed: number): Trial {
  const rand = mulberry32(seed);
  const si      = rand() < 0.5 ? 0 : 1;          // which pattern was shown
  const bioOk   = rand() < BIO_ACC;               // does bio model recall it?
  const lstmOk  = rand() < LSTM_ACC;              // does LSTM recall it?
  return {
    stimulusIdx:  si as 0 | 1,
    bioAnswer:    (bioOk  ? si : 1 - si) as 0 | 1,
    lstmAnswer:   (lstmOk ? si : 1 - si) as 0 | 1,
    bioCorrect:   bioOk,
    lstmCorrect:  lstmOk,
  };
}

// ── Bio activity visualisation (sparse graph, CSS pulsed) ───────────────────
function BioActivity({ active }: { active: boolean }) {
  // 6 nodes in a loose recurrent layout matching sparse connectome character
  const nodes: [number, number][] = [
    [18, 40], [50, 16], [50, 64], [88, 28], [88, 56], [114, 40],
  ];
  const edges: [number, number][] = [
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 5], [1, 4], [2, 3],
  ];
  const pulseClass = ["abb-p1","abb-p2","abb-p3","abb-p4","abb-p5","abb-p6"];

  return (
    <svg
      viewBox="0 0 130 80"
      className="w-full max-w-[160px]"
      aria-label="Illustrative sparse-network activity — not anatomical"
    >
      {edges.map(([s, t], i) => (
        <line
          key={i}
          x1={nodes[s][0]} y1={nodes[s][1]}
          x2={nodes[t][0]} y2={nodes[t][1]}
          stroke="var(--border-strong)" strokeWidth={0.8}
        />
      ))}
      {nodes.map(([x, y], i) => (
        <circle
          key={i} cx={x} cy={y} r={7}
          fill="var(--bio)"
          className={active ? pulseClass[i] : ""}
          style={{ opacity: active ? undefined : 0.12 }}
        />
      ))}
      <text
        x={65} y={78} textAnchor="middle"
        fontFamily="var(--font-mono)" fontSize={6}
        fill="var(--muted-foreground)" letterSpacing={0.5}
      >
        illustrative · not anatomical
      </text>
    </svg>
  );
}

// ── LSTM activity visualisation (4 gated cells, sequential fill) ────────────
function LstmActivity({ active }: { active: boolean }) {
  const labels = ["Input gate", "Forget gate", "Cell state", "Output gate"];
  return (
    <svg
      viewBox="0 0 130 90"
      className="w-full max-w-[160px]"
      aria-label="Illustrative LSTM gate states"
    >
      {labels.map((label, i) => (
        <g key={label} transform={`translate(0,${i * 22})`}>
          <rect x={4} y={2} width={120} height={16} fill="var(--border)" rx={1} />
          <rect
            x={4} y={2}
            width={active ? 120 : 0} height={16}
            fill="var(--ai)" rx={1} fillOpacity={0.55}
            style={{
              transition: active
                ? `width ${0.18 + i * 0.14}s ${i * 0.12}s ease-out`
                : "none",
            }}
          />
          <text
            x={8} y={14}
            fontFamily="var(--font-mono)" fontSize={6}
            fill="var(--muted-foreground)" letterSpacing={0.5}
          >
            {label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ── Gap timeline ─────────────────────────────────────────────────────────────
function GapTimeline({ steps, current }: { steps: number; current: number }) {
  return (
    <div className="flex flex-col items-center gap-0 py-2">
      <div className="label-tech mb-4 text-muted-foreground">↓ stimulus hidden</div>
      <div className="flex flex-col items-center gap-[10px]">
        {Array.from({ length: steps }).map((_, i) => (
          <div
            key={i}
            className="h-[9px] w-[9px] rounded-full border"
            style={{
              borderColor: i <= current ? "var(--foreground)" : "var(--border-strong)",
              background:  i <= current ? "var(--foreground)" : "transparent",
              transition: "background 200ms ease, border-color 200ms ease",
            }}
          />
        ))}
      </div>
      <div className="label-tech mt-4 text-muted-foreground">↓ recall</div>
    </div>
  );
}

// ── Main arena page ───────────────────────────────────────────────────────────
function ArenaPage() {
  const [phase,   setPhase]   = useState<Phase>("idle");
  const [trial,   setTrial]   = useState<Trial | null>(null);
  const [gapStep, setGapStep] = useState(0);
  const [delay,   setDelay]   = useState<DelayN>(6);
  const [score,   setScore]   = useState({ bio: 0, lstm: 0, total: 0 });
  const seedRef = useRef(Date.now());

  // Stimulus → gap
  useEffect(() => {
    if (phase !== "stimulus") return;
    const t = setTimeout(() => { setPhase("gap"); setGapStep(0); }, 1800);
    return () => clearTimeout(t);
  }, [phase]);

  // Gap step advance
  useEffect(() => {
    if (phase !== "gap") return;
    const t = setTimeout(() => {
      if (gapStep + 1 >= delay) {
        setPhase("deciding");
      } else {
        setGapStep(s => s + 1);
      }
    }, 390);
    return () => clearTimeout(t);
  }, [phase, gapStep, delay]);

  // Deciding → revealed (score tallied here)
  useEffect(() => {
    if (phase !== "deciding") return;
    const t = setTimeout(() => {
      setPhase("revealed");
      setTrial(prev => {
        if (prev) {
          setScore(s => ({
            bio:   s.bio   + (prev.bioCorrect  ? 1 : 0),
            lstm:  s.lstm  + (prev.lstmCorrect ? 1 : 0),
            total: s.total + 1,
          }));
        }
        return prev;
      });
    }, 1000);
    return () => clearTimeout(t);
  }, [phase]);

  const startRun = useCallback(() => {
    seedRef.current += 1;
    setTrial(generateTrial(seedRef.current));
    setPhase("stimulus");
  }, []);

  const resetAll = useCallback(() => {
    setPhase("idle");
    setTrial(null);
    setGapStep(0);
    setScore({ bio: 0, lstm: 0, total: 0 });
    seedRef.current = Date.now();
  }, []);

  // Real data from benchmark.ts
  const bioResult    = T002.results.find(r => r.architectureId === "A1-BIO"    && r.variant?.startsWith("SA-010"));
  const frozenResult = T002.results.find(r => r.architectureId === "A1-FROZEN" && r.variant?.startsWith("SA-010"));
  const lstmResult   = T002.results.find(r => r.architectureId === "A8-LSTM");
  const randResult   = T002.results.find(r => r.architectureId === "A0-RANDOM");

  const isRunning = phase === "stimulus" || phase === "gap" || phase === "deciding";

  return (
    <div className="page-x mx-auto max-w-[1600px]">
      {/* ── Page header ──────────────────────────────────────────────────── */}
      <header className="border-b border-border py-12 md:py-16">
        <div className="max-w-3xl">
          <div className="label-tech">Interactive Benchmark Replay · T-002</div>
          <h1 className="mt-5 text-4xl font-light leading-tight tracking-[-0.035em] md:text-5xl">
            Fruit Fly Brain<br />
            <span className="font-serif italic" style={{ color: "var(--muted-foreground)" }}>vs AI</span>
          </h1>
          <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-muted-foreground">
            Watch a MaleCNS-derived connectome-constrained model and an LSTM tackle
            the same delayed-recall challenge — the task where benchmark performance diverges most.
          </p>
          {/* Permanent scientific framing caveat */}
          <div
            className="mt-6 max-w-xl border-l pl-4 text-[13px] leading-relaxed"
            style={{ borderColor: "var(--caveat)", color: "var(--caveat)" }}
          >
            <strong>Benchmark replay.</strong> Trial outcomes are drawn from measured benchmark
            accuracy probabilities (A1-BIO+SA-010: {(BIO_ACC * 100).toFixed(1)}%,
            LSTM: {(LSTM_ACC * 100).toFixed(1)}%). The trained models are not running live in
            the browser.
          </div>
        </div>
      </header>

      {/* ── Controls + session score ─────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-x-8 gap-y-3 border-b border-border py-4">
        {/* Run / Reset */}
        <div className="flex items-center gap-3">
          {!isRunning ? (
            <button
              id="arena-run"
              type="button"
              onClick={startRun}
              className="label-tech border border-foreground/80 bg-foreground px-5 py-2.5 !text-background transition-colors hover:bg-foreground/85"
            >
              {phase === "revealed" ? "Run again" : "Run"}
            </button>
          ) : (
            <span className="label-tech border border-border px-5 py-2.5 text-muted-foreground">
              Running…
            </span>
          )}
          <button
            id="arena-reset"
            type="button"
            onClick={resetAll}
            className="label-tech border border-border px-4 py-2.5 transition-colors hover:border-border-strong hover:text-foreground"
          >
            Reset
          </button>
        </div>

        {/* Delay selector */}
        <div className="flex items-center gap-2">
          <span className="label-tech text-muted-foreground">Delay</span>
          {([3, 6, 10] as const).map(n => (
            <button
              key={n}
              id={`arena-delay-${n}`}
              type="button"
              onClick={() => setDelay(n)}
              disabled={isRunning}
              className="label-tech border px-3 py-1.5 transition-colors disabled:opacity-40"
              style={{
                borderColor: delay === n ? "var(--foreground)" : "var(--border)",
                color:       delay === n ? "var(--foreground)" : undefined,
              }}
            >
              {n}
            </button>
          ))}
          <span className="label-tech text-muted-foreground">steps</span>
        </div>

        {/* Session tally */}
        {score.total > 0 && (
          <div className="ml-auto flex items-center gap-6">
            <span className="label-tech" style={{ color: "var(--bio)" }}>
              MaleCNS {score.bio}/{score.total}
            </span>
            <span className="label-tech" style={{ color: "var(--ai)" }}>
              LSTM {score.lstm}/{score.total}
            </span>
          </div>
        )}
      </div>

      {/* ── Stage display ────────────────────────────────────────────────── */}
      <div className="min-h-[420px] py-10">

        {/* IDLE */}
        {phase === "idle" && (
          <div className="flex flex-col items-center gap-6 py-16 text-center">
            <div className="label-tech text-muted-foreground">T-002 · Delayed Recall</div>
            <p className="max-w-sm text-sm leading-relaxed text-muted-foreground">
              A binary pattern is shown briefly. After a configurable memory gap,
              each model must recall which pattern it saw. Same input, same delay —
              different architectures.
            </p>
            <button
              type="button"
              onClick={startRun}
              className="label-tech mt-2 border border-foreground/80 bg-foreground px-7 py-3.5 !text-background transition-colors hover:bg-foreground/85"
            >
              Begin
            </button>
          </div>
        )}

        {/* STIMULUS */}
        {phase === "stimulus" && trial && (
          <div className="flex flex-col items-center gap-6 py-8">
            <div className="label-tech text-muted-foreground">Remember this</div>
            <div
              className="flex h-44 w-44 flex-col items-center justify-center border"
              style={{ borderColor: "var(--border-strong)" }}
            >
              <span className="text-6xl" style={{ color: "var(--foreground)" }}>
                {PATTERNS[trial.stimulusIdx].glyph}
              </span>
              <span className="label-tech mt-3">{PATTERNS[trial.stimulusIdx].label}</span>
            </div>
            <div className="label-tech text-muted-foreground animate-pulse">memorising…</div>
          </div>
        )}

        {/* GAP */}
        {phase === "gap" && trial && (
          <div className="flex flex-col items-center gap-6 py-8">
            <div className="label-tech text-muted-foreground">
              Memory gap · {delay} step{delay > 1 ? "s" : ""}
            </div>
            <GapTimeline steps={delay} current={gapStep} />
          </div>
        )}

        {/* DECIDING + REVEALED */}
        {(phase === "deciding" || phase === "revealed") && trial && (
          <div className="space-y-6">
            {/* Recall prompt */}
            <div className="flex flex-wrap items-center gap-3 border-b border-border pb-5">
              <span className="label-tech text-muted-foreground">Stimulus was</span>
              <span className="tnum font-mono text-lg">
                {PATTERNS[trial.stimulusIdx].glyph} {PATTERNS[trial.stimulusIdx].label}
              </span>
              <span className="label-tech text-muted-foreground">
                — both models now recall from memory
              </span>
            </div>

            {/* Model panels */}
            <div className="grid gap-4 sm:grid-cols-2">
              {/* MaleCNS panel */}
              <div className="border border-border p-5">
                <div className="flex items-center gap-2 border-b border-border pb-3 mb-4">
                  <span className="h-[7px] w-[7px] rounded-full" style={{ background: "var(--bio)" }} />
                  <span className="label-tech" style={{ color: "var(--bio)" }}>MaleCNS-derived</span>
                </div>
                <div className="label-tech mb-3 text-muted-foreground">
                  Connectome-constrained · SA-010 temporal dynamics
                </div>
                <div className="flex justify-center py-3">
                  <BioActivity active={phase === "deciding"} />
                </div>
                {phase === "deciding" && (
                  <div className="mt-3 border-t border-border pt-3 text-center label-tech text-muted-foreground">
                    processing…
                  </div>
                )}
                {phase === "revealed" && (
                  <div className="mt-3 border-t border-border pt-4 text-center space-y-1">
                    <div className="label-tech text-muted-foreground">prediction</div>
                    <div className="tnum text-2xl font-light" style={{ color: "var(--bio)" }}>
                      {PATTERNS[trial.bioAnswer].glyph} {PATTERNS[trial.bioAnswer].label}
                    </div>
                    <div
                      className="label-tech text-[15px]"
                      style={{ color: trial.bioCorrect ? "var(--bio)" : "var(--destructive)" }}
                    >
                      {trial.bioCorrect ? "✓ CORRECT" : "✗ INCORRECT"}
                    </div>
                  </div>
                )}
              </div>

              {/* LSTM panel */}
              <div className="border border-border p-5">
                <div className="flex items-center gap-2 border-b border-border pb-3 mb-4">
                  <span className="h-[7px] w-[7px] rounded-full" style={{ background: "var(--ai)" }} />
                  <span className="label-tech" style={{ color: "var(--ai)" }}>LSTM</span>
                </div>
                <div className="label-tech mb-3 text-muted-foreground">
                  Conventional recurrent baseline · 266,626 parameters
                </div>
                <div className="flex justify-center py-3">
                  <LstmActivity active={phase === "deciding"} />
                </div>
                {phase === "deciding" && (
                  <div className="mt-3 border-t border-border pt-3 text-center label-tech text-muted-foreground">
                    processing…
                  </div>
                )}
                {phase === "revealed" && (
                  <div className="mt-3 border-t border-border pt-4 text-center space-y-1">
                    <div className="label-tech text-muted-foreground">prediction</div>
                    <div className="tnum text-2xl font-light" style={{ color: "var(--ai)" }}>
                      {PATTERNS[trial.lstmAnswer].glyph} {PATTERNS[trial.lstmAnswer].label}
                    </div>
                    <div
                      className="label-tech text-[15px]"
                      style={{ color: trial.lstmCorrect ? "var(--bio)" : "var(--destructive)" }}
                    >
                      {trial.lstmCorrect ? "✓ CORRECT" : "✗ INCORRECT"}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Run again nudge */}
            {phase === "revealed" && (
              <div className="flex gap-4 pt-2">
                <button
                  type="button"
                  onClick={startRun}
                  className="label-tech border border-foreground/80 bg-foreground px-5 py-2.5 !text-background transition-colors hover:bg-foreground/85"
                >
                  Run again
                </button>
                <button
                  type="button"
                  onClick={resetAll}
                  className="label-tech border border-border px-4 py-2.5 transition-colors hover:border-border-strong hover:text-foreground"
                >
                  Reset session
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Benchmark context (always visible) ───────────────────────────── */}
      <div className="rule-t py-10">
        <div className="label-tech mb-6 text-muted-foreground">
          Benchmark context · T-002 confirmed results (5-seed · held-out test set)
        </div>
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          {([
            { label: "MaleCNS-derived\n+ SA-010", r: bioResult,    color: "var(--bio)" },
            { label: "MaleCNS-derived\n(frozen) + SA-010", r: frozenResult, color: "var(--bio)" },
            { label: "LSTM", r: lstmResult,   color: "var(--ai)" },
            { label: "Random baseline", r: randResult,    color: "var(--border-strong)" },
          ] as const).map(({ label, r, color }) => (
            <div key={label} className="border border-border p-4">
              <div className="label-tech whitespace-pre-line" style={{ color }}>{label}</div>
              <div className="tnum mt-2 text-2xl font-light">
                {r?.mean !== undefined ? `${r.mean.toFixed(1)}%` : "—"}
              </div>
              {r?.ci95 !== null && r?.ci95 !== undefined && (
                <div className="label-tech mt-0.5 text-muted-foreground">
                  ± {r.ci95.toFixed(1)}% (95% CI)
                </div>
              )}
              <div className="label-tech mt-1.5 text-muted-foreground">
                {r?.params !== null && r?.params !== undefined
                  ? `${r.params.toLocaleString()} train params`
                  : "—"}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-6 max-w-2xl text-[13px] leading-relaxed text-muted-foreground">
          SA-010 configuration (decay 0.05 / leak 0.02) was selected via exploratory
          sensitivity analysis. Results are task-specific and should not be interpreted
          as a universal biological conclusion.{" "}
          <Link
            to="/experiment/$taskId"
            params={{ taskId: "t-002" }}
            className="underline decoration-muted-foreground/50 hover:text-foreground"
          >
            Full T-002 results →
          </Link>
        </p>
      </div>
    </div>
  );
}
