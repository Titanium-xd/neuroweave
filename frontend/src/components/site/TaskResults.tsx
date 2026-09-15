import { useState } from "react";
import { ARCH_BY_ID, FAMILY_META, type Family, type Measurement, type Task } from "@/data/benchmark";

const GROUPS: Family[] = ["baseline", "bio", "control", "ai"];

const fmt = (v: number | null, suffix = "") => (v === null ? "—" : `${v}${suffix}`);

function Row({
  m,
  index,
  open,
  onToggle,
}: {
  m: Measurement;
  index: number;
  open: boolean;
  onToggle: () => void;
}) {
  const arch = ARCH_BY_ID[m.architectureId]!;
  const color = FAMILY_META[arch.family].token;
  const lo = Math.max(0, m.mean - m.sd);
  const hi = Math.min(100, m.mean + m.sd);
  const emphasise = m.architectureId === "A1-BIO";

  return (
    <div className="border-b border-border last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-6 gap-y-3 py-4 text-left transition-colors duration-200 hover:bg-surface/60 md:grid-cols-[190px_minmax(0,1fr)_112px] md:gap-x-8"
      >
        <div className="flex min-w-0 flex-col gap-1 md:pl-4">
          <span
            className="tnum truncate font-mono text-[12.5px] tracking-wide"
            style={{ color: emphasise ? "var(--foreground)" : undefined }}
          >
            {emphasise && (
              <span
                className="mr-2 inline-block h-[7px] w-[7px] align-middle"
                style={{ background: color }}
              />
            )}
            {arch.label}
          </span>
          {m.variant && (
            <span className="truncate text-[11px] text-muted-foreground">{m.variant}</span>
          )}
        </div>

        {/* plot */}
        <div className="col-span-2 md:col-span-1">
          <div className="relative h-4">
            {[0, 25, 50, 75, 100].map((g) => (
              <span
                key={g}
                className="absolute top-0 h-4 w-px bg-border"
                style={{ left: `${g}%` }}
              />
            ))}
            <span className="absolute left-0 top-1/2 h-px w-full -translate-y-1/2 bg-border" />
            <span
              className="absolute top-1/2 h-px -translate-y-1/2"
              style={{
                left: `${lo}%`,
                width: `${Math.max(0, hi - lo)}%`,
                background: color,
                opacity: 0.6,
                transition: "opacity 200ms linear",
              }}
            />
            {[lo, hi].map((v, i) => (
              <span
                key={i}
                className="absolute top-1/2 h-[9px] w-px -translate-y-1/2"
                style={{ left: `${v}%`, background: color, opacity: 0.6 }}
              />
            ))}
            <span
              className="absolute top-1/2 h-[7px] w-[7px] -translate-x-1/2 -translate-y-1/2 rounded-full"
              style={{ left: `${m.mean}%`, background: color }}
            />
          </div>
        </div>


        <div className="tnum text-right font-mono text-[13px] md:pr-4">
          <span>{m.mean.toFixed(1)}%</span>
          <span className="ml-1 text-[11px] text-muted-foreground">± {m.sd.toFixed(1)}</span>
        </div>
      </button>

      {open && (
        <div className="grid gap-6 bg-surface/50 px-4 py-5 md:grid-cols-[190px_minmax(0,1fr)] md:gap-8">
          <div className="label-tech">Row {String(index + 1).padStart(2, "0")}</div>
          <div className="space-y-4">
            <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
              {arch.description}
            </p>
            {m.note && (
              <p
                className="max-w-2xl border-l pl-3 text-sm leading-relaxed"
                style={{ borderColor: "var(--caveat)", color: "var(--caveat)" }}
              >
                {m.note}
              </p>
            )}
            <dl className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-4">
              {[
                ["Mean accuracy", `${m.mean.toFixed(1)}%`],
                ["SD across seeds", `${m.sd.toFixed(1)}%`],
                ["95% CI (\u00b1)", m.ci95 !== null ? `\u00b1 ${m.ci95.toFixed(1)}%` : "\u2014"],
                ["Seeds", fmt(m.seeds)],
                ["Train params", m.params === null ? "\u2014" : m.params.toLocaleString()],
                ["Frozen params", m.frozenParams !== null && m.frozenParams > 0 ? m.frozenParams.toLocaleString() : (m.frozenParams === 0 ? "0" : "\u2014")],
                ["Training time", m.trainSeconds === null ? "\u2014" : `${m.trainSeconds.toFixed(1)}s`],
                ["Family", FAMILY_META[arch.family].label],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="label-tech">{k}</dt>
                  <dd className="tnum mt-1.5 font-mono text-[13px]">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}

export function TaskResults({ task }: { task: Task }) {
  const [openKey, setOpenKey] = useState<string | null>(null);
  const rows = task.results.map((m, i) => ({ m, i }));

  return (
    <div>
      {/* axis */}
      <div className="hidden grid-cols-[190px_minmax(0,1fr)_112px] items-end gap-x-8 border-b border-border pb-2 md:grid">
        <span className="label-tech pl-4">Architecture</span>
        <div className="relative h-4">
          {[0, 25, 50, 75, 100].map((g) => (
            <span
              key={g}
              className="label-tech absolute bottom-0 -translate-x-1/2"
              style={{ left: `${g}%` }}
            >
              {g}
            </span>
          ))}
        </div>
        <span className="label-tech pr-4 text-right">Accuracy</span>
      </div>

      {GROUPS.map((family) => {
        const group = rows.filter(({ m }) => ARCH_BY_ID[m.architectureId]!.family === family);
        if (!group.length) return null;
        return (
          <section key={family} className="border-b border-border last:border-b-0">
            <div className="flex items-center gap-3 bg-surface/40 px-4 py-2.5">
              <span
                className="inline-block h-[6px] w-[6px]"
                style={{ background: FAMILY_META[family].token }}
              />
              <span className="label-tech !text-foreground">{FAMILY_META[family].label}</span>
            </div>
            <div>
              {group.map(({ m, i }) => {
                const key = `${m.architectureId}-${i}`;
                return (
                  <Row
                    key={key}
                    m={m}
                    index={i}
                    open={openKey === key}
                    onToggle={() => setOpenKey(openKey === key ? null : key)}
                  />
                );
              })}
            </div>
          </section>
        );
      })}

      <p className="max-w-3xl px-4 py-4 font-mono text-[11px] leading-relaxed text-muted-foreground">
        Range marks show &plusmn;1 SD across seeds. All metrics (95% CI, parameter counts, seed counts,
        training times) are sourced from real NeuroWeave experiment artifacts. Click any row to expand.
      </p>

    </div>
  );
}
