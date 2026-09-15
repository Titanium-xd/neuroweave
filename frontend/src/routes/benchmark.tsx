import { createFileRoute, Link } from "@tanstack/react-router";
import { TaskResults } from "@/components/site/TaskResults";
import { FAMILY_META, TASKS, type Family } from "@/data/benchmark";

export const Route = createFileRoute("/benchmark")({
  head: () => ({
    meta: [
      { title: "Benchmark — T-001 & T-002 | ABB" },
      {
        name: "description",
        content:
          "Multi-seed benchmark results for MaleCNS-derived, topology-control and conventional architectures on static pattern discrimination and temporal sequence memory.",
      },
      { property: "og:title", content: "Benchmark — T-001 & T-002 | ABB" },
      {
        property: "og:description",
        content:
          "Accuracy with seed-level uncertainty across bio-constrained, control and conventional architectures.",
      },
    ],
  }),
  component: Benchmark,
});

function Benchmark() {
  return (
    <div className="page-x mx-auto max-w-[1600px]">
      <header className="grid gap-10 py-16 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:gap-20 md:py-24">
        <div>
          <div className="label-tech">Benchmark</div>
          <h1 className="mt-6 text-5xl font-light leading-[0.95] tracking-[-0.035em] md:text-[4.5rem]">
            Controlled
            <br />
            comparison
          </h1>
        </div>
        <div className="max-w-xl md:pt-14">
          <p className="text-[17px] leading-relaxed">
            Two confirmed tasks. Every architecture is trained and evaluated per seed under
            identical data conditions; the plots show mean accuracy with ±1 SD across seeds.
          </p>
          <div className="mt-8 flex flex-wrap gap-x-8 gap-y-3">
            {(["bio", "control", "ai", "baseline"] as Family[]).map((f) => (
              <span key={f} className="flex items-center gap-2">
                <span
                  className="inline-block h-[7px] w-[7px]"
                  style={{ background: FAMILY_META[f].token }}
                />
                <span className="label-tech">{FAMILY_META[f].label}</span>
              </span>
            ))}
          </div>
        </div>
      </header>

      {TASKS.map((task) => (
        <section key={task.id} className="rule-t py-14 md:py-20">
          <div className="grid gap-10 md:grid-cols-[240px_minmax(0,1fr)] md:gap-16">
            <div>
              <div className="label-tech md:sticky md:top-24">{task.code}</div>
              <h2 className="mt-3 text-2xl tracking-tight md:sticky md:top-32">
                {task.title}
                <span className="mt-2 block text-sm text-muted-foreground">{task.tagline}</span>
              </h2>
            </div>
            <div className="min-w-0">
              <p className="max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
                {task.description}
              </p>

              <div className="mt-10 border border-border">
                <TaskResults task={task} />
              </div>

              <div className="mt-10 grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <div>
                  <div className="label-tech">Reading</div>
                  <p className="mt-3 max-w-lg text-[15px] leading-relaxed">{task.conclusion}</p>
                </div>
                <div>
                  <div className="label-tech">Protocol</div>
                  <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted-foreground">
                    {task.protocol}
                  </p>
                </div>
              </div>

              {task.caveat && (
                <div
                  className="mt-10 border-l pl-5"
                  style={{ borderColor: "var(--caveat)" }}
                >
                  <div className="label-tech" style={{ color: "var(--caveat)" }}>
                    Methodological note
                  </div>
                  <p
                    className="mt-3 max-w-2xl text-[14px] leading-relaxed"
                    style={{ color: "var(--caveat)" }}
                  >
                    {task.caveat}
                  </p>
                </div>
              )}

              <div className="mt-10">
                <Link
                  to="/experiment/$taskId"
                  params={{ taskId: task.id }}
                  className="label-tech border-b border-border-strong pb-1 transition-colors duration-200 hover:text-foreground"
                >
                  Open {task.code} experiment detail
                </Link>
              </div>
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}
