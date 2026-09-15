import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { Disclosure, Section } from "@/components/site/Disclosure";
import { TaskResults } from "@/components/site/TaskResults";
import { ARCH_BY_ID, TASK_BY_ID, TASKS, type Task } from "@/data/benchmark";

export const Route = createFileRoute("/experiment/$taskId")({
  loader: ({ params }) => {
    const task = TASK_BY_ID[params.taskId];
    if (!task) throw notFound();
    return { task };
  },
  head: ({ loaderData }) => {
    if (!loaderData) {
      return { meta: [{ title: "Experiment unavailable | ABB" }, { name: "robots", content: "noindex" }] };
    }
    const t = loaderData.task;
    const title = `${t.code} — ${t.title} | ABB`;
    return {
      meta: [
        { title },
        { name: "description", content: t.description.slice(0, 155) },
        { property: "og:title", content: title },
        { property: "og:description", content: t.conclusion.slice(0, 155) },
      ],
    };
  },
  component: ExperimentDetail,
});

function ExperimentDetail() {
  const { task } = Route.useLoaderData() as { task: Task };
  const other = TASKS.find((t) => t.id !== task.id);

  return (
    <div className="page-x mx-auto max-w-[1600px]">
      <header className="grid gap-10 py-16 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:gap-20 md:py-24">
        <div>
          <div className="label-tech">Experiment detail · {task.code}</div>
          <h1 className="mt-6 text-4xl font-light leading-[1.0] tracking-[-0.035em] md:text-[3.6rem]">
            {task.title}
          </h1>
        </div>
        <div className="max-w-xl md:pt-14">
          <p className="text-[16px] leading-relaxed text-muted-foreground">{task.description}</p>
        </div>
      </header>

      <Section code="01" title="Results">
        <div className="border border-border">
          <TaskResults task={task} />
        </div>
        {task.caveat && (
          <div className="mt-8 border-l pl-5" style={{ borderColor: "var(--caveat)" }}>
            <div className="label-tech" style={{ color: "var(--caveat)" }}>
              Methodological note
            </div>
            <p className="mt-3 max-w-2xl text-[14px] leading-relaxed" style={{ color: "var(--caveat)" }}>
              {task.caveat}
            </p>
          </div>
        )}
      </Section>

      <Section code="02" title="Specification">
        <div>
          <Disclosure title="Task description" meta={task.tagline} defaultOpen>
            <p>{task.description}</p>
            <p>{task.conclusion}</p>
          </Disclosure>
          <Disclosure title="Observation space">
            <p>{task.observationSpace}</p>
          </Disclosure>
          <Disclosure title="Architectures evaluated" meta={`${task.results.length} rows`}>
            <ul className="space-y-3">
              {task.results.map((m, i) => (
                <li key={`${m.architectureId}-${i}`}>
                  <span className="tnum font-mono text-[12.5px] text-foreground">
                    {ARCH_BY_ID[m.architectureId]!.label}
                  </span>
                  {m.variant && <span className="ml-2 text-[12px]">{m.variant}</span>}
                  <span className="block">{ARCH_BY_ID[m.architectureId]!.description}</span>
                </li>
              ))}
            </ul>
          </Disclosure>
          <Disclosure title="Parameter counts and training time">
            <p>
              Trainable parameter counts, frozen parameter counts, and mean wall-clock training
              times per seed are sourced from the real ABB experiment artifact JSON files.
              Click any row in the results table above to see per-architecture details.
            </p>
          </Disclosure>
          <Disclosure title="Seeds, mean, SD and confidence intervals">
            <p>{task.protocol}</p>
            <p>
              The reported spread is the standard deviation across seeds. The 95% CI (half-width)
              is computed via a t-distribution and reported as a &plusmn; value in the expanded row detail.
              All runs used 5 seeds.
            </p>
          </Disclosure>
          <Disclosure title="Training curve">
            <p>
              Per-epoch training curves are not exported by the current prototype run. This section
              is reserved for the run-level curve data once the benchmark backend writes it.
            </p>
          </Disclosure>
          <Disclosure title="Benchmark configuration">
            <ul className="space-y-2">
              <li>Connectome source — MaleCNS v1.0, adult male <em>Drosophila</em> CNS.</li>
              <li>Prototype subgraph — 150 nodes, used as a fixed connectivity mask.</li>
              <li>Evaluation — multi-seed, identical data conditions per seed.</li>
              <li>Metric — classification accuracy on held-out data.</li>
            </ul>
          </Disclosure>
          <Disclosure title="Assumptions">
            <ul className="space-y-2">
              {task.assumptions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </Disclosure>
          <Disclosure title="Notes">
            <ul className="space-y-2">
              {task.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </Disclosure>
        </div>
      </Section>

      <Section code="03" title="Continue">
        <div className="flex flex-wrap gap-8">
          {other && (
            <Link
              to="/experiment/$taskId"
              params={{ taskId: other.id }}
              className="label-tech border-b border-border-strong pb-1 transition-colors duration-200 hover:text-foreground"
            >
              {other.code} — {other.title}
            </Link>
          )}
          <Link
            to="/methodology"
            className="label-tech border-b border-border-strong pb-1 transition-colors duration-200 hover:text-foreground"
          >
            Methodology
          </Link>
        </div>
      </Section>
    </div>
  );
}
