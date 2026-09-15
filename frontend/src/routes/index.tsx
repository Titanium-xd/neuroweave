import { createFileRoute, Link } from "@tanstack/react-router";
import { ConnectomeField } from "@/components/site/ConnectomeField";
import { SignatureMotif } from "@/components/site/SignatureMotif";
import { CountUp } from "@/components/site/CountUp";
import { AT_A_GLANCE, TASKS, T002, ARCH_BY_ID } from "@/data/benchmark";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Fruit Fly Brain vs AI — Animal Brain Benchmark" },
      {
        name: "description",
        content:
          "ABB compares connectome-topology-constrained neural architectures derived from the MaleCNS v1.0 Drosophila connectome against conventional artificial neural architectures under controlled tasks.",
      },
      { property: "og:title", content: "Fruit Fly Brain vs AI — Animal Brain Benchmark" },
      {
        property: "og:description",
        content:
          "Can neural wiring evolved by biology provide useful computational structure for artificial intelligence?",
      },
    ],
  }),
  component: Home,
});

function Home() {
  const headline = T002.results.find(
    (r) => r.architectureId === "A1-BIO" && r.variant?.startsWith("SA-010"),
  )!;

  return (
    <>
      {/* HERO */}
      <section className="relative overflow-hidden border-b border-border">
        <ConnectomeField className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.55]" />
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 80% at 15% 20%, transparent 20%, var(--background) 78%)",
          }}
        />
        <div className="page-x relative mx-auto max-w-[1600px] pb-20 pt-20 md:pb-28 md:pt-32">
          <div className="abb-enter grid gap-14 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)] lg:items-end lg:gap-20">
            <div>
              <div className="label-tech">MaleCNS v1.0 · Drosophila CNS · computational study</div>
              <h1 className="mt-7 text-[15vw] font-light leading-[0.86] tracking-[-0.045em] sm:text-[11vw] lg:text-[7.4rem] xl:text-[8.6rem]">
                <span className="block">FRUIT FLY</span>
                <span className="block">BRAIN</span>
                <span className="block font-serif font-normal italic tracking-[-0.02em] text-muted-foreground">
                  vs AI
                </span>
              </h1>
            </div>

            <div className="max-w-xl lg:pb-4">
              <p className="text-[19px] leading-relaxed text-foreground md:text-[21px]">
                Can neural wiring evolved by biology provide useful computational structure for
                artificial intelligence?
              </p>
              <p className="mt-6 text-sm leading-relaxed text-muted-foreground">
                ABB compares connectome-topology-constrained neural architectures derived from the
                MaleCNS v1.0 <em>Drosophila</em> connectome against conventional artificial neural
                architectures under controlled tasks.
              </p>
              <div className="mt-9 flex flex-wrap items-stretch gap-3">
                <Link
                  to="/benchmark"
                  className="label-tech flex items-center border border-foreground/85 bg-foreground px-5 py-3 !text-background transition-colors duration-200 hover:bg-foreground/85"
                >
                  Explore the benchmark
                </Link>
                <Link
                  to="/connectome"
                  className="label-tech flex items-center border border-border-strong px-5 py-3 transition-colors duration-200 hover:text-foreground"
                >
                  Explore the connectome
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* AT A GLANCE */}
      <section className="border-b border-border">
        <div className="page-x mx-auto max-w-[1600px]">
          <div className="grid divide-border md:grid-cols-4 md:divide-x">
            {AT_A_GLANCE.map((s, i) => (
              <div
                key={s.label}
                className={`border-b border-border py-7 md:border-b-0 ${i === 0 ? "md:pr-8" : "md:px-8"} ${i === 3 ? "md:pr-0" : ""} last:border-b-0`}
              >
                <div className="label-tech">{s.label}</div>
                <div className="tnum mt-3 flex items-baseline gap-2">
                  <span className="text-3xl font-light tracking-tight">{s.value}</span>
                  <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                    {s.unit}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="grid gap-x-16 gap-y-4 border-t border-border py-7 md:grid-cols-2">
            <p className="text-sm leading-relaxed text-muted-foreground">
              Prototype subgraphs of 150 nodes are extracted from the connectome and used as a
              fixed connectivity mask. Every architecture is evaluated across multiple random
              seeds; reported spread is the standard deviation across those seeds.
            </p>
            <p className="text-sm leading-relaxed text-muted-foreground">
              Comparison families: connectome-derived architectures, topology controls that match
              the graph statistics without the biological wiring, and conventional artificial
              baselines. No claim is made about living animals, complete nervous systems, or
              consciousness.
            </p>
          </div>
        </div>
      </section>

      {/* HEADLINE RESULT */}
      <section className="page-x mx-auto max-w-[1600px]">
        <div className="grid gap-12 py-16 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:gap-20 md:py-24">
          <div>
            <div className="label-tech">Current headline result · T-002</div>
            <div className="tnum mt-6 flex items-baseline gap-3">
              <span className="text-[4.5rem] font-light leading-none tracking-tight md:text-[6rem]">
                <CountUp to={headline.mean} />
                <span className="text-3xl md:text-4xl">%</span>
              </span>
              <span className="font-mono text-sm text-muted-foreground">
                ± {headline.ci95 !== null ? headline.ci95.toFixed(1) : headline.sd.toFixed(1)}
              </span>
            </div>
            <p className="mt-5 max-w-md text-sm leading-relaxed text-muted-foreground">
              {ARCH_BY_ID[headline.architectureId]!.label} with SA-010 temporal dynamics on the
              delayed-recall task. Conventional baselines evaluated here reach{" "}
              <span className="tnum font-mono">
                {T002.results.find((r) => r.architectureId === "A7-MLP")?.mean.toFixed(1)}%
              </span>{" "}
              (MLP) and{" "}
              <span className="tnum font-mono">
                {T002.results.find((r) => r.architectureId === "A8-LSTM")?.mean.toFixed(1)}%
              </span>{" "}
              (LSTM).
            </p>
            <p
              className="mt-6 max-w-md border-l pl-4 text-[13px] leading-relaxed"
              style={{ borderColor: "var(--caveat)", color: "var(--caveat)" }}
            >
              Configuration selected via exploratory sensitivity analysis. Task-specific finding —
              not a universal biological conclusion.
            </p>
            <div className="mt-9 flex flex-wrap gap-6">
              {TASKS.map((t) => (
                <Link
                  key={t.id}
                  to="/experiment/$taskId"
                  params={{ taskId: t.id }}
                  className="label-tech border-b border-border-strong pb-1 transition-colors duration-200 hover:text-foreground"
                >
                  {t.code} detail
                </Link>
              ))}
            </div>
          </div>

          <div className="rule-t pt-10 md:border-t-0 md:pt-0">
            <div className="label-tech">The comparison</div>
            <SignatureMotif className="mt-6 w-full" />
          </div>
        </div>
      </section>
    </>
  );
}
