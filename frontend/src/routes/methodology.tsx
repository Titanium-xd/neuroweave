import { createFileRoute, Link } from "@tanstack/react-router";
import { Disclosure, Section } from "@/components/site/Disclosure";
import { SignatureMotif } from "@/components/site/SignatureMotif";
import { ARCHITECTURES, FAMILY_META, SOURCE, TASKS, type Family } from "@/data/benchmark";

export const Route = createFileRoute("/methodology")({
  head: () => ({
    meta: [
      { title: "Methodology — connectome-constrained model families | ABB" },
      {
        name: "description",
        content:
          "How ABB derives architectures from the MaleCNS v1.0 connectome, which topology controls isolate wiring effects, and how multi-seed evaluation and limitations are handled.",
      },
      { property: "og:title", content: "Methodology | ABB" },
      {
        property: "og:description",
        content:
          "Connectome source, model families, topology controls, tasks, multi-seed evaluation and limitations.",
      },
    ],
  }),
  component: Methodology,
});

function FamilyBlock({ family }: { family: Family }) {
  const items = ARCHITECTURES.filter((a) => a.family === family);
  return (
    <div className="border-t border-border py-6 first:border-t-0 first:pt-0">
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-[7px] w-[7px]"
          style={{ background: FAMILY_META[family].token }}
        />
        <span className="label-tech !text-foreground">{FAMILY_META[family].label}</span>
      </div>
      <ul className="mt-4 space-y-3">
        {items.map((a) => (
          <li key={a.id} className="grid gap-1 md:grid-cols-[130px_minmax(0,1fr)] md:gap-6">
            <span className="tnum font-mono text-[12.5px]">{a.label}</span>
            <span className="text-sm leading-relaxed text-muted-foreground">{a.description}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Methodology() {
  return (
    <div className="page-x mx-auto max-w-[1600px]">
      <header className="grid gap-10 py-16 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:gap-20 md:py-24">
        <div>
          <div className="label-tech">Methodology</div>
          <h1 className="mt-6 text-5xl font-light leading-[0.95] tracking-[-0.035em] md:text-[4.5rem]">
            From wiring
            <br />
            to benchmark
          </h1>
        </div>
        <div className="max-w-xl md:pt-14">
          <p className="text-[17px] leading-relaxed">
            ABB treats connectome topology as an architectural prior and asks whether that prior
            provides measurable computational advantage under controlled tasks.
          </p>
          <p className="mt-5 text-sm leading-relaxed text-muted-foreground">{SOURCE.disclaimer}</p>
        </div>
      </header>

      <Section code="01" title="Pipeline">
        <SignatureMotif className="w-full max-w-2xl" />
        <p className="mt-8 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          A prototype subgraph is extracted from the connectome and used as a fixed connectivity
          mask. The same task, data conditions and seed protocol are then applied to the
          connectome-derived architecture, to topology controls, and to conventional artificial
          baselines.
        </p>
      </Section>

      <Section code="02" title="Connectome source">
        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <div className="label-tech">Dataset</div>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              {SOURCE.dataset} — publicly released connectome of the {SOURCE.subject}.
            </p>
          </div>
          <div>
            <div className="label-tech">Derived artefact</div>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              A 150-node directed prototype subgraph, used as a binary connectivity mask over a
              trainable weight matrix. Explore it in the{" "}
              <Link to="/connectome" className="text-foreground underline underline-offset-4">
                connectome explorer
              </Link>
              .
            </p>
          </div>
        </div>
      </Section>

      <Section code="03" title="Model families">
        <div>
          {(["bio", "control", "ai", "baseline"] as Family[]).map((f) => (
            <FamilyBlock key={f} family={f} />
          ))}
        </div>
      </Section>

      <Section code="04" title="Tasks">
        <div>
          {TASKS.map((t) => (
            <Disclosure key={t.id} title={`${t.code} — ${t.title}`} meta={t.tagline}>
              <p>{t.description}</p>
              <p>
                <span className="text-foreground">Observation space.</span> {t.observationSpace}
              </p>
              <p>
                <span className="text-foreground">Reading.</span> {t.conclusion}
              </p>
              {t.caveat && <p style={{ color: "var(--caveat)" }}>{t.caveat}</p>}
              <p>
                <Link
                  to="/experiment/$taskId"
                  params={{ taskId: t.id }}
                  className="text-foreground underline underline-offset-4"
                >
                  Full experiment detail
                </Link>
              </p>
            </Disclosure>
          ))}
        </div>
      </Section>

      <Section code="05" title="Evaluation">
        <div>
          <Disclosure title="Multi-seed protocol" defaultOpen>
            <p>
              Each architecture is trained and evaluated independently per random seed under
              identical data conditions. The reported value is the mean accuracy; the reported
              spread is the standard deviation across seeds.
            </p>
          </Disclosure>
          <Disclosure title="Confidence intervals">
            <p>
              Formal 95% confidence intervals require the published seed count, which the current
              prototype run does not export. Until it does, the benchmark shows ±1 SD ranges and
              leaves the CI column empty rather than estimating one.
            </p>
          </Disclosure>
          <Disclosure title="Temporal dynamics (SA-010)">
            <p>
              Temporal behaviour of the connectome-constrained layer is parameterised by an
              activation decay and a leak term. The SA-010 setting (decay 0.05 / leak 0.02) is the
              confirmed configuration for T-002; it was located through exploratory sensitivity
              analysis and supersedes the earlier default (decay 0.15 / leak 0.08).
            </p>
          </Disclosure>
        </div>
      </Section>

      <Section code="06" title="Limitations">
        <ul className="max-w-2xl space-y-4 text-sm leading-relaxed text-muted-foreground">
          {[
            "The subgraph is a 150-node prototype. It is not the complete Drosophila nervous system and results should not be extrapolated to whole-connectome scale.",
            "Nothing in this project recreates a living animal, biological function, or consciousness. All models are computational.",
            "T-001 saturates for several architectures, so it cannot discriminate between them.",
            "T-002 spreads are wide relative to the differences between conditions; the strongest and weakest seeds differ substantially for some architectures.",
            "The SA-010 temporal configuration was selected through exploratory analysis on T-002 and has not been validated as a general setting.",
            "Conventional baselines were evaluated at their defaults without architecture-specific tuning.",
            "Parameter counts, training times and seed counts are not yet exported by the benchmark backend.",
          ].map((l) => (
            <li key={l} className="border-l border-border pl-4">
              {l}
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}
