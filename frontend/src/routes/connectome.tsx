import { createFileRoute } from "@tanstack/react-router";
import { ConnectomeExplorer } from "@/components/site/ConnectomeExplorer";

export const Route = createFileRoute("/connectome")({
  head: () => ({
    meta: [
      { title: "Connectome Explorer — MaleCNS v1.0-derived subgraph | ABB" },
      {
        name: "description",
        content:
          "Interactive exploration of the 150-node MaleCNS v1.0-derived prototype subgraph: neurons, directed edges, activity state and graph statistics.",
      },
      { property: "og:title", content: "Connectome Explorer | ABB" },
      {
        property: "og:description",
        content:
          "Neurons, directed edges and activity propagation in the 150-node MaleCNS v1.0-derived prototype subgraph.",
      },
    ],
  }),
  component: ConnectomePage,
});

function ConnectomePage() {
  return (
    <div className="page-x mx-auto max-w-[1600px]">
      <header className="grid gap-10 py-16 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:gap-20 md:py-24">
        <div>
          <div className="label-tech">Connectome explorer</div>
          <h1 className="mt-6 text-5xl font-light leading-[0.95] tracking-[-0.035em] md:text-[4.5rem]">
            Sparse
            <br />
            directed
            <br />
            topology
          </h1>
        </div>
        <div className="max-w-xl md:pt-14">
          <p className="text-[17px] leading-relaxed">
            The 150-node MaleCNS v1.0-derived prototype subgraph used as the connectivity mask in
            the benchmark.
          </p>
          <p className="mt-5 text-sm leading-relaxed text-muted-foreground">
            This is a prototype subgraph, not the complete fly nervous system. Node coordinates are
            an illustrative layout for display, not anatomical positions. Activity and stimulation
            markers show the demonstration state of the graph.
          </p>
        </div>
      </header>

      <div className="pb-6">
        <ConnectomeExplorer />
      </div>

      <div className="grid gap-8 py-14 md:grid-cols-3 md:gap-16">
        {[
          {
            t: "Directed edges",
            b: "Each edge carries signal in one direction. Hovering a neuron highlights only its incident edges, making local fan-in and fan-out legible.",
          },
          {
            t: "Activity state",
            b: "Active neurons are drawn in moss; stimulated neurons carry a square marker. Signal animations travel along a sampled subset of edges to indicate flow direction.",
          },
          {
            t: "Why it matters",
            b: "The mask's sparsity and degree structure are what the topology controls (A3-ER, A3-CONFIG, A3-DENSE) are designed to isolate.",
          },
        ].map((c) => (
          <div key={c.t}>
            <div className="label-tech">{c.t}</div>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{c.b}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
