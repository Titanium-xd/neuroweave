/**
 * ABB signature motif — the flow from connectome to benchmark to two
 * model families. Pure SVG, typographic, no illustration.
 */
export function SignatureMotif({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 640 340"
      className={className}
      role="img"
      aria-label="Flow diagram: biological connectome to MaleCNS-derived model, evaluated by benchmark against biological topology and conventional AI models"
    >
      <defs>
        <marker id="abb-arrow" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M1 1 L7 4 L1 7 z" fill="var(--border-strong)" />
        </marker>
      </defs>
      <g
        fontFamily="var(--font-mono)"
        fontSize="10.5"
        letterSpacing="1.6"
        textAnchor="middle"
        fill="var(--muted-foreground)"
      >
        <text x="320" y="26" fill="var(--foreground)">
          BIOLOGICAL CONNECTOME
        </text>
        <text x="320" y="46" fontSize="9">
          MALECNS V1.0 · DROSOPHILA CNS
        </text>

        <text x="320" y="116" fill="var(--bio)">
          MALECNS-DERIVED ARCHITECTURE
        </text>
        <text x="320" y="136" fontSize="9">
          CONNECTOME-TOPOLOGY-CONSTRAINED
        </text>

        <text x="320" y="206" fill="var(--foreground)">
          CONTROLLED BENCHMARK
        </text>
        <text x="320" y="226" fontSize="9">
          T-001 · T-002 · MULTI-SEED
        </text>

        <text x="150" y="312" fill="var(--bio)">
          BIOLOGICAL TOPOLOGY
        </text>
        <text x="490" y="312" fill="var(--ai)">
          CONVENTIONAL AI MODELS
        </text>
      </g>
      <g stroke="var(--border-strong)" strokeWidth="1" fill="none" markerEnd="url(#abb-arrow)">
        <path d="M320 58 V 96" />
        <path d="M320 148 V 186" />
        <path d="M320 238 V 258" markerEnd="none" />
        <path d="M150 258 H 490" markerEnd="none" />
        <path d="M150 258 V 292" />
        <path d="M490 258 V 292" />
      </g>
      <g fill="var(--border-strong)">
        <circle cx="320" cy="258" r="2" />
      </g>
    </svg>
  );
}
