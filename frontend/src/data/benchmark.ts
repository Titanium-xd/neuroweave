/**
 * ABB local data layer.
 *
 * Values are taken directly from the experiment artifact JSON files:
 *   artifacts/campaign_1/         — T-001, 10 architectures, 5 seeds each
 *   artifacts/confirmation_t002/  — T-002 confirmation, 5 architectures, 5 seeds
 *
 * Fields that are genuinely unavailable render as null → "—" in the UI.
 * Nothing is invented.
 */

export type Family = "bio" | "control" | "ai" | "baseline";

export interface Architecture {
  id: string;
  label: string;
  family: Family;
  /** Short description of what the architecture is. */
  description: string;
}

export interface Measurement {
  architectureId: string;
  /** Mean accuracy, percent. */
  mean: number;
  /** Standard deviation across seeds, percent. */
  sd: number;
  /** Trainable parameter count. */
  params: number | null;
  /** Frozen parameter count. */
  frozenParams: number | null;
  /** Wall-clock training time in seconds per seed (mean). */
  trainSeconds: number | null;
  /** Number of seeds used. */
  seeds: number | null;
  /** 95% confidence interval half-width (±). */
  ci95: number | null;
  /** Optional per-row methodological annotation. */
  note?: string;
  /** Variant qualifier, e.g. temporal dynamics setting. */
  variant?: string;
}

export interface Task {
  id: string;
  code: string;
  title: string;
  tagline: string;
  /** One-paragraph task description. */
  description: string;
  observationSpace: string;
  protocol: string;
  conclusion: string;
  caveat: string | null;
  results: Measurement[];
  assumptions: string[];
  notes: string[];
}

export const ARCHITECTURES: Architecture[] = [
  {
    id: "A0-RANDOM",
    label: "A0 Random",
    family: "baseline",
    description: "Untrained random-output reference establishing chance level.",
  },
  {
    id: "A1-BIO",
    label: "A1-BIO",
    family: "bio",
    description:
      "MaleCNS-derived architecture: connectivity mask taken from the connectome subgraph, weights trained.",
  },
  {
    id: "A1-FROZEN",
    label: "A1-FROZEN",
    family: "bio",
    description:
      "Same MaleCNS-derived topology with the connectome-constrained layer held frozen.",
  },
  {
    id: "A1-BIAS",
    label: "A1-BIAS",
    family: "bio",
    description: "MaleCNS-derived topology with bias terms enabled.",
  },
  {
    id: "A1-EDGE",
    label: "A1-EDGE",
    family: "bio",
    description: "MaleCNS-derived topology with edge-weight parameterisation.",
  },
  {
    id: "A3-ER",
    label: "A3-ER",
    family: "control",
    description: "Erdős–Rényi random graph control matched on node and edge count.",
  },
  {
    id: "A3-CONFIG",
    label: "A3-CONFIG",
    family: "control",
    description: "Configuration-model control preserving the degree sequence.",
  },
  {
    id: "A3-DENSE",
    label: "A3-DENSE",
    family: "control",
    description: "Fully connected control over the same node set.",
  },
  {
    id: "A7-MLP",
    label: "A7-MLP",
    family: "ai",
    description: "Conventional multilayer perceptron baseline.",
  },
  {
    id: "A8-LSTM",
    label: "A8-LSTM",
    family: "ai",
    description: "Conventional recurrent baseline (LSTM).",
  },
];

export const ARCH_BY_ID = Object.fromEntries(ARCHITECTURES.map((a) => [a.id, a]));

// ---------------------------------------------------------------------------
// T-001 — Binary Pattern Discrimination
// Source: artifacts/campaign_1/  (5 seeds each, 40 epochs)
// ---------------------------------------------------------------------------

export const T001: Task = {
  id: "t-001",
  code: "T-001",
  title: "Binary Pattern Discrimination",
  tagline: "Static pattern",
  description:
    "A static, globally observable binary classification task. The model receives a single pattern and must assign it to one of two classes. There is no temporal structure and no delay between stimulus and decision.",
  observationSpace: "Static binary pattern vector presented in a single step.",
  protocol:
    "Multi-seed evaluation (5 seeds, 40 epochs). Each architecture is trained and evaluated independently per seed; the reported value is the mean accuracy with the standard deviation across seeds. 95% CI computed via t-distribution.",
  conclusion:
    "Simple static global classification does not reveal a unique MaleCNS advantage — dense, random-graph and conventional baselines reach the same ceiling.",
  caveat: null,
  results: [
    {
      architectureId: "A0-RANDOM",
      mean: 52.4, sd: 5.7, ci95: 7.9,
      params: 0, frozenParams: 0, trainSeconds: 0.0, seeds: 5,
    },
    {
      architectureId: "A1-BIO",
      mean: 97.4, sd: 0.8, ci95: 1.1,
      params: 13714, frozenParams: 0, trainSeconds: 3.2, seeds: 5,
    },
    {
      architectureId: "A1-FROZEN",
      mean: 97.8, sd: 1.6, ci95: 2.2,
      params: 10052, frozenParams: 3662, trainSeconds: 1.7, seeds: 5,
    },
    {
      architectureId: "A1-BIAS",
      mean: 97.8, sd: 1.7, ci95: 2.4,
      params: 10352, frozenParams: 3362, trainSeconds: 1.8, seeds: 5,
    },
    {
      architectureId: "A1-EDGE",
      mean: 97.4, sd: 0.8, ci95: 1.1,
      params: 13414, frozenParams: 300, trainSeconds: 3.0, seeds: 5,
    },
    {
      architectureId: "A3-ER",
      mean: 99.4, sd: 0.8, ci95: 1.1,
      params: 13714, frozenParams: 0, trainSeconds: 3.1, seeds: 5,
    },
    {
      architectureId: "A3-CONFIG",
      mean: 98.2, sd: 1.2, ci95: 1.6,
      params: 13714, frozenParams: 0, trainSeconds: 3.1, seeds: 5,
    },
    {
      architectureId: "A3-DENSE",
      mean: 100.0, sd: 0.0, ci95: 0.0,
      params: 55352, frozenParams: 0, trainSeconds: 1.3, seeds: 5,
    },
    {
      architectureId: "A7-MLP",
      mean: 100.0, sd: 0.0, ci95: 0.0,
      params: 41602, frozenParams: 0, trainSeconds: 1.2, seeds: 5,
    },
    {
      architectureId: "A8-LSTM",
      mean: 100.0, sd: 0.0, ci95: 0.0,
      params: 272770, frozenParams: 0, trainSeconds: 2.9, seeds: 5,
    },
  ],
  assumptions: [
    "The connectome-derived connectivity mask is fixed for the duration of a run.",
    "All architectures see identical data splits per seed.",
    "Accuracy is measured on held-out patterns.",
  ],
  notes: [
    "Because several architectures saturate at or near 100%, this task is used as a sanity check rather than a discriminative benchmark.",
    "Standard deviation of 0.0% indicates identical accuracy across all evaluated seeds.",
    "A3-ER and A3-CONFIG match MaleCNS on static tasks, confirming T-001 tests global density rather than specific biological wiring.",
  ],
};

// ---------------------------------------------------------------------------
// T-002 — Temporal Sequence Memory
// Source: artifacts/confirmation_t002/  (5 seeds each, 30 epochs)
// Candidate config (0.05/0.02) selected from exploratory sensitivity sweep.
// ---------------------------------------------------------------------------

export const T002: Task = {
  id: "t-002",
  code: "T-002",
  title: "Temporal Sequence Memory",
  tagline: "Temporal memory",
  description:
    "A delayed-recall task with temporal structure: information presented early in a sequence must be retained across intervening steps before a decision is required. This probes whether a computational architecture supports useful state retention rather than instantaneous mapping.",
  observationSpace: "Sequence of step-wise observations with a delay before the decision step.",
  protocol:
    "Multi-seed evaluation (5 seeds, 30 epochs) of the final confirmed configuration. Temporal dynamics are parameterised by an activation decay and a leak term applied to the connectome-constrained layer. 95% CI computed via t-distribution.",
  conclusion:
    "The MaleCNS-derived architecture with SA-010 temporal dynamics shows strong performance on this specific delayed-recall task, while the conventional baselines evaluated here do not.",
  caveat:
    "The SA-010 decay/leak setting (decay 0.05 / leak 0.02) was selected through exploratory sensitivity analysis on this task. It should be read as a task-specific configuration finding, not as a universal biological conclusion. Confidence intervals are wide relative to the differences between conditions.",
  results: [
    {
      architectureId: "A0-RANDOM",
      mean: 50.6, sd: 4.8, ci95: 6.6,
      params: 0, frozenParams: 0, trainSeconds: 0.0, seeds: 5,
    },
    {
      architectureId: "A1-BIO",
      mean: 94.8, sd: 4.4, ci95: 6.2,
      params: 6514, frozenParams: 0, trainSeconds: 18.6, seeds: 5,
      variant: "SA-010 · decay 0.05 / leak 0.02",
    },
    {
      architectureId: "A1-FROZEN",
      mean: 92.4, sd: 3.9, ci95: 5.5,
      params: 2852, frozenParams: 3662, trainSeconds: 8.1, seeds: 5,
      variant: "SA-010 · decay 0.05 / leak 0.02",
    },
    {
      architectureId: "A1-BIO",
      mean: 74.0, sd: 17.3, ci95: 24.0,
      params: 6514, frozenParams: 0, trainSeconds: 15.9, seeds: 5,
      variant: "previous default · decay 0.15 / leak 0.08",
      note: "Superseded configuration, retained for comparison. Very large spread across seeds.",
    },
    {
      architectureId: "A7-MLP",
      mean: 47.4, sd: 2.3, ci95: 3.2,
      params: 35458, frozenParams: 0, trainSeconds: 1.5, seeds: 5,
    },
    {
      architectureId: "A8-LSTM",
      mean: 57.6, sd: 19.3, ci95: 26.8,
      params: 266626, frozenParams: 0, trainSeconds: 5.5, seeds: 5,
      note: "Very large spread across seeds; unstable on this task configuration.",
    },
  ],
  assumptions: [
    "Temporal dynamics are applied uniformly across the connectome-constrained layer.",
    "The delay interval is identical for every architecture.",
    "No architecture-specific hyperparameter search was performed for the conventional baselines beyond their defaults.",
  ],
  notes: [
    "Sensitivity analysis over decay/leak was exploratory; the reported SA-010 setting is the confirmed final configuration.",
    "The spread on A8-LSTM and on the superseded A1-BIO configuration is large enough that seed-level variation dominates.",
    "A1-FROZEN achieves 92.4% with 2,852 trainable parameters vs. A8-LSTM at 57.6% with 266,626 — a 40× parameter efficiency difference on this task.",
  ],
};

export const TASKS: Task[] = [T001, T002];
export const TASK_BY_ID = Object.fromEntries(TASKS.map((t) => [t.id, t]));

export const FAMILY_META: Record<Family, { label: string; token: string; short: string }> = {
  bio: { label: "Bio-constrained", token: "var(--bio)", short: "BIO" },
  control: { label: "Topology controls", token: "var(--control)", short: "CTRL" },
  ai: { label: "Conventional AI", token: "var(--ai)", short: "AI" },
  baseline: { label: "Chance reference", token: "var(--border-strong)", short: "REF" },
};

export const AT_A_GLANCE = [
  { value: "MaleCNS", unit: "v1.0", label: "Connectome source" },
  { value: "150", unit: "nodes", label: "Prototype subgraph" },
  { value: "2", unit: "tasks", label: "Confirmed benchmarks" },
  { value: "10", unit: "architectures", label: "Bio, control, conventional" },
];

export const SOURCE = {
  dataset: "MaleCNS v1.0",
  subject: "adult male Drosophila central nervous system",
  disclaimer:
    "ABB models are computational. Nothing here recreates a living animal, a complete nervous system, or consciousness.",
};
