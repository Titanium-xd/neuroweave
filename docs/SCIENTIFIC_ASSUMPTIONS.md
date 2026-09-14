# SCIENTIFIC_ASSUMPTIONS.md
# Animal Brain Benchmark — Scientific Assumptions Register

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

This document is a **living register** of every significant scientific assumption made in ABB.
Each entry records what was assumed, why, the biological grounding (or lack thereof), the
expected effect on results, and a unique identifier used in code comments and manifests.

Entries are classified by **confidence tier**:

| Tier | Meaning |
|---|---|
| **A — Established** | Strong experimental consensus; replicated across independent studies |
| **B — Plausible** | Some experimental support; minority dissent or context-dependence exists |
| **C — Engineering** | No direct biological evidence; adopted for computational tractability |
| **D — Speculative** | Adopted in the absence of data; results may not be biologically meaningful |

---

## Part I: Biological Facts (Directly From MaleCNS Data or Established Neuroscience)

These are not assumptions. They are listed here to draw a clear boundary.

### BF-001 — Neural Connectivity
**Fact:** MaleCNS v1.0 records ~125 million directed chemical synaptic connections among
~166,700 neurons from a single adult male *Drosophila melanogaster* specimen.
**Source:** MaleCNS v1.0 dataset; Janelia FlyEM *Cell* (2026).
**Qualification:** Reconstruction accuracy is high but not perfect; residual merge and split
errors exist (see DATA_PROVENANCE §5.1 and R-004).

### BF-002 — Synapse Polarity
**Fact:** Chemical synapses in *Drosophila* are directed; a presynaptic neuron releases
neurotransmitter onto one or more postsynaptic partners.
**Source:** Established neuroscience; consistent with connectome data format.

### BF-003 — Cell Type Annotation
**Fact:** MaleCNS v1.0 provides cell type labels for most neurons (~11,710 types).
These were assigned by expert human annotators based on morphology, projection patterns, and
connectivity fingerprints.
**Qualification:** Assignments involve expert judgment; consistency across the full dataset
is not guaranteed.

### BF-004 — Single Specimen
**Fact:** The entire MaleCNS connectome is from one individual fly. Biological variation
across individuals is not captured.
**Implication:** The "biological architecture" we test is one instantiation of the species
wiring plan, not the invariant species-level plan.

### BF-005 — Neurotransmitter Predictions (Computational Prediction, Not Physiological Measurement)
**Fact:** Neurotransmitter identity per neuron was computationally predicted by the
Eckstein et al. (2024, *Cell* 187(10):2574–2594.e23) 3D CNN classifier trained on EM
synaptic ultrastructure images.
**Corrected validation note (finalized 2026-09-14):** Accuracy figures from Eckstein et al.
vary by EM imaging modality:
- **FAFB/FlyWire (serial-section TEM):** ~87% per synapse, ~94% per neuron
- **Hemibrain (FIB-SEM):** ~78% per synapse, ~91% per neuron

MaleCNS v1.0 uses serial-section TEM (same modality as FAFB/FlyWire). The FAFB figures
are the closest available proxy. However, **no MaleCNS-specific validation has been
publicly reported**; MaleCNS-specific accuracy is CLOSED-AS-UNAVAILABLE from public sources.
**Qualification:** This is a computational prediction, not a physiological measurement.
Accuracy on specific ABB subgraphs may differ and is not separately validated.
Neuromodulators (5-HT, DA, Oct) are less reliably predicted than fast-acting NTs.

---

## Part II: Modeling Assumptions

### SA-001 — Synapse Count as Functional Weight (Tier B)

**What:** Synapse count (the number of physical synaptic contacts between two neurons) is
used as the initial proxy for connection strength in A1-BIO and A1-EDGE initialization.
**Why:** Synapse count is the only connection strength metric available in the MaleCNS
dataset. It correlates with total synaptic area, which correlates with postsynaptic current
amplitude (Rees et al., 2017, *Nature Neuroscience*).
**Biological grounding:** Partial — the correlation is well-established but not 1:1.
Individual synaptic strength varies substantially due to receptor density, vesicle release
probability, and receptor subtype composition.
**Normalization schemes tested (each is a separate ablation):**
- Raw count (A1-BIO default)
- Log-scaled count: log(1 + count)
- Out-degree normalized: count / total_out_degree(presynaptic_neuron)
- Binary adjacency: 1 if count > 0, else 0

**Expected effect on results:** Overweights hub neurons relative to biology. May inflate
importance of high-degree neurons.
**Code label:** `# [SA-001] ASSUMPTION: synapse_count_as_weight`

---

### SA-002 — Neurotransmitter Sign Assignment (Tier B/C)

**What:** A connection sign (+1 excitatory, −1 inhibitory, 0 neuromodulatory/unknown) is
assigned to each edge based on the computationally predicted neurotransmitter of the
presynaptic neuron.
**Why:** Signed weights are required for meaningful GNN message passing and for LIF models.
**Biological grounding:**

| Predicted NT | Sign Applied | Biological Justification | Confidence |
|---|---|---|---|
| Acetylcholine (ACh) | +1 | Dominant excitatory NT in insect CNS | High [BIO-FACT] |
| GABA | −1 | Dominant inhibitory NT | High [BIO-FACT] |
| Glutamate | −1 | Predominantly inhibitory in *Drosophila* CNS; context-dependent | Medium |
| Dopamine | 0 (or learnable) | Neuromodulator; effect is receptor-subtype-dependent, not annotated | Low [ASSUMPTION] |
| Serotonin | 0 (or learnable) | Neuromodulator | Low [ASSUMPTION] |
| Octopamine | 0 (or learnable) | Insect noradrenaline analogue | Low [ASSUMPTION] |
| Unknown / None | 0 (or learnable) | Insufficient information | Very Low [ASSUMPTION] |

**Validation caveat:** These sign rules apply to neurons where NT prediction is correct.
Using FAFB accuracy as proxy (same TEM modality as MaleCNS): ~87% per synapse / ~94%
per neuron. For ~6–13% of neurons with incorrect NT predictions, the sign applied will
be wrong. Hemibrain-condition accuracy was ~78% per synapse / ~91% per neuron (different
FIB-SEM modality; not directly applicable to MaleCNS).

**Ablation variants:**
- SA-002 (default): NT-classification sign rule as above
- SA-002-ALT: All edge signs are learnable parameters (no fixed rule)
- SA-002-NULL: All weights unsigned (no sign applied; absolute values only)
- SA-002-RAND: Signs randomly reassigned (implemented as control A5-WSIGN)

**Code label:** `# [SA-002] ASSUMPTION: nt_sign_rule`

---

### SA-003 — Uniform LIF Parameters Across Neurons (Tier C)

**What:** All neurons in LIF agents (A2 family) use identical biophysical parameters unless
cell-type-specific values are available from literature.
**Why:** Individual biophysical parameters for 166,700 neurons are not available in the
connectome dataset. Patch-clamp data exists for only a small fraction of identified
*Drosophila* neurons.
**Biological grounding:** None for the uniformity assumption. *Drosophila* neurons are known
to be highly diverse in their intrinsic properties.

**Default parameter values (all Tier C — engineering defaults):**

| Parameter | Default Value | Source of Default | Bio-Specific? |
|---|---|---|---|
| τ_m (membrane time constant) | 20 ms | Shiu et al. (Nature 2024) LIF model default | Weak — not Drosophila-specific measurement |
| V_th (threshold) | −50 mV | Conventional LIF default | No |
| V_rest (resting potential) | −70 mV | Conventional LIF default | No |
| V_reset (post-spike reset) | −80 mV | Conventional LIF default | No |
| τ_ref (refractory period) | 2 ms | Typical insect neuron estimate | Weak |

**Phase 3 plan:** Fit per-cell-type parameters from published *Drosophila* electrophysiology
literature for the subset of well-characterized cell types.
**Code label:** `# [SA-003] ASSUMPTION: uniform_lif_params`

---

### SA-004 — Static Synapse Weights During Inference (Tier C)

**What:** Synapse weights are fixed after the training phase. No on-line synaptic plasticity
occurs during evaluation.
**Biological grounding:** *Drosophila* synapses exhibit plasticity. We do not model it.
**Expected effect:** Cannot capture any learning that occurs during the task itself.
Learning in ABB is a pre-task gradient descent optimization phase, not a biological process.
**Code label:** `# [SA-004] ASSUMPTION: static_synapse_weights`

---

### SA-005 — Discrete-Time Simulation for LIF Agents (Tier C)

**What:** LIF simulation uses discrete time steps, dt = 1 ms default (A2 family).
**Biological grounding:** 1 ms captures most relevant LIF dynamics. Sub-millisecond
coincidence detection is not captured.
**Code label:** `# [SA-005] ASSUMPTION: discrete_time_dt_1ms`

---

### SA-006 — Subgraph Selection (Tier C)

**What:** ABB experiments use subgraphs of MaleCNS, not the full 166,700-neuron graph.
**Why:** Computational infeasibility of the full graph.
**Biological grounding:** No biological principle justifies the specific subgraph boundary.
**Constraints enforced:** See BENCHMARK_DESIGN.md §4 (connectivity requirements, 3-subgraph
variance reporting).
**Code label:** `# [SA-006] ASSUMPTION: subgraph_selection`

---

### SA-007-GNN — Input Encoding for GNN Agents (Tier C)

**What:** For GNN agents (A1 family, A3 family), the D_obs-dimensional observation vector is
projected into the input neurons via a learned linear projection matrix W_in
(shape: [n_input_neurons × D_obs]). W_in is initialized with an orthonormal random matrix
and is trainable in A1-BIAS, A1-EDGE, and A1-BOTH; frozen (fixed projection) in A1-BIO and
A1-FROZEN.
**Why:** GNN nodes require a node feature vector; there is no biologically established
protocol for projecting arbitrary task observations into specific neuron populations.
**Biological grounding:** Partial — the set of input neurons is designated based on cell type
annotations (e.g., known sensory projection neurons). The projection matrix itself is an
engineering choice.
**Note:** One observation step for GNN agents = one message-passing iteration (with T_mp
sub-steps per observation step; T_mp is a hyperparameter).
**Code label:** `# [SA-007-GNN] ASSUMPTION: gnn_input_encoding`

---

### SA-007-LIF — Input Encoding for LIF Agents (Tier C)

**What:** For LIF agents (A2 family), the D_obs-dimensional observation vector is encoded
as a Poisson spike train over T_sim discrete time steps (where T_sim ms = one observation
step). Each dimension of the observation maps to a firing rate for one designated input
neuron (rate encoding). T_sim is a hyperparameter (default: 50 ms).
**Why:** Poisson rate coding is a standard SNN input encoding method.
**Biological grounding:** Rate coding is one of several plausible neural codes used in
*Drosophila* sensory systems. It is not established as the only encoding and is adopted for
tractability.
**Alternative considered:** Temporal/rank-order coding (first spike timing). Reserved as
Phase 3 ablation.
**Note:** One observation step for LIF agents = T_sim ms of simulated biological time,
during which T_sim / dt = T_sim discrete simulation time steps occur.
**Code label:** `# [SA-007-LIF] ASSUMPTION: lif_input_encoding`

---

### SA-008 — GNN Message Passing as Approximation to Neural Dynamics (Tier C)

**What:** GNN agents (A1/A3 family) use differentiable message passing rather than LIF
spiking dynamics. Neuron activations are continuous-valued; the operator is a custom signed
sparse linear layer (see §SA-009-OPERATOR).
**Biological grounding:** Low — GNN message passing preserves the topology but loses
spike timing, refractory dynamics, and membrane potential history.
**Expected effect:** GNN approximation is expected to outperform LIF on task metrics due
to gradient tractability. This is not evidence that GNN better represents the biology.
**Code label:** `# [SA-008] ASSUMPTION: gnn_approximation`

---

### SA-009 — GNN Primary Operator: Custom Signed Sparse Linear (Tier C)

**What:** The primary GNN operator for A1 variants is a custom sparse linear message
passing layer where the adjacency matrix defines connectivity and each non-zero edge
has a scalar weight (trainable or fixed). No degree normalization is applied. No attention
mechanism is applied.

Mathematically, for neuron i at step t+1:
```
h_i(t+1) = activation( b_i + sum_{j → i} w_ij * h_j(t) )
```
where w_ij is the signed edge weight (product of synapse-count weight and NT sign under
SA-001 and SA-002) and b_i is per-neuron bias.

**Why custom operator:** Standard GCNConv normalizes by `D^{-1/2} A D^{-1/2}`, which
down-weights high-degree neurons — the opposite of what biology suggests. GATConv adds
per-edge learnable attention that overrides the synapse count prior with additional
parameters. Both corrupt the intended weight initialization from synapse counts.

**Ablation variants (explicitly labeled, not the primary condition):**
- SA-009-GCN: Use GCNConv normalization (ablation only; label `[SA-009-GCN]`)
- SA-009-GAT: Use GATConv attention (ablation only; label `[SA-009-GAT]`)

**Code label:** `# [SA-009] ASSUMPTION: custom_signed_sparse_linear`

---

### SA-010 — No Neuromodulation Model (Tier D)

**What:** Global neuromodulatory signals (dopamine, serotonin, octopamine) are not modeled.
**Biological grounding:** None for the omission. Neuromodulation is a core feature of fly behavior.
**Expected effect:** Many fly behaviors are strongly modulated. Phase 2+ may add a simplified
global modulation signal as a scalar context variable.
**Code label:** `# [SA-010] ASSUMPTION: no_neuromodulation`

---

### SA-011 — Task Selection Scope (Tier C)

**What:** Benchmark tasks are selected by ABB designers and are not derived from
experimental fly behavior protocols.
**Expectation:** The biological topology may have informative inductive biases for temporal
integration and sensorimotor tasks — but this is a hypothesis (PRQ-1), not an established fact.
**Important:** A null result on ABB tasks does NOT imply biological topology is computationally
uninformative for all tasks. It implies it is not informative for these specific tasks under
these specific assumptions.

---

### SA-012 — Multi-Axis Capacity Reporting (Tier C)

**What:** Architecture comparison uses Pareto frontiers across parameter counts and three
budget axes (sample, compute, wall-clock), rather than a single matched-parameter point.
**Why:** No single axis constitutes a "fair" comparison. Two architectures with identical
parameter counts may have radically different inductive biases, FLOPs, and practical utility.
**Limitation:** Even multi-axis comparison cannot fully equalize architectures whose
expressiveness is not captured by any single scalar measure.

---

## Part III: Summary Matrix (Revised)

| ID | Description | Tier | Biological Grounding | Impact If Wrong |
|---|---|---|---|---|
| SA-001 | Synapse count as weight | B | Partial | Overweights hub neurons |
| SA-002 | NT sign assignment | B/C | Partial | ~10–15% wrong-signed edges |
| SA-003 | Uniform LIF parameters | C | None | Obscures cell-type diversity |
| SA-004 | No inference-time plasticity | C | None | Cannot capture on-task learning |
| SA-005 | Discrete LIF time (1 ms) | C | Weak | Sub-ms phenomena not captured |
| SA-006 | Subgraph selection | C | None | Results subgraph-specific |
| SA-007-GNN | GNN input encoding | C | Partial | Input representation drives response |
| SA-007-LIF | LIF input encoding | C | Partial | Firing rate code may not be correct |
| SA-008 | GNN approximation | C | Low | Loses spiking dynamics |
| SA-009 | Custom signed sparse linear | C | Low | Operator choice affects expressivity |
| SA-010 | No neuromodulation | D | None | Major behavioral phenomena absent |
| SA-011 | Task selection scope | C | Partial | Tasks may miss biological advantage |
| SA-012 | Multi-axis capacity reporting | C | None | No single axis is fully fair |

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | HIGH-01: SA-009 added specifying custom signed sparse linear as primary operator, GCNConv/GATConv demoted to ablations; HIGH-02: BF-005 corrected to specify validation conditions and uncertainty explicitly; MED-02: SA-007 split into SA-007-GNN and SA-007-LIF with architecture-specific time step definitions; updated Summary Matrix |
| 2026-09-14 | ABB-0.1-rev1 (C4-final) | BF-005 accuracy attribution corrected: 87%/94% = FAFB/FlyWire (TEM); 78%/91% = Hemibrain (FIB-SEM). MaleCNS-specific accuracy CLOSED-AS-UNAVAILABLE. SA-002 caveat updated to use FAFB proxy figures. |
