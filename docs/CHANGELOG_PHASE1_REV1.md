# CHANGELOG_PHASE1_REV1.md
# Animal Brain Benchmark — Phase 1 Revision Change Log

> **Revision:** ABB-0.1 → ABB-0.1-rev1
> **Date:** 2026-09-14
> **Trigger:** Adversarial design review identifying 8 CRITICAL, 6 HIGH, 7 MEDIUM, 4 LOW issues
> **Review document:** `design_review.md` (artifact)

This document records every substantive change made during the ABB-0.1-rev1 revision.
All changes to source documents are traceable to a specific review finding (CRIT-XX,
HIGH-XX, MED-XX, or LOW-XX).

---

## Document: PROJECT_SCOPE.md

### Change 1 — Rewrote Primary Research Question (CRIT-01)
**Finding:** The original research question ("How useful is a real biological neural
connectivity architecture?") was a category error. A1 tests a GNN with a fixed sparsity
pattern, not "the biological architecture." The wording invited conclusions far stronger
than the evidence supports.

**Change:** Replaced single vague question with five precise sub-questions (PRQ-1 through
PRQ-5), each explicitly naming what is being tested and which experimental conditions
answer it. PRQ-1 (inductive bias via gradient training) and PRQ-2 (structural prior without
training) are now explicitly separated. The vocabulary "inductive bias" and "structural
prior" are defined with distinct technical meanings.

### Change 2 — Expanded Architecture Matrix with Required Controls (CRIT-02, CRIT-08)
**Finding:** Erdős–Rényi and Barabási–Albert controls were insufficient to isolate specific
structural hypotheses. Six critical baselines were missing entirely.

**Change:** Architecture matrix expanded from 10 entries to a structured family system:

**Added — Biological family:**
- A1-BIO: fixed synapse-count weights, frozen (the most data-direct condition)
- A1-FROZEN: fixed random weights, frozen (topology-only prior)
- A1-BIAS: trainable bias vectors only (~N parameters)
- A1-EDGE: trainable edge weights (~E parameters)
- A1-BOTH: trainable edges + biases (~N+E parameters)

**Added — Structural control family (A3/A5 expanded):**
- A3-ER: Erdős–Rényi (retained; reclassified from "A3")
- A3-CONFIG: Configuration model — exact per-node degree sequence (NEW; critical for PRQ-3)
- A3-SBM: Stochastic block model — community structure (NEW)
- A3-DENSE: Fully-connected GNN — no sparsity prior (NEW; critical for isolating sparsity)
- A5-WSHUFFLE: Shuffled weight magnitudes (retained; reclassified from "A5")
- A5-WSIGN: Shuffled NT signs (NEW; isolates sign structure)
- A5-ETARGET: Shuffled edge targets (NEW; isolates weight distribution vs. topology)

**Added — Required trivial baselines:**
- A0: Random policy (absolute lower bound; was missing)

**Removed:** Barabási–Albert (A4) superseded by A3-CONFIG which is a more precise control.

**Added — LIF family (Phase 3):**
- A2-BIO: LIF with frozen synapse-count weights
- A2-TRAIN: LIF with surrogate gradient training

### Change 3 — Replaced Single-Axis Fairness with Three-Axis Protocol (CRIT-03, CRIT-04)
**Finding:** (1) Parameter-budget matching collapses to a single uninterpretable point.
(2) "Same training steps" gives hidden compute advantage to fast architectures.

**Change:** §5 "Fairness Protocol" completely rewritten. Defines three independent comparison
axes (sample budget, compute budget, wall-clock budget) all reported simultaneously.
Performance reported as Pareto frontier over parameter counts, not single matched-budget point.
Universal Adam lr=1e-3 default removed; replaced with per-architecture hyperparameter search.

### Change 4 — Corrected A6 Phase Assignment (MED-01)
**Finding:** PROJECT_SCOPE.md §6 listed A6 in Phase 2; ROADMAP.md listed it in Phase 3.
Contradiction between documents.

**Change:** A6 moved to Phase 3 in PROJECT_SCOPE.md to match ROADMAP.md. Rationale:
A6 (simple LIF SNN) shares a simulation backend with A2-family and is correctly developed
together with LIF infrastructure.

### Change 5 — Standardized Vocabulary (LOW-01)
**Finding:** "Inductive bias" (ML term) and "structural prior" (Bayesian term) used
interchangeably throughout. They have different technical meanings.

**Change:** Both terms defined explicitly in PROJECT_SCOPE.md §3. "Inductive bias" used
only when weights are optimized by gradient descent. "Structural prior" used only for
fixed-weight conditions.

---

## Document: BENCHMARK_DESIGN.md

### Change 6 — Fixed Task Observation Dimension (CRIT-05)
**Finding:** T-001 specified "input vector of length N where N = number of input neurons."
This entangles the task definition with the subgraph size, making results non-reproducible
across subgraph scales.

**Change:** Every task now specifies a fixed D_obs constant:
- T-001: D_obs = 64
- T-002: D_obs = 16
- T-003-GRID: D_obs = 32
- T-003-CONT: D_obs = 8

The biological agent projects D_obs into its input neurons via W_in — an architectural
detail, not a task detail. This makes the task identical across all architectures and
subgraph sizes.

### Change 7 — Redesigned T-002 (Sequence Memory) (CRIT-06)
**Finding:** (1) "Time step" was undefined: 1 observation step = 1 ms for LIF but
dimensionless for GNN/LSTM, making cross-architecture results incomparable. (2) The task
could be trivially solved by storing the full observation history. (3) Missing the
capacity-limited buffer baseline.

**Change:**
- Observation steps are now architecture-independent (one call to observe/step/act)
- Each architecture's internal time step mapping is documented in SA-007-GNN and SA-007-LIF
- Task redesigned around a CAPACITY CONSTRAINT: the agent must compress the observation
  history; unrestricted history buffer is prohibited
- M-006-BUFFER baseline added: a sliding buffer of fixed capacity C, trivially achieves
  K ≤ C delay

### Change 8 — Split T-004 into Three Sub-Tasks (CRIT-07)
**Finding:** T-004 "node removal" had completely different semantics across architectures.
Cell-type-based lesioning was inapplicable to MLP/LSTM. The single lesioning task compared
incomparable quantities across architectures.

**Change:** T-004 split into:
- **T-004a** (Post-Training Node Ablation): Measures trained-solution sensitivity to node
  removal. Only "random" and "activation variance" strategies are cross-architecture
  comparable. Degree-based removal is graph-model only, reported separately.
- **T-004b** (Robustness-Trained Lesioning): Retrains with dropout; tests whether biological
  topology enables more distributed representations.
- **T-004c** (Interface Damage Robustness): Damages the D_obs input vector and output
  signal — equally applicable to ALL architectures regardless of internal structure.

Prohibited language: T-004 results must never be described as "biological robustness to
cellular damage."

### Change 9 — Added T-003-CONT; Supplemented T-003-GRID (HIGH-06)
**Finding:** T-003-GRID (discrete grid, compass observations) has mild spatial structure
biased toward recurrent architectures and is biologically unmotivated for VNC circuits.

**Change:** T-003-CONT added (Phase 3): a continuous sensorimotor regulation task analogous
to VNC motor circuit function. T-003-GRID retained (Phase 2) with explicit note that it
is NOT applied to CNN architectures. Task selection criterion updated to require "observation/
action neutrality" with explicit notes on when that criterion is not met.

### Change 10 — Replaced Fixed Optimizer with Hyperparameter Search Protocol (HIGH-05)
**Finding:** Universal Adam lr=1e-3 biases architectures that are well-conditioned at that
rate. GNN architectures at scale require a very different lr regime than small MLPs.

**Change:** §3.2 "Hyperparameter Search Protocol" added. 20 random search trials per
architecture per task. Search uses a held-out development split. The best hyperparameters
from search are stored in the manifest and used for all evaluation runs. Search budget
counts against the architecture's sample budget.

### Change 11 — Replaced Type-Stratified Subgraph with Connectivity-Respecting Methods (HIGH-04)
**Finding:** Type-stratified random sampling produces disconnected induced subgraphs where
most neurons have near-zero within-subgraph degree. A disconnected subgraph is not testing
the MaleCNS topology.

**Change:** §4.1 replaced with connectivity-respecting methods: K-hop neighborhood, random
walk sample, spectral sample, full input-output pathway. Type-stratified random sampling
demoted to secondary use only (cross-subgraph variance analysis). Mandatory connectivity
statistics added (largest CC fraction ≥ 0.90, mean in/out-degree ≥ 3.0, etc.).

### Change 12 — Fixed M-001 Task Applicability; Added M-007 AULC (HIGH-03, LOW-04)
**Finding:** M-001 (mean episode reward) listed T-002 (supervised) as an applicable task.
M-007 ("steps to 80% of final performance") had a circular dependency: requires final
performance to compute, which varies across architectures making the metric incomparable.

**Change:** M-001 applicability corrected to T-003-GRID and T-003-CONT only. M-007
replaced with AULC (Area Under Learning Curve), normalized to [0,1] using task-specific
floor and ceiling values. AULC is directly comparable across architectures.

### Change 13 — Increased RL Seeds to 10; Bootstrap CI (MED-03)
**Finding:** 5 seeds for RL tasks produces wide uninformative CIs with likely non-overlapping
means that are statistical artifacts. Student t-interval CI is invalid for non-normal
reward distributions.

**Change:** RL tasks (T-003-GRID, T-003-CONT) now require 10 seeds minimum. All tasks use
bootstrap 95% CI with B=2000 samples instead of t-interval.

### Change 14 — Expanded Manifest Schema to v1.1 (MED-04)
**Finding:** Original manifest lacked critical biological-model-specific fields: GNN operator
type, number of message-passing steps, weight initialization scheme, trainable parameter
spec, sign rule, input projection method, subgraph connectivity statistics, cross-subgraph
run tracking.

**Change:** Manifest version bumped to 1.1. Added: `biological_model_config` block,
`subgraph_connectivity` stats block, `subgraph_runs` cross-subgraph tracking, `hardware`
block, full `software` version block, `budget_axis_used` field, `hparam_search_log`.

### Change 15 — Added Benchmark Versioning Specification (LOW-03)
**Finding:** RISKS.md referenced a "BENCHMARK_DESIGN §version system" that did not exist.

**Change:** §1.4 "Benchmark Versioning" added to BENCHMARK_DESIGN.md. Defines
MAJOR.MINOR.revision semantics, tag requirements, and migration guide policy for
MAJOR version bumps.

### Change 16 — Added Required Trivial Baselines Table (CRIT-08)
**Finding:** Random policy (A0) was missing. No systematic list of trivial baselines required.

**Change:** §2.4 "Trivial Baselines" table added. Requires A0 (random policy),
T-001-CHANCE (majority class), T-002-BUFFER (buffer ceiling), T-003-STATIC (always-stay)
to be evaluated and recorded in every relevant experiment report.

---

## Document: SCIENTIFIC_ASSUMPTIONS.md

### Change 17 — Split SA-007 into GNN and LIF Variants (MED-02)
**Finding:** SA-007 simultaneously specified "inject currents" (LIF-specific) and named
"Poisson spike trains" as the Phase 2 default. Both cannot be the default. Current injection
has no meaning for GNN agents.

**Change:** SA-007 split into:
- **SA-007-GNN:** Input encoding for GNN agents — learned linear projection W_in from D_obs
  into input neurons; one observation step = T_mp message-passing sub-steps
- **SA-007-LIF:** Input encoding for LIF agents — Poisson rate coding over T_sim ms;
  one observation step = T_sim ms of simulated biological time

Both are labeled separately in code and manifests.

### Change 18 — Added SA-009 for Custom GNN Operator (HIGH-01)
**Finding:** GCNConv and GATConv were listed as acceptable primary operators for A1,
but GCNConv's degree normalization contradicts the synapse count weight prior and
GATConv adds parameters that override it.

**Change:** SA-009 added: the primary operator is specified as a custom signed sparse
linear layer with no normalization and no attention. GCNConv and GATConv are explicitly
demoted to labeled ablation variants (SA-009-GCN, SA-009-GAT).

### Change 19 — Corrected BF-005 NT Accuracy Statement (HIGH-02)
**Finding:** BF-005 cited "~87% per synapse, ~90%+ per neuron" as if this were a validated
fact for MaleCNS v1.0. These figures came from Hemibrain evaluations and may not transfer.

**Change:** BF-005 rewritten to specify:
- Accuracy figures are from Hemibrain v1.x validation conditions, not MaleCNS v1.0
- The exact classifier and validation methodology for MaleCNS v1.0 must be confirmed from
  the *Cell* (2026) methods section before being cited
- Accuracy on ABB-specific subgraphs is unvalidated and may differ
- The statement is labeled "current best estimate (unconfirmed for MaleCNS v1.0)"

---

## Document: DATA_PROVENANCE.md

### Change 20 — Corrected NT Accuracy Text (HIGH-02)
**Finding:** §4 cited ~87% accuracy as a fact without validation context.

**Change:** §4.1 "Neurotransmitter Accuracy — Precise Statement" added with IMPORTANT alert.
Language updated to match corrected BF-005 in SCIENTIFIC_ASSUMPTIONS.md.

### Change 21 — Fixed neuPrint Token Code Example (MED-06)
**Finding:** Code used `os.environ['NEUPRINT_TOKEN']` which raises a confusing KeyError if
unset. Text included "or a secrets manager" which opened the door to patterns that could
log the token.

**Change:** Code example replaced with `os.environ.get()` pattern and an explicit
`RuntimeError` with a helpful user-facing message and PowerShell/bash syntax. "Secrets
manager" option removed; environment variable is the only permitted method.

### Change 22 — Corrected Reconstruction Error Statement (MED-05)
**Finding:** §5.1 mitigation stated "errors affect the biological model uniformly, so
relative comparisons are still valid." This is incorrect: errors affect only biological
models (A1, A2), not synthetic controls (A3–A10), creating an asymmetric confound.

**Change:** §5.1 rewritten to describe the asymmetric confound. The corrected statement:
reconstruction errors create a confound where any observed disadvantage of A1/A2 could
partly reflect reconstruction noise rather than topology properties. This cannot be resolved
without error-corrected data and must be stated as a fundamental limitation.

---

## Document: ARCHITECTURE.md

### Change 23 — Specified Custom Signed Sparse Linear Operator (HIGH-01)
**Finding:** §5.1 listed "GCNConv / GATConv / custom signed-weight linear layer
(config-selectable)" as equally valid primary operators. GCNConv and GATConv corrupt the
weight initialization prior from biology.

**Change:** §4 added with full mathematical specification of the custom signed sparse linear
operator. §4.2 explicitly lists GCNConv and GATConv as labeled ablation variants only.
§4.3 specifies weight initialization per A1 variant (A1-BIO through A1-BOTH).

### Change 24 — Expanded File Layout for A1 Variants (MED-07)
**Finding:** `malecns_gnn.py` was listed as a single file for all A1 variants. "Trainable
parameters: Edge weights + per-neuron bias" was described ambiguously (four orders of
magnitude difference between the two options).

**Change:** File layout expanded to show five separate files for A1-BIO, A1-FROZEN,
A1-BIAS, A1-EDGE, A1-BOTH. `abb/operators/signed_sparse_linear.py` added as a dedicated
module. Each variant's parameter count and trainability specified explicitly in §4.3.

### Change 25 — Moved Docker to Phase 2 (LOW-02)
**Finding:** Docker containers listed as Phase 4 (publication) deliverable. Phase 2/3
results cannot be independently reproduced without a containerized environment.

**Change:** `docker/Dockerfile` added to repository layout. Phase 2 deliverable 2.2 is
"Docker image built and tested." §8 dependencies table updated. Design decision table
updated with rationale.

### Change 26 — Added `flops_per_step` to AbstractAgent Interface
**Finding:** Three-axis budget comparison (CRIT-04) requires tracking FLOPs per training
step. AbstractAgent had no mechanism to report this.

**Change:** `flops_per_step: int` abstract property added to AbstractAgent.
Manifest v1.1 includes `flops_per_step_estimate` field per architecture.

---

## Document: RISKS.md

### Change 27 — Corrected R-004 Reconstruction Error Mitigation (MED-05)
**Finding:** R-004 mitigation item 4 stated "reconstruction errors affect the biological
model uniformly across all experiments, so relative comparisons are still valid." This is
factually incorrect.

**Change:** Mitigation item 4 replaced with a CAUTION alert describing the asymmetric
confound. The incorrect claim is explicitly noted as removed and corrected.

### Change 28 — Updated R-007, R-008, R-012 (CRIT-03, CRIT-08, MED-05)
**Change:**
- R-007 (parameter matching): Updated to reference three-axis protocol as the primary mitigation
- R-008 (training not biologically realistic): Updated to reference A1-BIO/A1-FROZEN
  fixed-weight conditions as the fixed-weight comparison
- R-012 (overclaiming): Mitigation updated to explicitly name prohibited language about
  A1-BIO and T-004 results

### Change 29 — Added R-015: Custom Operator Performance Risk
**Change:** New risk R-015 added documenting the possibility that the custom signed sparse
linear operator is slower than standard PyG kernels. Mitigation includes benchmarking,
kernel optimization, and a documented fallback.

---

## Document: ROADMAP.md

### Change 30 — Corrected A6 Phase Assignment (MED-01)
**Finding:** Phase 2 delivery list included A6 as a Phase 2 deliverable, contradicting
both its placement in Phase 3 in the ROADMAP table and the rational that it shares
infrastructure with A2 (LIF simulation).

**Change:** A6 correctly placed in Phase 3 throughout ROADMAP.md.

### Change 31 — Moved Docker to Phase 2 Deliverables (LOW-02)
**Change:** Deliverable 2.2 added: "Docker image — built and tested." Removed from Phase 4.

### Change 32 — Expanded Phase 2 Deliverables
**Change:** Phase 2 model deliverables expanded from 5 entries to 15 to reflect full
A1-family (5 variants), full A3/A5 control set (7 models), A0 random policy. Phase 3
deliverables updated to include T-002, T-003-CONT, T-004a/b/c. Phase 1 deliverables
updated to include design review documents (1.8, 1.9) and NT accuracy verification (1.13).

---

## Document: README.md

### Change 33 — Updated Architecture Matrix and Research Questions
**Change:** README updated to reflect revised architecture families, task D_obs values,
three-axis fairness protocol, and precise research questions. Prohibited claims stated
explicitly. A1-BIO described as "most data-direct, least-trained condition" not simulation.

---

## Summary: Issues Addressed

| Issue ID | Severity | Status | Documents Changed |
|---|---|---|---|
| CRIT-01 | CRITICAL | ✅ Fixed | PROJECT_SCOPE.md, README.md |
| CRIT-02 | CRITICAL | ✅ Fixed | PROJECT_SCOPE.md, BENCHMARK_DESIGN.md, ROADMAP.md, README.md |
| CRIT-03 | CRITICAL | ✅ Fixed | PROJECT_SCOPE.md, BENCHMARK_DESIGN.md, ARCHITECTURE.md |
| CRIT-04 | CRITICAL | ✅ Fixed | PROJECT_SCOPE.md, BENCHMARK_DESIGN.md, ARCHITECTURE.md |
| CRIT-05 | CRITICAL | ✅ Fixed | BENCHMARK_DESIGN.md, PROJECT_SCOPE.md, SCIENTIFIC_ASSUMPTIONS.md |
| CRIT-06 | CRITICAL | ✅ Fixed | BENCHMARK_DESIGN.md, SCIENTIFIC_ASSUMPTIONS.md |
| CRIT-07 | CRITICAL | ✅ Fixed | BENCHMARK_DESIGN.md, PROJECT_SCOPE.md |
| CRIT-08 | CRITICAL | ✅ Fixed | PROJECT_SCOPE.md, BENCHMARK_DESIGN.md, ROADMAP.md |
| HIGH-01 | HIGH | ✅ Fixed | ARCHITECTURE.md, SCIENTIFIC_ASSUMPTIONS.md |
| HIGH-02 | HIGH | ✅ Fixed | SCIENTIFIC_ASSUMPTIONS.md, DATA_PROVENANCE.md |
| HIGH-03 | HIGH | ✅ Fixed | BENCHMARK_DESIGN.md |
| HIGH-04 | HIGH | ✅ Fixed | BENCHMARK_DESIGN.md |
| HIGH-05 | HIGH | ✅ Fixed | BENCHMARK_DESIGN.md, PROJECT_SCOPE.md |
| HIGH-06 | HIGH | ✅ Fixed | BENCHMARK_DESIGN.md, PROJECT_SCOPE.md, ROADMAP.md |
| MED-01 | MEDIUM | ✅ Fixed | PROJECT_SCOPE.md, ROADMAP.md |
| MED-02 | MEDIUM | ✅ Fixed | SCIENTIFIC_ASSUMPTIONS.md |
| MED-03 | MEDIUM | ✅ Fixed | BENCHMARK_DESIGN.md |
| MED-04 | MEDIUM | ✅ Fixed | BENCHMARK_DESIGN.md |
| MED-05 | MEDIUM | ✅ Fixed | DATA_PROVENANCE.md, RISKS.md |
| MED-06 | MEDIUM | ✅ Fixed | DATA_PROVENANCE.md |
| MED-07 | MEDIUM | ✅ Fixed | PROJECT_SCOPE.md, ARCHITECTURE.md |
| LOW-01 | LOW | ✅ Fixed | PROJECT_SCOPE.md |
| LOW-02 | LOW | ✅ Fixed | ARCHITECTURE.md, ROADMAP.md |
| LOW-03 | LOW | ✅ Fixed | BENCHMARK_DESIGN.md |
| LOW-04 | LOW | ✅ Fixed | BENCHMARK_DESIGN.md |
