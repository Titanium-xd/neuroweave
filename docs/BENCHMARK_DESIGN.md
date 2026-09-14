# BENCHMARK_DESIGN.md
# Animal Brain Benchmark — Benchmark Design Specification

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

---

## 1. Design Philosophy

### 1.1 Core Principle: Structural Fairness

The benchmark is valid only if every architectural comparison is structurally identical
across all agents on the properties relevant to the comparison. The following are held
constant per-experiment:

| Property | Requirement |
|---|---|
| Task and environment | Same code, same random seed for environment |
| Observation vector dimension (D_obs) | Fixed per task; independent of subgraph size |
| Action space | Same discrete or continuous space for all agents |
| Episode length | Same maximum steps per episode |
| Evaluation protocol | Same evaluation episodes, seeds, and metrics |
| Random seeds | Full set of seeds reported, never only the best |
| Hyperparameter search budget | Same search procedure and number of trials per architecture (see §3.2) |

**Training budget is NOT held constant to a single number.** Three separate budget axes
are used and all three are reported (see §3.1).

### 1.2 Prohibited Practices

- **No single "fair" training step budget** — three budget axes are required (§3.1)
- **No task-specific hyperparameter tuning for one architecture** without applying the
  same search procedure to all others
- **No cherry-picking runs** — all seeds reported; mean ± CI across seeds
- **No changing the task definition between architectures**
- **No hidden post-hoc normalization**
- **No reporting only successful architectures** — all architectures in a comparison set
  are reported together
- **No claiming any result as biologically meaningful** without explicitly listing the
  assumptions under which it was obtained

### 1.3 Reproducibility Requirements

Every experiment generates a manifest file (§5). The manifest contains all information
needed to exactly reproduce the run, including the subgraph hash, hyperparameter search
log, software versions, and hardware spec.

### 1.4 Benchmark Versioning

Benchmark versions follow `MAJOR.MINOR.revision` semantics:
- **MAJOR** bump: task definition changes (breaks comparability with prior results)
- **MINOR** bump: new architectures or controls added (prior results remain valid)
- **revision** bump: documentation or metadata corrections only

Each benchmark version is tagged in the Git repository. Results files embed the benchmark
version string. Plots generated from results files automatically inherit the version.
Task suite changes require a MAJOR bump and a migration guide explaining how prior results
relate to the new version.

**Current version:** `ABB-0.1-rev1`

---

## 2. Task Suite Design

### 2.1 Task Observation Interface

**Critical design invariant:** All tasks define a **fixed observation dimension D_obs**
that is a task constant, independent of the size of the biological subgraph used in any
agent. Every agent receives this same D_obs-dimensional float32 observation vector.

The biological GNN/LIF agents translate this D_obs-dimensional observation into their
internal representation via a learned or fixed input projection matrix W_in
(shape: [n_input_neurons × D_obs]). This projection is part of the agent's architecture,
not part of the task. The MLP and LSTM baselines similarly receive the same D_obs vector
through their own input layers.

This design ensures the task is identical across architectures regardless of subgraph size.

### 2.2 Task Selection Criteria

1. **Fixed observation dimension** — D_obs is a task constant, not a function of any agent property
2. **Temporal structure** — tasks require some form of temporal integration or state maintenance
3. **Computationally tractable** — runnable on a single GPU in reasonable wall-clock time
4. **Well-defined metrics** — unambiguous quantitative measures; no task-specific post-hoc normalization
5. **Observation/action neutrality** — does not structurally favor any specific architecture family
   unless that favoritism is the explicit experimental variable being tested
6. **Cannot be trivially solved** — no architecture can approach ceiling performance without
   using the task's intended computation (no replay-buffer memorization, no lookup tables)

### 2.3 Task Specifications

---

#### T-001: Binary Pattern Discrimination

**Type:** Supervised binary classification
**Fixed observation dimension:** D_obs = 64
**Rationale:** Tests discriminative representation learning without favoring spatial or
temporal structure specifically.

| Parameter | Value |
|---|---|
| Observation | Float32 vector, length 64; each dimension drawn i.i.d. ∈ {−1, +1} with class-conditional correlation structure |
| Output | Binary class label {0, 1} |
| Class structure | 2 classes; balanced by default; class-imbalanced as variant |
| Correlation structure | Class-predictive dimensions: randomly selected subset of 8 of 64 |
| Training signal | Cross-entropy loss (supervised mode); binary reward for correct classification (RL mode) |
| Primary metrics | M-002 (accuracy), M-003 (AUC-ROC) |
| Variants | Noisy inputs σ ∈ {0.1, 0.3, 0.5}; adversarial perturbation; class-imbalance ratio ∈ {1:1, 1:5, 1:10} |

**Cannot be solved trivially:** The 8 predictive dimensions are randomly selected and unknown
to the agent. All agents must learn to identify them from training data. No architecture has
structural access to the ground-truth relevant dimensions.

**Input projection for biological agents:** W_in (n_input_neurons × 64), initialized
orthogonally for A1 variants; same initialization rule for all.

---

#### T-002: Capacity-Limited Sequence Memory

**Type:** Working memory under capacity constraint
**Fixed observation dimension:** D_obs = 16 (one-hot encoding over 16 symbols)
**Rationale:** Tests whether an architecture maintains compressed internal representations
across a delay period. The capacity constraint prevents trivial replay-buffer solutions.

| Parameter | Value |
|---|---|
| Observation | One-hot vector, length 16 (one of 16 symbols per step) |
| Task structure | Observe sequence of T_seq symbols; then observe K delay steps (null symbol); then output the symbol from T_seq steps ago |
| Output | One of 16 symbol labels (classification) |
| Delay period | K ∈ {5, 10, 20, 50} observation steps |
| Sequence length | T_seq ∈ {1, 3, 5} |
| Primary metrics | M-002 (accuracy by delay K), M-006 (memory duration threshold) |
| Trivial baseline included | M-006-BUFFER: a capacity-limited sliding buffer readout (see §2.4) |
| Variants | Distractor symbols during delay; multi-symbol recall; compressed recall |

**Architecture-independent observation steps:** Delay K is defined in observation steps.
Each observation step corresponds to one call to `agent.observe() → agent.step() → agent.act()`.
The mapping between observation steps and the agent's internal time steps is architectural
and must be documented per agent (see SCIENTIFIC_ASSUMPTIONS.md SA-007-GNN, SA-007-LIF).

- For **GNN agents (A1/A3 family):** one observation step = one message-passing iteration.
  Number of message-passing sub-steps per observation step is a hyperparameter T_mp.
- For **LIF agents (A2 family):** one observation step = T_sim ms of simulated biological
  time, where T_sim is a hyperparameter. The observation is encoded as a spike train
  over T_sim time steps at dt = 1 ms resolution.
- For **RNN agents (A8):** one observation step = one RNN unrolling step. Standard.

**Cannot be solved trivially:** The agent's internal state has a bounded representation
capacity enforced by the architecture. The task cannot be solved by storing the full
observation history verbatim — it requires compression. For GNN/LIF agents, compression
capacity is determined by the number of output neurons. For RNN agents, by hidden state
size. For MLP agents, a fixed-size context window is imposed; MLP receives only the
current observation (no history access), forcing it to learn temporal integration.

**Capacity-limited baseline (M-006-BUFFER):** A sliding-window buffer of capacity C symbols
receives the full observation sequence and outputs the item at position −K with perfect
recall up to C delay steps. This baseline provides the trivial ceiling for any K ≤ C.
Any agent that does not outperform M-006-BUFFER at its target delay K has not demonstrated
genuine compressed memory.

---

#### T-003-GRID: Discrete Navigation

**Type:** Goal-directed navigation, discrete grid world, reinforcement learning
**Fixed observation dimension:** D_obs = 32
**Rationale:** Canonical RL benchmark; tests credit assignment over time.

> [!NOTE]
> T-003-GRID uses a discrete compass-based observation that has mild spatial structure.
> This does NOT favor CNN architectures because the observation is a 1D vector of
> compass readings, not a 2D spatial grid. CNNs are not used on T-003-GRID (A9 is
> Phase 3 and only applied to T-003-CONT where spatial structure is explicit).

| Parameter | Value |
|---|---|
| Observation | Float32 vector, length 32: [4 directional distance cues, 4 directional wall cues, agent orientation sin/cos, goal direction sin/cos, goal distance, step fraction remaining, 16 spare reserved] |
| Output | Discrete action {forward, left, right, stay} |
| Reward | +1 reaching goal; −0.01 per step; −1 on collision (optional) |
| Grid | 9×9 to 15×15 cells; wall layout randomized per episode |
| Primary metrics | M-001 (mean episode reward), M-004 (steps to goal), M-005 (success rate) |
| RL seeds | 10 seeds (not 5) — see §3.1 |
| Variants | Maze complexity; reward delay; observation noise; goal repositioning |

**Orientation is explicitly observable** as sin/cos components. Path integration is not
required; the task tests sensorimotor coupling rather than dead-reckoning.

---

#### T-003-CONT: Continuous Sensorimotor Integration (Phase 3)

**Type:** Continuous control, reinforcement learning
**Fixed observation dimension:** D_obs = 8
**Rationale:** Structurally analogous to what descending motor neurons and VNC circuits do —
continuous sensory feedback driving continuous output correction. More biologically motivated
for VNC-topology subgraphs than discrete grid navigation.

| Parameter | Value |
|---|---|
| Observation | Float32 vector, length 8: [current signal value, target value, error, error derivative, error integral (clipped), step fraction, time since last reward, action from t−1] |
| Output | Continuous scalar in [−1, +1]: correction magnitude and direction |
| Task | Regulate a noisy scalar signal toward a moving target; target follows a smooth random walk |
| Reward | −|error| per step; sparse +1 at episode end if mean |error| < threshold |
| Primary metrics | M-001 (episode reward), M-010 (wall-time per step, especially relevant here) |
| Variants | Noise amplitude; target walk speed; actuation delay; partial observability |

---

#### T-004a: Post-Training Node Ablation

**Type:** Structural sensitivity analysis (meta-task, applied after training)
**Applies to:** All agents trained to criterion on T-001 or T-002

**Procedure:**
1. Train agent to performance criterion on base task (T-001 or T-002), save checkpoint.
2. Progressively remove nodes from the agent's internal representation:
   - For GNN/LIF: zero out the activations of the selected neurons (soft lesion)
   - For MLP: zero out the selected hidden units
   - For LSTM: zero out selected components of the hidden state vector
3. Re-evaluate performance without retraining after each removal step.
4. Compute M-008 (lesion robustness AUC) over the removal curve.

**Node removal strategies:**
- Random (provides baseline degradation curve)
- By degree rank (highest out-degree first) — for GNN/LIF only; equivalent (random) for MLP/LSTM
- By activation variance (most active units first) — applicable to all architectures

> [!CAUTION]
> T-004a results MUST NOT be described as measuring "biological robustness to cellular
> damage." They measure post-training sensitivity of a specific trained solution to
> architectural perturbation. The biological model was NOT trained with awareness of
> lesioning; neither was the MLP.

**Architecture comparability:** Only the "random removal" and "activation variance" strategies
are directly comparable across all architectures. Degree-based removal applies only to graph
architectures (A1/A3 family) and is reported separately from the cross-architecture comparison.

---

#### T-004b: Robustness-Trained Lesioning

**Type:** Dropout robustness evaluation (meta-task)
**Applies to:** Agents retrained with structural dropout

**Procedure:**
1. Retrain each agent on T-001 or T-002 WITH node dropout at rate p ∈ {0.05, 0.10, 0.20}.
2. Evaluate the dropout-trained agent on the base task (no dropout at test time).
3. Then apply the T-004a ablation protocol to the dropout-trained agent.
4. Compare M-008 curves between standard-trained (T-004a) and dropout-trained (T-004b) agents.

**Research question answered:** Does training with dropout produce more distributed
representations, and does the biological topology benefit more or less from this than
synthetic controls? Results test whether the biological sparsity pattern is inherently more
redundant or whether redundancy must be learned.

---

#### T-004c: Architecture-Comparable Interface Damage

**Type:** Input/output robustness evaluation (meta-task; comparable across ALL architectures)
**Applies to:** All agents

**Procedure:**
1. Train agent to criterion on T-001 or T-002.
2. Damage the INPUT interface: zero out a fraction f of the D_obs observation dimensions.
3. Damage the OUTPUT interface: add Gaussian noise σ to the agent's action output.
4. Evaluate performance at each damage level without retraining.

**Why this is architecture-comparable:** Every architecture receives a D_obs-dimensional
input and produces a scalar or discrete action output. Input and output damage is equally
defined for all of them. This removes the "cell type lesioning is inapplicable to MLP"
problem from T-004a.

**Research question answered:** Does any architecture's INTERNAL routing allow it to be
more robust to INPUT or OUTPUT damage specifically? This tests the information compression
properties of the architecture's internal structure independently of node-removal effects.

---

### 2.4 Trivial Baselines (Required for Each Task)

The following trivial baselines are mandatory for every task report:

| Baseline ID | Description | Purpose |
|---|---|---|
| A0 | Random policy | Absolute performance lower bound |
| T-001-CHANCE | Majority class predictor | Label imbalance floor for classification |
| T-002-BUFFER | Capacity-limited sliding buffer readout (M-006-BUFFER) | Memory task trivial ceiling |
| T-003-STATIC | Always "stay" policy | Navigation task comparison |

Any architecture that fails to outperform its relevant trivial baseline at statistical
significance is not a useful model and must be highlighted prominently in reports.

---

## 3. Training and Evaluation Protocol

### 3.1 Three-Axis Training Budget

Every ABB comparison reports performance under all three budget conditions:

| Axis | Definition | Fixed Across Architectures |
|---|---|---|
| **Sample Budget (SB)** | Total environment interactions (observations or training examples) | Yes — same count for all |
| **Compute Budget (CB)** | Total training FLOPs (measured pre-run or estimated from operation counts) | Yes — same FLOP target |
| **Wall-Clock Budget (WB)** | Total training wall-clock hours on the reference hardware | Yes — same duration on same machine |

For each axis, performance is plotted as a curve (performance vs. budget consumed) for
each architecture. Final performance is evaluated after the budget is exhausted.

A Pareto frontier of (performance, parameter_count) is reported for each architecture
family under the Sample Budget condition.

**Reference hardware** must be specified per benchmark run (GPU model, CPU model, RAM)
and is recorded in the experiment manifest.

### 3.2 Hyperparameter Search Protocol

**No universal default learning rate or optimizer is imposed.**

Every architecture receives an equal hyperparameter search budget:
- **Method:** Random search (Phase 2); Bayesian optimization (Phase 3+)
- **Trials:** 20 trials per architecture per task
- **Search space:** Defined in `configs/search_spaces/<architecture_id>.yaml`
- **Target:** Performance on the **development split** of the task
  (a held-out task variant not used in final evaluation)
- **Report:** The best hyperparameters found are stored in the manifest and used for
  all evaluation runs

The development split is defined per task (e.g., a smaller grid size for T-003-GRID).
It must not share episodes/patterns with the evaluation split.

**Hyperparameter search counts against the Sample Budget for that architecture.**
This ensures the total budget (including search) is equalized.

### 3.3 Seed Protocol

| Task type | Minimum seeds | CI method |
|---|---|---|
| Supervised (T-001, T-002) | 5 seeds | Bootstrap 95% CI, B = 2000 samples |
| RL (T-003-GRID, T-003-CONT) | 10 seeds | Bootstrap 95% CI, B = 2000 samples |
| Meta-tasks (T-004a/b/c) | Inherits from base task | Same CI method |

Seed values: seeds 0–4 for supervised tasks; seeds 0–9 for RL tasks. All seeds are
reported. Bootstrap CI does not assume normality and is therefore appropriate for
RL reward distributions.

### 3.4 Statistical Reporting Requirements

All reported results MUST include:
- Mean across seeds
- 95% bootstrap CI across seeds (B = 2000)
- Individual seed values in result files (not necessarily in tables)
- Training curves (smoothed with window = 1/20 of total steps, plus raw)
- Performance at each of the three budget axes

The following are NOT acceptable in any ABB publication:
- Reporting only the best seed
- Reporting mean without CI
- Using t-interval CI for RL results (use bootstrap)
- Changing evaluation protocol between architectures

---

## 4. Subgraph Selection Protocol

### 4.1 Connectivity-Respecting Subgraph Methods

**Critical requirement:** Selected subgraphs must be connected (or near-connected).
Type-stratified random sampling is prohibited as a primary selection method because it
produces disconnected induced subgraphs where most neurons have near-zero within-subgraph
degree.

**Required subgraph connectivity report:** Before any subgraph is used in a benchmark run,
the following statistics must be logged in the manifest:

| Statistic | Requirement |
|---|---|
| Largest connected component fraction | ≥ 0.90 |
| Mean in-degree within subgraph | ≥ 3.0 |
| Mean out-degree within subgraph | ≥ 3.0 |
| Fraction of isolated nodes | ≤ 0.05 |

If any requirement is violated, the subgraph is rejected and a new one is sampled.

### 4.2 Approved Subgraph Selection Methods

| Method | Description | Use Case |
|---|---|---|
| **K-hop neighborhood** | Select seed neurons (e.g., all neurons of one sensory type); include all neurons within K directed hops | Tests biological topology in a functionally motivated context |
| **Random walk sample** | Perform weighted random walks from seed neurons; include visited nodes | Produces connected subgraph preserving local density |
| **Spectral sample** | Select nodes that minimize graph Laplacian spectral distortion | Preserves global graph topology properties |
| **Full input-output pathway** | All neurons on shortest directed paths from input to output neurons, within diameter D | Ensures complete sensorimotor pathways |

**Type-stratified random sampling** is permitted ONLY as a secondary comparison in §4.4
(cross-subgraph variance analysis), not as the primary subgraph for any benchmark result.

### 4.3 Subgraph Size Targets

| Size | Use Case | Notes |
|---|---|---|
| ~500 neurons | Unit tests, rapid prototyping | Not scientifically significant |
| ~5,000 neurons | Phase 2 primary benchmark | ~3% of CNS; tractable on single GPU with GNN |
| ~20,000 neurons | Phase 3 extended benchmark | Requires efficient sparse implementation |
| ~100,000 neurons | Phase 4 / long-term | Research-scale |

### 4.4 Cross-Subgraph Variance Reporting

Every Phase 2+ result reports performance over at least 3 independent subgraph selections
using the same selection method. The variance in performance across subgraphs is reported
alongside variance across seeds. This separates "this result depends on the specific
subgraph" from "this result is robust to subgraph choice."

### 4.5 Control Graph Construction

For each biological subgraph used in an experiment, control graphs (A3-ER, A3-CONFIG,
A3-SBM) are constructed as follows:

- **A3-ER:** Erdős–Rényi directed graph with same N and same total edge count E
- **A3-CONFIG:** Configuration model with same N and exact per-node (in_degree, out_degree)
  sequence; generated using the Bayati-Kim-Saberi algorithm
- **A3-SBM:** Stochastic block model with block membership derived from biological cell type
  annotations; within-block and between-block edge probabilities fit to biological subgraph
- **A3-DENSE:** Fully-connected directed graph (N nodes, N² edges), with self-loops removed

All control graphs are constructed with the same random seed as the biological subgraph
selection, stored in the manifest, and reproducible from seed + parameters.

---

## 5. Experiment Manifest Schema (Revised)

Every ABB experiment produces a machine-readable YAML manifest:

```yaml
manifest_version: "1.1"
benchmark_version: "ABB-0.1-rev1"
experiment_id: "exp_20260914_001"
timestamp: "2026-09-14T18:54:00Z"

hardware:
  gpu: "NVIDIA A100 80GB"
  cpu: "AMD EPYC 7763"
  ram_gb: 256
  cuda_version: "12.2"

software:
  abb_version: "0.1.0"
  python_version: "3.11.5"
  torch_version: "2.1.0"
  pyg_version: "2.4.0"

data:
  dataset: "male-cns:v1.0"
  subgraph_method: "k_hop_neighborhood"
  subgraph_seed_neurons: ["LC4", "LPLC2"]
  subgraph_n_hops: 3
  subgraph_n_neurons: 4823
  subgraph_n_edges: 51204
  subgraph_hash: "<sha256 of edge list>"
  subgraph_connectivity:
    largest_cc_fraction: 0.97
    mean_in_degree: 10.6
    mean_out_degree: 10.6
    isolated_fraction: 0.01
  download_date: "2026-09-14"
  neuprint_server: "https://neuprint.janelia.org"
  cache_checksum: "<sha256>"

architecture:
  id: "A1-EDGE"
  name: "MaleCNS-GNN-Edge"
  topology: "biological_malecns"

  biological_model_config:
    gnn_operator: "custom_signed_sparse_linear"
    n_message_passing_steps: 5
    weight_init: "synapse_count_log_normalized"
    weights_trainable: "edges_only"   # one of: none, bias_only, edges_only, edges_and_bias
    sign_rule: "nt_classification"    # one of: nt_classification, unsigned, learnable
    sign_assumption_id: "SA-002"
    input_projection: "learned_orthogonal"   # W_in initialization
    output_readout: "learned_linear"

  trainable_params: 51204             # set at runtime from subgraph edge count
  flops_per_step_estimate: 102408     # 2 × edges (multiply-accumulate)

task:
  id: "T-001"
  name: "BinaryPatternDiscrimination"
  d_obs: 64
  config_hash: "<sha256 of task config>"
  development_split_hash: "<sha256>"

training:
  budget_axis_used: "sample"          # one of: sample, compute, wall_clock
  sample_budget: 500000
  compute_budget_flops: null          # set when budget_axis = compute
  wall_clock_budget_hours: null       # set when budget_axis = wall_clock
  seed: 0
  optimizer: "adam"
  lr: 0.0003                          # from hyperparameter search
  hparam_search_trials: 20
  hparam_search_log: "hparam_search_exp_001.yaml"
  batch_size: 128
  grad_clip_norm: 1.0
  checkpoint_every_steps: 10000

evaluation:
  eval_frequency_steps: 10000
  eval_episodes: 50
  final_eval_episodes: 200
  eval_seed_offset: 100               # evaluation seeds are 100+ to avoid training seed overlap
  n_seeds: 5                          # 10 for RL tasks
  ci_method: "bootstrap"
  ci_bootstrap_samples: 2000

subgraph_runs:                        # cross-subgraph variance
  n_subgraphs: 3
  subgraph_hashes: ["<sha256-1>", "<sha256-2>", "<sha256-3>"]

assumptions_applied:
  - SA-001   # synapse_count_as_weight
  - SA-002   # nt_sign_rule
  - SA-006   # subgraph_selection
  - SA-007-GNN   # gnn_input_encoding
  - SA-008   # gnn_approximation
  - SA-009   # learnable_weights_fixed_topology

results_path: "results/exp_20260914_001/"
results_checksum: "<sha256>"
```

---

## 6. Metrics Registry (Revised)

| Metric ID | Name | Formula / Definition | Task Applicability |
|---|---|---|---|
| M-001 | Mean Episode Reward | Mean total reward over evaluation episodes | RL tasks: T-003-GRID, T-003-CONT only |
| M-002 | Classification Accuracy | Correct / Total predictions | T-001, T-002 |
| M-003 | AUC-ROC | Area under ROC curve | T-001 |
| M-004 | Steps to Goal | Mean steps per successful episode | T-003-GRID |
| M-005 | Success Rate | Episodes reaching goal / Total episodes | T-003-GRID, T-003-CONT |
| M-006 | Memory Duration | Delay K at which accuracy first falls below 75% | T-002 |
| M-006-BUFFER | Buffer Ceiling | Delay at which the capacity-limited buffer baseline falls below 75% | T-002 (trivial baseline) |
| M-007 | Area Under Learning Curve (AULC) | ∫ performance(t) dt / T_total, normalized to [0, 1] | All tasks |
| M-008 | Lesion Robustness AUC | Area under performance-vs-fraction-removed curve | T-004a, T-004b |
| M-009 | Inference FLOPs | Floating point ops per forward pass (one observation step) | All architectures |
| M-010 | Inference Wall Time | ms per forward pass on reference hardware | All architectures |
| M-011 | Peak Memory | Maximum GPU/CPU memory during inference (single batch) | All architectures |
| M-012 | Training FLOPs | Total FLOPs consumed during training (for compute budget axis) | All architectures |
| M-013 | Pareto Efficiency | Performance at each parameter count; frontier plotted per architecture family | All architectures |

**M-007 definition note:** AULC is computed as the discrete integral of the evaluation
performance curve (M-002 or M-001 at each eval checkpoint) divided by total training budget.
It is comparable across architectures because it uses an absolute performance axis, not
"80% of final performance." All AULC values are normalized to [0, 1] using task-specific
chance-level and ceiling-level values.

**M-001 applies to T-003-GRID and T-003-CONT only.** It does not apply to T-002 (which
is a supervised classification task, not RL).

---

## 7. Reproducibility Checklist

A result is considered fully reproducible when:

- [ ] Full experiment manifest v1.1 (§5) is present
- [ ] Exact software version string recorded (`abb.__version__`, Python, PyTorch, PyG, CUDA)
- [ ] All random seeds recorded (training + evaluation)
- [ ] Data provenance recorded (dataset version, download date, cache checksum)
- [ ] Subgraph hash(es) recorded and subgraph connectivity statistics logged
- [ ] Hyperparameter search log stored
- [ ] Training curves for all seeds and budget axes stored
- [ ] Final evaluation results (all seeds) stored
- [ ] Hardware specification recorded
- [ ] Wall-clock training time recorded
- [ ] Assumptions applied (SA-XXX IDs) listed in manifest
- [ ] All trivial baselines (§2.4) evaluated and recorded
- [ ] Cross-subgraph variance (≥ 3 subgraphs) recorded

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | CRIT-03/04: replaced single-axis fairness with three-axis protocol and Pareto reporting; CRIT-05: T-001 redesigned with fixed D_obs=64; CRIT-06: T-002 redesigned with capacity constraint, architecture-independent observation steps, M-006-BUFFER baseline; CRIT-07: T-004 split into T-004a/b/c; CRIT-08: trivial baselines table added; HIGH-03: fixed M-001 applicability (T-002 removed); HIGH-04: replaced type-stratified subgraph with connectivity-respecting methods; HIGH-05: replaced universal lr=1e-3 with hyperparameter search protocol; HIGH-06: added T-003-CONT; MED-03: increased RL seeds to 10, bootstrap CI; MED-04: expanded manifest schema v1.1; LOW-03: added benchmark versioning §1.4; LOW-04: replaced circular M-007 with AULC |
