# PROJECT_SCOPE.md
# Animal Brain Benchmark — Project Scope and Research Questions

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

---

## 1. Project Identity

**Full Name:** Animal Brain Benchmark (ABB)
**First Biological Architecture Source:** Adult male *Drosophila melanogaster* central nervous
system connectome (MaleCNS v1.0)
**Nature of the Project:** An open-source, reproducible benchmark framework

---

## 2. What ABB Is

ABB is a rigorous, reproducible benchmark framework for comparing:

1. **Connectome-topology-constrained networks** — neural network architectures whose
   connectivity graph (which nodes connect to which) is derived directly from published
   biological connectome data, but whose per-synapse weights may be initialized from
   synapse counts, randomized, or optimized by gradient descent.
2. **Conventional artificial neural architectures** — MLPs, LSTMs/GRUs, CNNs, transformers,
   simple SNNs, and graph neural networks with randomly-initialized or randomly-generated
   connectivity.
3. **Structural control graphs** — synthetic graphs engineered to preserve specific statistical
   properties of the biological connectome (edge count, degree sequence, community structure)
   while randomizing others.

All architectures compete on **identical tasks** with **identical observation vectors**,
**identical action spaces**, **identical training budgets** (defined on three axes — see §5),
and are evaluated by **identical metrics**.

---

## 3. Precise Primary Research Questions

> [!IMPORTANT]
> The primary question has been deliberately narrowed from ABB-0.1 to prevent overclaiming.
> Each sub-question is answered by a distinct set of experimental conditions listed in §6.

### PRQ-1 — Topology as Learned Inductive Bias

> **"When connectome-derived sparsity and connectivity structure is used as a fixed
> architectural prior for a trainable recurrent graph network (all weights optimized by
> gradient descent), does that prior lead to better task performance, data efficiency,
> or robustness than (a) architecturally equivalent dense networks, (b) random sparse
> networks with the same number of edges, (c) random graphs preserving the exact same
> per-node degree sequence, and (d) conventional recurrent architectures (MLP, LSTM)?"**

Answered by comparing: A1 vs. A3-ER, A3-CONFIG, A3-SBM, A3-DENSE, A7, A8.

*Vocabulary note:* **Inductive bias** refers here to the structural preference for certain
learnable function classes that is introduced by fixing the sparsity pattern before training.
It is meaningful only when weights are optimized by gradient descent.

### PRQ-2 — Topology + Synapse Counts as a Structural Prior (Zero-Shot / Few-Shot)

> **"Does the MaleCNS topology combined with synapse-count-derived weights, used
> entirely without gradient-based optimization, produce above-chance performance on
> structured tasks? Does it outperform random-weight instances of the same topology?"**

Answered by comparing: A1-BIO vs. A1-FROZEN vs. A0.

*Vocabulary note:* **Structural prior** refers to the performance of the fixed-weight model
derived directly from connectome data, without any gradient-based optimization. It is
categorically different from PRQ-1.

### PRQ-3 — Topology Structure vs. Degree Sequence

> **"How much of any observed advantage of the biological topology is due to the specific
> wiring pattern (which neuron connects to which), versus the degree sequence alone
> (how many connections each neuron has)?"**

Answered by comparing: A1 vs. A3-CONFIG (same degree sequence, random wiring).

### PRQ-4 — Robustness and Lesioning

> **"Is the biological topology's trained solution more or less sensitive to post-training
> node removal than comparable synthetic architectures, and does training with structural
> dropout improve robustness differentially?"**

Answered by T-004a and T-004b. Results are NOT claimed to measure biological robustness
to cellular damage — they measure trained-solution sensitivity to architectural perturbation.

### PRQ-5 — Computational Cost

> **"What is the Pareto frontier of task performance vs. computational cost (FLOPs,
> wall-clock time, memory) for each architecture family?"**

Answered by M-009/M-010/M-011 reporting across all architectures.

---

## 4. What ABB Is NOT

The following claims are explicitly prohibited in any ABB publication, documentation,
or communication. Any text in any document that implies one of these claims is a bug:

| Prohibited Claim | Why It Is Incorrect |
|---|---|
| "We have simulated / digitally recreated a living fruit fly" | The connectome records static synaptic structure, not physiology, biochemistry, or dynamics. |
| "A1-BIO reproduces fly behavior" | A1-BIO is the most data-direct connectome-derived condition, not a biological simulation. |
| "The model is conscious or sentient" | Consciousness is not a property of connectivity graphs. |
| "A higher benchmark score means the fly brain is computationally superior" | Task performance depends on training procedure, task choice, subgraph selection, and many assumptions. |
| "Results generalize to vertebrate or human brains" | The invertebrate CNS architecture differs radically from vertebrate brains. |
| "Lesion experiments measure biological robustness to cellular damage" | They measure post-training sensitivity to architectural perturbation in a specific model variant. |
| "The biological model faithfully captures Drosophila biology" | Many critical biophysical parameters are unavailable or approximated (see SCIENTIFIC_ASSUMPTIONS.md). |

---

## 5. Fairness Protocol — Three-Axis Comparison

> [!IMPORTANT]
> A single "fair comparison" does not exist. ABB uses three separate comparison axes and
> reports all three. No single axis is declared "the" fair comparison.

### Axis 1 — Fixed Sample Budget (Data Efficiency)
All architectures are trained on the same number of environment interactions (observations
or training examples). Tests how efficiently each architecture uses data. FLOPs and
wall-clock time may differ substantially.

### Axis 2 — Fixed Compute Budget (Compute Efficiency)
All architectures are allocated the same total floating-point operations for training.
Tests performance per unit of compute. Training duration and data volume may differ.

### Axis 3 — Fixed Wall-Clock Budget (Practical Efficiency)
All architectures train for the same wall-clock duration on the same reference hardware.
Tests practical deployability. FLOPs and sample counts may differ.

**Reporting requirement:** For every comparison table, all three axes are reported. The reader
chooses the axis relevant to their use case. No single axis result is promoted as "the" answer.

**Hyperparameter search:** Every architecture receives the same hyperparameter search budget
(search procedure, number of trials, and search space) on a held-out development task variant
before evaluation. The search is identical in procedure across architectures, not in the
specific hyperparameters searched. No universal default learning rate is imposed.

**Parameter count reporting:** Performance is reported as a Pareto frontier curve across
parameter counts (not a single matched-budget point). Each architecture family's frontier
is plotted separately. This makes capacity tradeoffs visible rather than collapsing them to
a single comparison point.

---

## 6. Architecture Comparison Matrix (Revised)

All architectures implement the `AbstractAgent` interface (see ARCHITECTURE.md §3).

### 6.1 Primary Biological Architecture Family

| ID | Name | Description | Phase | Answers |
|---|---|---|---|---|
| **A1-BIO** | MaleCNS-BIO | Fixed MaleCNS topology; weights initialized from synapse counts and FROZEN (no gradient training) | Phase 2 | PRQ-2 |
| **A1-FROZEN** | MaleCNS-FROZEN | Fixed MaleCNS topology; weights RANDOMLY initialized and FROZEN (no gradient training) | Phase 2 | PRQ-2 |
| **A1-BIAS** | MaleCNS-GNN-Bias | Fixed MaleCNS topology; only per-neuron bias vectors are trainable (~N params) | Phase 2 | PRQ-1, PRQ-3 |
| **A1-EDGE** | MaleCNS-GNN-Edge | Fixed MaleCNS topology; all non-zero edge weights are trainable (~E params) | Phase 2 | PRQ-1, PRQ-3 |
| **A1-BOTH** | MaleCNS-GNN-Full | Fixed MaleCNS topology; bias + edge weights both trainable (~N+E params) | Phase 2 | PRQ-1 |

> [!NOTE]
> A1-BIO is described as the "most data-direct, least-trained connectome-derived condition."
> It is NOT described as a biological simulation. The only biological origin is the topology
> and the synapse-count weight initialization.

### 6.2 Structural Control Graph Family (Topology Hypothesis Tests)

| ID | Name | Preserved from MaleCNS | Randomized | Tests |
|---|---|---|---|---|
| **A3-ER** | Erdős–Rényi GNN | Node count, edge count | Everything else | Edge count alone |
| **A3-CONFIG** | Configuration-Model GNN | Node count, exact per-node in/out degree sequence | Wiring | Degree sequence vs. specific wiring |
| **A3-SBM** | Stochastic Block Model GNN | Node count, community block structure, within-block edge density | Within-block wiring | Community structure |
| **A3-DENSE** | Fully-Connected GNN | Node count, GNN operator type | Sparsity | Sparsity prior |
| **A5-WSHUFFLE** | Shuffled-Weight GNN | MaleCNS topology, weight magnitude distribution | Weight assignments to edges | Weight distribution vs. topology |
| **A5-WSIGN** | Shuffled-Sign GNN | MaleCNS topology, synapse counts | NT-derived signs (randomly reassigned) | NT sign structure |
| **A5-ETARGET** | Shuffled-Target GNN | MaleCNS topology, edge weight magnitudes per source neuron | Which target neuron each weight connects to | Weight distribution vs. specific wiring |

### 6.3 Conventional Architecture Baselines

| ID | Name | Description | Phase |
|---|---|---|---|
| **A0** | Random Policy | Uniformly random action selection; absolute performance lower bound | Phase 2 |
| **A6** | Simple LIF SNN | Randomly sparse LIF spiking network, matched edge density | Phase 3 |
| **A7** | MLP | Multilayer perceptron; Pareto sweep over parameter counts | Phase 2 |
| **A8** | LSTM/GRU | Recurrent architecture; Pareto sweep over hidden sizes | Phase 2 |
| **A9** | CNN | Convolutional policy network (spatially structured tasks only) | Phase 3 |
| **A10** | Transformer | Attention-based architecture | Phase 4 |

### 6.4 LIF Spiking Architecture Family (Phase 3)

| ID | Name | Description | Phase |
|---|---|---|---|
| **A2-BIO** | MaleCNS-LIF-BIO | MaleCNS topology + synapse count weights, LIF neurons, FROZEN | Phase 3 |
| **A2-TRAIN** | MaleCNS-LIF-Trained | MaleCNS topology, LIF neurons, trainable weights (surrogate gradient) | Phase 3 |

---

## 7. Task Suite (Overview)

Full task specifications are in BENCHMARK_DESIGN.md.

| ID | Name | Type | Fixed Obs Dim | Phase |
|---|---|---|---|---|
| T-001 | Binary Pattern Discrimination | Supervised classification | D_obs = 64 | Phase 2 |
| T-002 | Capacity-Limited Sequence Memory | Working memory (capacity-constrained) | D_obs = 16 | Phase 2 |
| T-003-GRID | Discrete Navigation | RL — goal-directed navigation | D_obs = 32 | Phase 2 |
| T-003-CONT | Sensorimotor Integration | RL — continuous control | D_obs = 8 | Phase 3 |
| T-004a | Post-Training Node Ablation | Structural sensitivity meta-task | Inherits from T-001/T-002 | Phase 3 |
| T-004b | Robustness-Trained Lesioning | Trained robustness meta-task | Inherits from T-001/T-002 | Phase 3 |
| T-004c | Interface Damage Robustness | Architecture-comparable input/output damage | Inherits from T-001/T-002 | Phase 3 |

---

## 8. Common Agent Interface

All agents implement the following interface (Python details in ARCHITECTURE.md §3):

```
Agent.reset()           → Resets internal state for new episode
Agent.observe(obs)      → Accepts D_obs-dimensional float32 observation vector
Agent.step()            → Advances internal dynamics one observation step
Agent.act()             → Returns action (or action distribution)
Agent.save(path)        → Serializes agent state and weights
Agent.load(path)        → Restores agent state and weights
Agent.get_config()      → Returns machine-readable config dict
Agent.n_trainable_params → Count of trainable scalar parameters
Agent.flops_per_step    → Estimated FLOPs for one forward pass
```

**Key design invariant:** `observe()` always receives a fixed D_obs-dimensional vector,
regardless of subgraph size. The agent's internal projection from D_obs into its internal
representation is an architectural detail, not a task detail.

---

## 9. Knowledge Classification System

All significant claims in ABB code, configs, and documents MUST be tagged:

| Tag | Category | Description |
|---|---|---|
| `[BIO-FACT]` | Experimental biological fact | Published and replicated experimental result |
| `[DATASET-FACT]` | MaleCNS dataset datum | Directly read from MaleCNS v1.0; subject to reconstruction error |
| `[PUB-MODEL]` | Published computational model | From peer-reviewed computational paper |
| `[ASSUMPTION]` | Engineering assumption | ABB design choice; no direct biological basis |
| `[RESULT]` | ABB empirical result | Produced by our software; not a biological claim |

---

## 10. Scope Boundaries by Phase

### Phase 1 (Complete) — Research and Documentation
Establish the scientific and design foundation. No simulation code.

### Phase 2 — Core Framework and First GNN Comparison
Implement data access, A1-family, A3-family, A0, A7, A8. Tasks T-001 and T-003-GRID.
Produce first multi-axis comparison report.

### Phase 3 — LIF Simulation and Extended Comparison
Implement A2-family, A6. Tasks T-002, T-003-CONT, T-004a/b/c.
Full assumption ablation study. External scientific review.

### Phase 4 — Advanced Architectures and Publication Readiness
A9, A10. Larger subgraphs. Cross-connectome comparisons. Publication preparation.

---

## 11. Out-of-Scope Items (Explicitly Excluded)

- Gene expression, protein dynamics, metabolic processes
- Gap junctions (electrical synapses) — not in MaleCNS v1.0 annotations
- Developmental stages (larval, pupal)
- Multi-specimen population variability
- Claims about vertebrate or human cognition or medicine
- Real-time neuromorphic deployment (beyond Phase 4 stretch goal)
- Any claim that ABB measures fly intelligence, consciousness, or behavior

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | CRIT-01: rewrote core research questions; CRIT-02/08: expanded architecture matrix with structural controls and required baselines; CRIT-03/04: replaced single-axis fairness with three-axis protocol; MED-07: defined A1-BIAS/EDGE/BOTH; LOW-01: standardized inductive bias vs structural prior; MED-01: corrected A6 phase assignment |
