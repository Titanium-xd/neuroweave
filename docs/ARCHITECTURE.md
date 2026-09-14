# ARCHITECTURE.md
# Animal Brain Benchmark — Software Architecture

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

---

## 1. Architectural Principles

1. **Reproducibility first** — deterministic behavior given the same seed; full provenance tracking
2. **Configuration-driven** — all experimental parameters in YAML config files; no magic numbers
3. **Modular and replaceable** — each component has a clean interface; swappable independently
4. **Sparse representations** — connectome is extremely sparse; no dense tensor operations
   for graph-level computation
5. **Fixed D_obs interface** — all agents receive the same D_obs-dimensional observation;
   internal projections are architecture-specific, not task-specific
6. **Clean separation of concerns** — data, simulation, environment, agent, training, evaluation,
   and visualization are distinct modules with no circular dependencies
7. **Automated testing** — each module has unit tests; integration tests validate full runs
8. **Benchmark versioning** — experiment configs and results are versioned; old results
   re-plottable from saved files without re-running
9. **Machine-readable results** — all results stored as JSON/Parquet/CSV; plots generated from
   saved results, never from in-memory computation alone
10. **Containerized reproducibility** — Docker image created in Phase 2 alongside first
    implementation; not deferred to publication

---

## 2. Repository Layout

```
animal-brain-benchmark/
│
├── docs/                          # Scientific documentation
│   ├── PROJECT_SCOPE.md
│   ├── SCIENTIFIC_ASSUMPTIONS.md
│   ├── DATA_PROVENANCE.md
│   ├── BENCHMARK_DESIGN.md
│   ├── ARCHITECTURE.md            ← this file
│   ├── RISKS.md
│   ├── ROADMAP.md
│   ├── CHANGELOG_PHASE1_REV1.md
│   └── DESIGN_FREEZE_CHECKLIST.md
│
├── abb/                           # Main Python package
│   ├── __init__.py                # Package version, public API
│   ├── config/
│   │   ├── schema.py              # Pydantic v2 schemas for all config types
│   │   └── loader.py              # YAML loading with validation
│   │
│   ├── data/
│   │   ├── neuprint_client.py     # neuPrint API wrapper (authenticated)
│   │   ├── cache.py               # Local Parquet cache + .provenance.json sidecar
│   │   ├── graph.py               # Graph construction; connectivity validation
│   │   ├── subgraph.py            # Subgraph selection (k-hop, random walk, spectral, pathway)
│   │   └── annotations.py         # Neuron metadata: type, NT sign, ROI
│   │
│   ├── models/                    # Neural network architectures
│   │   ├── base.py                # AbstractAgent interface
│   │   │
│   │   ├── biological/            # Connectome-topology models (Phase 2)
│   │   │   ├── malecns_gnn_bio.py     # [A1-BIO] Frozen synapse-count weights
│   │   │   ├── malecns_gnn_frozen.py  # [A1-FROZEN] Frozen random weights
│   │   │   ├── malecns_gnn_bias.py    # [A1-BIAS] Trainable bias only
│   │   │   ├── malecns_gnn_edge.py    # [A1-EDGE] Trainable edge weights
│   │   │   └── malecns_gnn_full.py    # [A1-BOTH] Trainable bias + edges
│   │   │
│   │   ├── controls/              # Structural control graphs (Phase 2)
│   │   │   ├── erdos_renyi.py         # [A3-ER] Erdős–Rényi control
│   │   │   ├── configuration.py       # [A3-CONFIG] Configuration model
│   │   │   ├── sbm.py                 # [A3-SBM] Stochastic block model
│   │   │   ├── dense_gnn.py           # [A3-DENSE] Fully-connected GNN
│   │   │   ├── shuffled_weight.py     # [A5-WSHUFFLE] Weight magnitude shuffled
│   │   │   ├── shuffled_sign.py       # [A5-WSIGN] NT sign shuffled
│   │   │   └── shuffled_target.py     # [A5-ETARGET] Edge target shuffled
│   │   │
│   │   ├── baselines/             # Conventional architectures (Phase 2)
│   │   │   ├── random_policy.py       # [A0] Random policy (trivial lower bound)
│   │   │   ├── mlp.py                 # [A7] MLP; Pareto sweep over param counts
│   │   │   └── lstm.py                # [A8] LSTM/GRU; Pareto sweep over hidden sizes
│   │   │
│   │   └── lif/                   # LIF spiking models (Phase 3)
│   │       ├── malecns_lif_bio.py     # [A2-BIO] Frozen synapse-count weights, LIF
│   │       └── malecns_lif_train.py   # [A2-TRAIN] Trainable weights, LIF
│   │
│   ├── operators/                 # Custom neural operators
│   │   └── signed_sparse_linear.py   # Custom sparse message-passing operator
│   │                                 # (no GCNConv normalization; no GAT attention)
│   │
│   ├── environments/
│   │   ├── base.py                # BaseEnvironment ABC (gymnasium-compatible)
│   │   ├── t001_pattern.py        # [T-001] Binary pattern discrimination; D_obs=64
│   │   ├── t002_memory.py         # [T-002] Capacity-limited sequence memory; D_obs=16
│   │   ├── t003_grid.py           # [T-003-GRID] Discrete navigation; D_obs=32
│   │   └── t003_cont.py           # [T-003-CONT] Continuous sensorimotor; D_obs=8
│   │
│   ├── training/
│   │   ├── supervised.py          # Supervised training loop (T-001, T-002)
│   │   ├── reinforce.py           # REINFORCE policy gradient
│   │   ├── ppo.py                 # PPO (Phase 3+)
│   │   ├── hparam_search.py       # Hyperparameter search (random; Bayesian in Phase 3)
│   │   └── checkpointing.py       # Checkpoint save/load with provenance sidecar
│   │
│   ├── evaluation/
│   │   ├── evaluator.py           # Multi-seed, multi-budget-axis evaluator
│   │   ├── metrics.py             # M-001 through M-013 implementations
│   │   ├── lesion.py              # T-004a/b/c runner
│   │   └── pareto.py              # Pareto frontier computation and plotting data
│   │
│   ├── visualization/
│   │   ├── training_curves.py
│   │   ├── comparison_bars.py
│   │   ├── pareto_frontiers.py    # Performance vs. parameter count plots
│   │   ├── lesion_curves.py
│   │   └── graph_stats.py
│   │
│   └── utils/
│       ├── reproducibility.py     # Seed setting; torch.use_deterministic_algorithms
│       ├── provenance.py          # Manifest v1.1 writing and reading
│       ├── logging.py             # Structured logging (no token values ever logged)
│       ├── sparse.py              # Sparse tensor utilities
│       └── graph_construction.py  # Control graph generators (ER, CONFIG, SBM)
│
├── configs/
│   ├── experiments/               # One YAML per experiment
│   └── search_spaces/             # Hyperparameter search spaces per architecture
│       ├── A1-EDGE.yaml
│       ├── A7.yaml
│       └── A8.yaml
│
├── data/                          # NOT in Git
│   └── raw/malecns_v1_0/
│
├── results/                       # NOT in Git; archivable separately
│
├── scripts/
│   └── data/download_malecns.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── notebooks/                     # Exploration only; NOT for results generation
│
├── docker/
│   └── Dockerfile                 # Created in Phase 2 alongside first implementation
│
├── .env.example                   # NEUPRINT_TOKEN= (template; value blank)
├── .gitignore                     # data/, results/, .env
├── pyproject.toml
├── README.md
└── CITATIONS.bib                  # Phase 2
```

---

## 3. Core Interface: AbstractAgent

All agents MUST implement this interface exactly. The interface is stable across
benchmark versions within MAJOR.x.x; any change requires a MAJOR version bump.

```python
# abb/models/base.py  — Canonical definition (Phase 2 implementation)

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class AbstractAgent(ABC):
    """
    Common interface for all ABB agents.

    Key invariant: observe() always receives a float32 array of shape (D_obs,)
    where D_obs is the task's fixed observation dimension. The agent is responsible
    for projecting D_obs into its internal representation.

    [ARCH-001] Interface stability: changes to this signature require a MAJOR
    benchmark version bump.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for the start of a new episode."""
        ...

    @abstractmethod
    def observe(self, observation: np.ndarray) -> None:
        """
        Accept the current environment observation.
        observation: float32 array, shape (D_obs,).
        D_obs is fixed per task; must not depend on subgraph size.
        """
        ...

    @abstractmethod
    def step(self) -> None:
        """
        Advance the agent's internal dynamics by one observation step.
        For GNN agents: one observation step = T_mp message-passing sub-steps.
        For LIF agents: one observation step = T_sim ms of simulated time.
        For RNN agents: one RNN unrolling step.
        These are NOT equivalent quantities — see SA-007-GNN and SA-007-LIF.
        """
        ...

    @abstractmethod
    def act(self) -> np.ndarray:
        """Return action for current observation step."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize weights and write sidecar provenance JSON."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore weights; verify provenance JSON matches current config."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return machine-readable config dict for manifest recording.
        Must include: architecture ID, all hyperparameters, applied assumption IDs,
        n_trainable_params, flops_per_step_estimate.
        """
        ...

    @property
    @abstractmethod
    def n_trainable_params(self) -> int:
        """Count of trainable scalar parameters."""
        ...

    @property
    @abstractmethod
    def flops_per_step(self) -> int:
        """Estimated FLOPs for one forward pass (one observation step)."""
        ...
```

---

## 4. Custom Signed Sparse Linear Operator

The primary message-passing operator for all A1 and A3 GNN variants is a custom
sparse linear layer. This is the reference implementation description; Phase 2
implements it in `abb/operators/signed_sparse_linear.py`.

### 4.1 Mathematical Specification

For neuron i at observation step t+1:

```
h_i(t+1) = activation_fn( b_i + Σ_{j: (j,i) ∈ E} w_ij · h_j(t) )
```

Where:
- `E` is the fixed edge set from the subgraph (sparsity mask; never changes after construction)
- `w_ij` is the edge weight scalar for edge (j → i)
- `b_i` is per-neuron bias scalar
- `activation_fn` is configurable (default: ReLU for all hidden neurons)

**NO degree normalization is applied.** This means high-degree neurons accumulate
larger pre-activation values — consistent with the biology (hub neurons receive
many inputs and are thus highly influenced by their presynaptic population).

**NO attention mechanism is applied** in the primary operator. Attention would add
per-edge learnable parameters that override the weight prior, breaking the connection
between SA-001 and the model.

### 4.2 Ablation Variants (Labeled; NOT Primary Condition)

```python
# Used only as labeled ablations; never in primary A1 benchmark condition
# [SA-009-GCN] ABLATION: gcnconv_normalization
# [SA-009-GAT] ABLATION: gatconv_attention
```

| Variant | Operator | When Used |
|---|---|---|
| Primary (A1) | Custom signed sparse linear (§4.1) | All primary benchmark runs |
| A1-GCN | GCNConv (`D^{-1/2} A D^{-1/2}`) | Ablation study only; labeled [SA-009-GCN] |
| A1-GAT | GATConv (per-edge attention) | Ablation study only; labeled [SA-009-GAT] |

### 4.3 Weight Initialization by Agent Variant

| Agent ID | Edge Weight Init | Trainable? |
|---|---|---|
| A1-BIO | Synapse count (SA-001 formula) | No — frozen |
| A1-FROZEN | Xavier uniform random | No — frozen |
| A1-BIAS | Synapse count (SA-001) | Biases only trainable; edges frozen |
| A1-EDGE | Synapse count (SA-001) | Edges only trainable; biases frozen |
| A1-BOTH | Synapse count (SA-001) | Both edges and biases trainable |
| A3-* | Uniform random in [−0.1, 0.1] | Edges trainable; biases trainable |

---

## 5. Data Access Layer

```
abb.data.neuprint_client  →  neuPrint API (NEUPRINT_TOKEN env var only)
                   ↓
abb.data.cache            →  Parquet cache + .provenance.json sidecar
                   ↓
abb.data.graph            →  scipy.sparse.csr_matrix + PyG Data
                   ↓
abb.data.subgraph          →  Subgraph selection with connectivity validation
                   ↓
abb.data.annotations       →  Neuron metadata (type, NT sign, ROI)
```

**Connectivity validation gate:** `abb.data.subgraph` enforces the connectivity
requirements from BENCHMARK_DESIGN.md §4.1 before returning a subgraph.
A subgraph that fails validation is logged and rejected; a new one is sampled.

**Graph representation:**

Edge attributes (per non-zero edge in the subgraph):
- `weight` — raw synapse count [DATASET-FACT]
- `weight_norm` — normalized by scheme in SA-001 ablation config [ASSUMPTION SA-001]
- `sign` — NT-derived sign [ASSUMPTION SA-002]
- `signed_weight` — `sign × weight_norm` (effective model weight for A1-BIO/A1-FROZEN)

Node attributes (per neuron):
- `body_id` — MaleCNS body ID [DATASET-FACT]
- `cell_type` — string cell type label [DATASET-FACT with judgment]
- `predicted_nt` — predicted neurotransmitter [COMPUTATIONAL MODEL]
- `nt_sign` — sign derived from `predicted_nt` [ASSUMPTION SA-002]
- `is_input` — boolean: receives W_in projection for this experiment [ASSUMPTION SA-007-GNN]
- `is_output` — boolean: readout from this node for action head [ASSUMPTION SA-007-GNN]

---

## 6. Environment Interface

```python
# abb/environments/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import gymnasium as gym

class BaseEnvironment(ABC, gym.Env):
    """
    Base class for all ABB task environments.

    Key invariant: observation_space.shape == (D_obs,) where D_obs is a class
    constant. D_obs must not depend on any agent property or subgraph size.
    """

    task_id: str       # class attribute, e.g., 'T-001'
    task_name: str     # class attribute, e.g., 'BinaryPatternDiscrimination'
    d_obs: int         # class attribute; fixed observation dimension

    @abstractmethod
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Returns (obs, info). obs.shape == (d_obs,)."""
        ...

    @abstractmethod
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Returns (obs, reward, terminated, truncated, info)."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Machine-readable config for manifest."""
        ...
```

---

## 7. Training Architecture

### 7.1 Three-Axis Budget Enforcement

The training loop enforces budget consumption tracking for all three axes:

```python
# Conceptual training loop structure (Phase 2 implementation)

budget_tracker = BudgetTracker(
    sample_budget=config.sample_budget,
    compute_budget_flops=config.compute_budget_flops,
    wall_clock_budget_hours=config.wall_clock_budget_hours,
)

while budget_tracker.within_budget(axis=config.active_budget_axis):
    obs, info = env.reset(seed=episode_seed)
    agent.reset()
    while not done:
        agent.observe(obs)
        agent.step()
        action = agent.act()
        obs, reward, done, _, info = env.step(action)
        buffer.add(obs, action, reward, done)
        budget_tracker.record_sample()
        budget_tracker.record_flops(agent.flops_per_step)

    if buffer.ready():
        loss = update_agent(agent, buffer, optimizer)

    if budget_tracker.eval_due():
        metrics = evaluator.evaluate(agent, env, n_episodes=config.eval_episodes)
        results_writer.write(budget_tracker.current_budget(), metrics, manifest)

    checkpointer.save_if_due(agent, budget_tracker)
```

### 7.2 Hyperparameter Search Integration

```
configs/search_spaces/<arch_id>.yaml
         ↓
abb.training.hparam_search
         ↓  (20 trials, dev split, random search Phase 2)
best_config.yaml  →  stored in manifest
         ↓
evaluation runs (all seeds, evaluation split)
```

The hyperparameter search CONSUMES sample budget from the architecture's allocation.
This ensures the total budget (including search overhead) is equal across architectures.

### 7.3 Determinism Policy

```python
# abb/utils/reproducibility.py — Phase 2 implementation
import torch, numpy as np, random

def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)       # raises on non-deterministic ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

If `torch.use_deterministic_algorithms(True)` raises on a specific operation, that
operation must be replaced with a deterministic alternative. Non-deterministic fallbacks
are not permitted in benchmark evaluation runs.

---

## 8. Dependency Stack

| Layer | Library | Version Policy |
|---|---|---|
| Python | ≥ 3.11 | Pinned in `pyproject.toml` |
| Deep learning | PyTorch ≥ 2.1 | Pin exact version per benchmark version |
| Graph neural networks | PyTorch Geometric (PyG) ≥ 2.4 | Pin per benchmark version |
| Sparse matrices | SciPy ≥ 1.11 | Pin per benchmark version |
| RL environments | Gymnasium ≥ 0.29 | Pin per benchmark version |
| Data access | neuprint-python (pin exact) | Pin per benchmark version |
| Data frames | Pandas ≥ 2.0 + PyArrow | Pin per benchmark version |
| Config validation | Pydantic ≥ 2.0 | Pin per benchmark version |
| Visualization | Matplotlib ≥ 3.8, seaborn | Pin per benchmark version |
| Reproducible environments | Docker (see `docker/Dockerfile`) | Created Phase 2 |
| Testing | pytest ≥ 8.0 | Latest compatible with pinned stack |

Lock file: `uv.lock` (preferred) or `requirements-lock.txt`. One lock file per
benchmark version. Lock files are committed to Git.

---

## 9. Key Design Decisions (Revised)

| Decision | Rationale | Alternative Considered |
|---|---|---|
| Custom signed sparse linear as primary GNN op | Preserves synapse count weight prior; no spurious degree normalization or attention | GCNConv (normalizes by degree — anti-biological), GATConv (adds parameters that override weight prior) |
| D_obs fixed per task; not per subgraph | Ensures task is identical across architectures and subgraph sizes | Dynamic D_obs = n_input_neurons (creates incomparable tasks across scales) |
| Three-axis budget comparison | No single axis is "fair"; multi-axis reveals Pareto tradeoffs | Single matched-parameter comparison (uninterpretable — CRIT-03) |
| Docker in Phase 2 | Reproducibility must exist from the first experiment, not deferred to publication | Docker in Phase 4 (Phase 2/3 results would be unreproducible) |
| Hyperparameter search per architecture | Removes hidden lr-tuning bias | Universal lr=1e-3 default (biases well-conditioned models — HIGH-05) |
| Bootstrap CI for RL results | RL rewards are non-normal; t-interval CI is inappropriate | t-interval CI (invalid for RL reward distributions) |

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | HIGH-01: primary GNN operator changed to custom signed sparse linear; GCNConv/GATConv demoted to labeled ablations; §4 added for operator specification; §4.3 added for weight init by variant; MED-07: file layout expanded to show A1-BIO, A1-FROZEN, A1-BIAS, A1-EDGE, A1-BOTH as separate files; LOW-02: Docker moved to Phase 2 (§2 repo layout + §8); CRIT-04: three-axis budget enforcement in §7.1; HIGH-05: hyperparameter search integration in §7.2; SA-007-GNN/LIF: AbstractAgent step() docstring clarified; `flops_per_step` property added to interface |
