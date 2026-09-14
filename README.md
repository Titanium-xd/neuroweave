# Animal Brain Benchmark (ABB)

> A rigorous, reproducible benchmark for comparing connectome-topology-constrained
> neural networks against conventional artificial neural architectures on identical tasks.

---

## Precise Project Description

ABB tests a single, carefully scoped question per research sub-question (see
[PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md) §3 for exact definitions):

- **PRQ-1 (Inductive Bias):** Is the MaleCNS sparsity pattern a better structural prior
  for gradient-trained networks than random-graph controls or dense baselines?
- **PRQ-2 (Structural Prior):** Does the MaleCNS topology + synapse-count weights, with
  NO gradient training, produce above-chance task performance?
- **PRQ-3 (Topology vs. Degree):** How much of any observed advantage is explained by
  the degree sequence alone vs. the specific wiring?
- **PRQ-4 (Robustness):** Is the trained solution more or less sensitive to post-training
  architectural perturbation?
- **PRQ-5 (Compute Cost):** What is the Pareto frontier of performance vs. compute for
  each architecture family?

**ABB does NOT claim:**
- The model simulates a living fruit fly
- A1-BIO is a biological simulation (it is the most data-direct, least-trained condition)
- Any result implies the fly brain is computationally superior
- Lesion experiments measure biological robustness to cellular damage
- Results generalize to vertebrate or human brains

---

## Documentation

| Document | Description |
|---|---|
| [PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md) | Precise research questions, prohibited claims, architecture matrix, fairness protocol |
| [SCIENTIFIC_ASSUMPTIONS.md](docs/SCIENTIFIC_ASSUMPTIONS.md) | Full assumption register: SA-001 through SA-012, tier classification, ablation plans |
| [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) | MaleCNS identity, CC-BY license, API access, NT accuracy qualification |
| [BENCHMARK_DESIGN.md](docs/BENCHMARK_DESIGN.md) | Task specifications (T-001–T-004c), three-axis fairness protocol, manifest v1.1, metrics |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Repository layout, interfaces, custom sparse operator, Docker policy |
| [RISKS.md](docs/RISKS.md) | Risk register R-001–R-015; corrected R-004 asymmetric confound statement |
| [ROADMAP.md](docs/ROADMAP.md) | Four-phase plan with entry/exit criteria |
| [CHANGELOG_PHASE1_REV1.md](docs/CHANGELOG_PHASE1_REV1.md) | Every change made after adversarial design review |
| [DESIGN_FREEZE_CHECKLIST.md](docs/DESIGN_FREEZE_CHECKLIST.md) | Pre-Phase-2 sign-off checklist |

---

## Data Source

**MaleCNS v1.0** — HHMI Janelia Research Campus FlyEM Project Team
- **Project:** https://male-cns.janelia.org
- **Access:** https://neuprint.janelia.org (dataset: `male-cns:v1.0`)
- **License:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Required citation:** FlyEM Project Team et al., "Sexual dimorphism in the complete
  connectome of the Drosophila male central nervous system," *Cell* (September 3, 2026)

---

## Architecture Comparison Matrix

### Biological Architecture Family

| ID | Name | Weights | Trainable? | Answers |
|---|---|---|---|---|
| A1-BIO | MaleCNS-BIO | Synapse counts | No — frozen | PRQ-2 |
| A1-FROZEN | MaleCNS-FROZEN | Random | No — frozen | PRQ-2 |
| A1-BIAS | MaleCNS-GNN-Bias | Synapse counts | Biases only | PRQ-1 |
| A1-EDGE | MaleCNS-GNN-Edge | Synapse counts | Edges only | PRQ-1, PRQ-3 |
| A1-BOTH | MaleCNS-GNN-Full | Synapse counts | Edges + biases | PRQ-1 |

### Structural Control Graph Family (Topology Hypothesis Tests)

| ID | Preserves from MaleCNS | Randomizes | Tests |
|---|---|---|---|
| A3-ER | Edge count | Everything | Edge count alone |
| A3-CONFIG | Per-node degree sequence | Wiring | Degree sequence vs. wiring |
| A3-SBM | Community structure | Within-block wiring | Community structure |
| A3-DENSE | Node count | Sparsity | Sparsity prior |
| A5-WSHUFFLE | Topology, weight magnitudes | Weight assignments | Distribution vs. topology |
| A5-WSIGN | Topology, synapse counts | NT-derived signs | NT sign structure |
| A5-ETARGET | Topology, per-source magnitudes | Target neuron assignments | Topology vs. weights |

### Conventional Baselines

| ID | Name | Phase |
|---|---|---|
| A0 | Random Policy | Phase 2 |
| A6 | Simple LIF SNN | Phase 3 |
| A7 | MLP | Phase 2 |
| A8 | LSTM/GRU | Phase 2 |
| A9 | CNN (T-003-CONT only) | Phase 3 |
| A10 | Transformer | Phase 4 |

### LIF Spiking Family (Phase 3)

| ID | Name |
|---|---|
| A2-BIO | MaleCNS-LIF-BIO (frozen synapse-count weights) |
| A2-TRAIN | MaleCNS-LIF-Trained (surrogate gradient) |

---

## Task Suite

| ID | Name | D_obs | Type | Phase |
|---|---|---|---|---|
| T-001 | Binary Pattern Discrimination | 64 | Supervised classification | Phase 2 |
| T-002 | Capacity-Limited Sequence Memory | 16 | Working memory | Phase 3 |
| T-003-GRID | Discrete Navigation | 32 | RL | Phase 2 |
| T-003-CONT | Continuous Sensorimotor Integration | 8 | RL | Phase 3 |
| T-004a | Post-Training Node Ablation | — | Meta-task | Phase 3 |
| T-004b | Robustness-Trained Lesioning | — | Meta-task | Phase 3 |
| T-004c | Interface Damage Robustness | — | Meta-task | Phase 3 |

**D_obs is a task constant — fixed observation vector dimension independent of subgraph size.**

---

## Fairness Protocol

No single comparison axis is called "the fair comparison." ABB reports three:

1. **Sample Budget** (same number of training examples) → data efficiency
2. **Compute Budget** (same total FLOPs) → compute efficiency
3. **Wall-Clock Budget** (same training hours on same hardware) → practical efficiency

Performance is reported as a Pareto frontier over parameter counts, not a single
matched-budget point.

---

## Project Status

**Phase 1: Research and Documentation** ✅ Complete (ABB-0.1-rev1)

Phase 2 begins when:
- [ ] neuPrint API access confirmed for `male-cns:v1.0`
- [ ] Minimal test query succeeds
- [ ] NT accuracy specification confirmed from MaleCNS paper methods
- [ ] All team members have signed off on DESIGN_FREEZE_CHECKLIST.md

---

## Technical Principles

- Fixed D_obs interface — all agents receive identical observation vectors; no task changes with subgraph
- Custom signed sparse linear operator — no spurious degree normalization
- Three-axis budget comparison — sample, compute, wall-clock all reported
- Pareto frontier reporting — not single matched-budget points
- Hyperparameter search per architecture — no universal lr default
- 10 seeds for RL tasks; 5 seeds for supervised tasks; bootstrap 95% CI
- ≥3 subgraphs per experiment; subgraph connectivity validated before use
- Docker in Phase 2 — not deferred to publication
- Deterministic ops enforced — `torch.use_deterministic_algorithms(True)`
- All assumptions labeled SA-XXX in code and manifests

---

## License

- **ABB code:** MIT License (to be added in Phase 2)
- **MaleCNS-derived data products:** CC-BY 4.0 (attribution required)

---

## Acknowledgments

This project uses **MaleCNS v1.0** produced by the FlyEM Project Team at HHMI Janelia
Research Campus. We are grateful for their open data release under CC-BY 4.0.
