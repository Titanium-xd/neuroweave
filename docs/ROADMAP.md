# ROADMAP.md
# Animal Brain Benchmark — Development Roadmap

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

---

## Overview

ABB is developed in four phases. Each phase has explicit entry criteria (must be met before
beginning), deliverables, and exit criteria (must be met before moving to the next phase).
Work is ordered by scientific defensibility: the foundation must be solid before implementing
the comparison.

```
Phase 1: Research + Documentation       ← COMPLETE (ABB-0.1-rev1)
Phase 2: Core Framework + GNN Comparison
Phase 3: LIF Simulation + Extended Comparison
Phase 4: Advanced Architectures + Publication Readiness
```

---

## Phase 1 — Research and Scientific Foundation

**Goal:** Establish a scientifically defensible knowledge base before writing simulation code.

**Status:** ✅ COMPLETE (ABB-0.1-rev1 documentation approved)

### Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1.1 | `docs/PROJECT_SCOPE.md` — Research questions, non-claims, architecture matrix | ✅ Complete (rev1) |
| 1.2 | `docs/SCIENTIFIC_ASSUMPTIONS.md` — Assumption register, tier classification | ✅ Complete (rev1) |
| 1.3 | `docs/DATA_PROVENANCE.md` — Dataset identity, license, access, annotation schema | ✅ Complete (rev1) |
| 1.4 | `docs/BENCHMARK_DESIGN.md` — Task suite, evaluation protocol, manifest schema | ✅ Complete (rev1) |
| 1.5 | `docs/ARCHITECTURE.md` — Repository layout, interfaces, technology choices | ✅ Complete (rev1) |
| 1.6 | `docs/RISKS.md` — Risk register with mitigation strategies | ✅ Complete (rev1) |
| 1.7 | `docs/ROADMAP.md` — This file | ✅ Complete (rev1) |
| 1.8 | `docs/CHANGELOG_PHASE1_REV1.md` — Adversarial review change log | ✅ Complete |
| 1.9 | `docs/DESIGN_FREEZE_CHECKLIST.md` — Pre-Phase-2 sign-off document | ✅ Complete |
| 1.10 | Verify neuPrint API access with MaleCNS v1.0 | ⬜ Pending |
| 1.11 | Obtain neuPrint API token; confirm `male-cns:v1.0` is accessible | ⬜ Pending |
| 1.12 | Run a minimal neuprint-python test query (100 neurons + adjacencies) | ⬜ Pending |
| 1.13 | Confirm NT accuracy specification in MaleCNS *Cell* (2026) supplementary materials | ⬜ Pending |

### Phase 1 Exit Criteria

- [ ] All documentation at ABB-0.1-rev1 is internally consistent (cross-document audit complete)
- [ ] DESIGN_FREEZE_CHECKLIST.md is signed off
- [ ] neuPrint API access confirmed for `male-cns:v1.0`
- [ ] Minimal test query succeeds
- [ ] NT accuracy specification confirmed from paper methods
- [ ] All team members have reviewed SCIENTIFIC_ASSUMPTIONS.md

---

## Phase 2 — Core Framework and First GNN Comparison

**Goal:** Implement the full benchmark pipeline and run the first scientifically valid
multi-axis comparison: A1-family vs. A3-family controls vs. A7 (MLP) vs. A8 (LSTM).

**Estimated Duration:** ~8–12 weeks after Phase 1 exit criteria are met

### Phase 2 Entry Criteria

- [ ] All Phase 1 exit criteria met
- [ ] neuPrint API access confirmed

### Phase 2 Deliverables

#### Infrastructure
| # | Deliverable |
|---|---|
| 2.1 | `abb/` Python package scaffolding with `pyproject.toml`; `uv.lock` |
| 2.2 | `docker/Dockerfile` — reproducible environment; built and tested in Phase 2 |
| 2.3 | `abb/data/`: neuprint_client, cache, graph (with connectivity validation), subgraph, annotations |
| 2.4 | `abb/config/`: Pydantic v2 schema + YAML loader |
| 2.5 | `abb/utils/`: reproducibility (deterministic seed), provenance (manifest v1.1), logging, sparse |
| 2.6 | `abb/operators/signed_sparse_linear.py` — custom sparse message-passing operator |
| 2.7 | `scripts/data/download_malecns.py` — full cache reproduction with provenance |
| 2.8 | `tests/unit/` — unit tests for all Phase 2 modules |
| 2.9 | CI pipeline (GitHub Actions): unit tests on every push |

#### Models (Phase 2)
| # | Deliverable |
|---|---|
| 2.10 | `abb/models/base.py` — AbstractAgent (with `flops_per_step` property) |
| 2.11 | `abb/models/biological/malecns_gnn_bio.py` — [A1-BIO] Frozen synapse-count weights |
| 2.12 | `abb/models/biological/malecns_gnn_frozen.py` — [A1-FROZEN] Frozen random weights |
| 2.13 | `abb/models/biological/malecns_gnn_bias.py` — [A1-BIAS] Trainable bias only |
| 2.14 | `abb/models/biological/malecns_gnn_edge.py` — [A1-EDGE] Trainable edge weights |
| 2.15 | `abb/models/biological/malecns_gnn_full.py` — [A1-BOTH] Trainable edges + bias |
| 2.16 | `abb/models/controls/erdos_renyi.py` — [A3-ER] |
| 2.17 | `abb/models/controls/configuration.py` — [A3-CONFIG] Exact degree sequence |
| 2.18 | `abb/models/controls/sbm.py` — [A3-SBM] Stochastic block model |
| 2.19 | `abb/models/controls/dense_gnn.py` — [A3-DENSE] Fully-connected GNN |
| 2.20 | `abb/models/controls/shuffled_weight.py` — [A5-WSHUFFLE] |
| 2.21 | `abb/models/controls/shuffled_sign.py` — [A5-WSIGN] |
| 2.22 | `abb/models/controls/shuffled_target.py` — [A5-ETARGET] |
| 2.23 | `abb/models/baselines/random_policy.py` — [A0] |
| 2.24 | `abb/models/baselines/mlp.py` — [A7] Pareto sweep |
| 2.25 | `abb/models/baselines/lstm.py` — [A8] Pareto sweep |

#### Environments (Phase 2)
| # | Deliverable |
|---|---|
| 2.26 | `abb/environments/base.py` — BaseEnvironment (D_obs invariant enforced) |
| 2.27 | `abb/environments/t001_pattern.py` — [T-001] D_obs=64 |
| 2.28 | `abb/environments/t003_grid.py` — [T-003-GRID] D_obs=32 |

#### Training and Evaluation (Phase 2)
| # | Deliverable |
|---|---|
| 2.29 | `abb/training/supervised.py` — Supervised loop with three-axis budget tracking |
| 2.30 | `abb/training/reinforce.py` — REINFORCE with three-axis budget tracking |
| 2.31 | `abb/training/hparam_search.py` — Random search, 20 trials, dev split |
| 2.32 | `abb/training/checkpointing.py` — Checkpoint + provenance sidecar |
| 2.33 | `abb/evaluation/evaluator.py` — Multi-seed, multi-budget-axis |
| 2.34 | `abb/evaluation/metrics.py` — M-001 through M-013, including AULC |
| 2.35 | `abb/evaluation/pareto.py` — Pareto frontier computation |
| 2.36 | `abb/visualization/`: training curves, comparison bars, pareto frontiers |
| 2.37 | Manifest writer/reader v1.1 |

#### First Benchmark Runs (Phase 2)
| # | Deliverable |
|---|---|
| 2.38 | T-001: A1-BIO, A1-FROZEN, A1-BIAS, A1-EDGE, A1-BOTH vs. A3-ER, A3-CONFIG, A3-SBM, A3-DENSE, A5-WSHUFFLE, A5-WSIGN, A5-ETARGET, A0, A7, A8 (5 seeds, 3 subgraphs, 3 budget axes) |
| 2.39 | T-003-GRID: A1-EDGE, A1-BOTH vs. A3-ER, A3-CONFIG, A0, A7, A8 (10 seeds, 3 subgraphs, 3 budget axes) |
| 2.40 | SA-001 ablation: raw vs. log-scaled vs. normalized vs. binary adjacency on T-001 |
| 2.41 | SA-002 ablation: NT sign vs. learnable sign vs. unsigned on T-001 |
| 2.42 | SA-009 ablation: custom op vs. GCNConv vs. GATConv on T-001 (labeled as ablations) |
| 2.43 | `results/phase2_report.md` — multi-axis comparison with full assumption table |

### Phase 2 Exit Criteria

- [ ] All Phase 2 modules have >80% unit test coverage
- [ ] Integration test: full experiment (train + eval + manifest + plots) completes in <24h on reference hardware
- [ ] All benchmark runs (2.38–2.42) produce results with manifests and plots
- [ ] `results/phase2_report.md` includes full assumption table, sensitivity results, limitations
- [ ] Docker container builds cleanly and reproduces at least one Phase 2 result exactly
- [ ] A0 (random policy) and T-001-CHANCE (majority class) baselines are evaluated and recorded

---

## Phase 3 — LIF Simulation and Extended Comparison

**Goal:** Add LIF agents (A2 family), T-002, T-003-CONT, T-004a/b/c, A6. Full ablation study.
External scientific review.

**Estimated Duration:** ~10–16 weeks after Phase 2 completion

### Phase 3 Entry Criteria

- [ ] All Phase 2 exit criteria met
- [ ] Phase 2 results internally reviewed

### Phase 3 Deliverables

#### LIF Simulation
| # | Deliverable |
|---|---|
| 3.1 | `abb/models/lif/malecns_lif_bio.py` — [A2-BIO] Frozen synapse-count weights, LIF |
| 3.2 | `abb/models/lif/malecns_lif_train.py` — [A2-TRAIN] Trainable, surrogate gradient LIF |
| 3.3 | `abb/models/baselines/simple_snn.py` — [A6] Randomly sparse LIF baseline |
| 3.4 | SA-003 sensitivity: vary τ_m, V_th on T-001, T-002 |
| 3.5 | Comparison A1-EDGE vs. A2-TRAIN on T-001, T-002 — does GNN/LIF approximation matter? |

#### Extended Task Suite
| # | Deliverable |
|---|---|
| 3.6 | `abb/environments/t002_memory.py` — [T-002] D_obs=16; capacity-limited; M-006-BUFFER baseline |
| 3.7 | `abb/environments/t003_cont.py` — [T-003-CONT] D_obs=8; continuous sensorimotor |
| 3.8 | `abb/training/ppo.py` — PPO for T-003-CONT |
| 3.9 | `abb/evaluation/lesion.py` — T-004a, T-004b, T-004c runner |

#### Ablation Studies
| # | Deliverable |
|---|---|
| 3.10 | SA-007-LIF: Poisson rate vs. temporal encoding ablation |
| 3.11 | Subgraph size ablation: 500 / 2,000 / 5,000 / 20,000 neurons |
| 3.12 | T-004a: post-training node ablation on best A1 and A7 (random + activation variance strategies) |
| 3.13 | T-004b: dropout-trained robustness on A1-EDGE vs. A7 |
| 3.14 | T-004c: interface damage robustness on all Phase 2 architectures |
| 3.15 | Noise robustness: observation noise σ ∈ {0, 0.1, 0.3, 0.5} on T-001 |

#### Phase 3 Results
| # | Deliverable |
|---|---|
| 3.16 | Full comparison on T-001, T-002, T-003-GRID, T-003-CONT |
| 3.17 | `results/phase3_report.md` — comprehensive results with all ablations |
| 3.18 | External review by ≥1 computational neuroscientist before any public release |

### Phase 3 Exit Criteria

- [ ] LIF simulation runs stably for ≤5,000-neuron subgraphs
- [ ] T-002 M-006-BUFFER baseline computed and all architectures compared against it
- [ ] T-004a/b/c all completed with architecture-appropriate strategies documented
- [ ] Phase 3 report reviewed by at least one external expert
- [ ] No undisclosed assumptions in any code path

---

## Phase 4 — Advanced Architectures and Publication Readiness

**Goal:** Complete the full architecture comparison matrix; prepare for public release.

**Estimated Duration:** ~12–20 weeks after Phase 3 completion

### Phase 4 Entry Criteria

- [ ] All Phase 3 exit criteria met
- [ ] Phase 3 results passed external review

### Phase 4 Deliverables

#### Advanced Architectures
| # | Deliverable |
|---|---|
| 4.1 | `abb/models/baselines/cnn.py` — [A9] Convolutional baseline (T-003-CONT only) |
| 4.2 | `abb/models/baselines/transformer.py` — [A10] Transformer baseline |
| 4.3 | Per-cell-type LIF parameter fitting from published *Drosophila* electrophysiology literature |

#### Larger Scale and Cross-Connectome
| # | Deliverable |
|---|---|
| 4.4 | Subgraph experiments at 20,000+ neurons |
| 4.5 | FlyWire (female) cross-connectome comparison |
| 4.6 | MANC VNC-only vs. brain-only subgraph comparison |

#### Benchmark Release
| # | Deliverable |
|---|---|
| 4.7 | `CITATIONS.bib` — complete, verified citations for all referenced papers |
| 4.8 | `README.md` — complete user-facing documentation |
| 4.9 | PyPI package publication (`abb`) |
| 4.10 | Dataset card (HuggingFace format or equivalent) |
| 4.11 | ABB benchmark paper draft |

### Phase 4 Exit Criteria

- [ ] Full A0–A10 comparison on T-001, T-002, T-003-GRID, T-003-CONT complete
- [ ] External replication: another team reproduces ≥1 Phase 2 result from scratch
- [ ] All paper claims have corresponding manifest + result files
- [ ] No assumption left unlabeled in any code path

---

## Dependency Graph (Phase 2–4)

```
Phase 1: docs/ ────────────────────────────────────────────────────────►
                                                                        │
Phase 2: data + operator + A1/A3/A5/A0/A7/A8 + T001/T003-GRID ───────►
                                                                        │
Phase 3: A2/A6 + T002/T003-CONT/T004a-c + ablations ─────────────────►
                                                                        │
Phase 4: A9/A10 + large-scale + cross-connectome + publication ────────►
```

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | MED-01: A6 correctly placed in Phase 3 (not Phase 2); LOW-02: Docker moved to Phase 2 deliverables (2.2); Phase 2 deliverables expanded to include full A1-family (A1-BIO, A1-FROZEN, A1-BIAS, A1-EDGE, A1-BOTH), full A3/A5 control set, A0 random policy; T-002 and T-003-CONT moved to Phase 3; T-004 expanded to T-004a/b/c in Phase 3; Phase 1 deliverables updated to reflect design review documents (1.8, 1.9); NT accuracy verification added as Phase 1 task (1.13); Phase 2 exit criteria updated to require Docker and trivial baselines |
