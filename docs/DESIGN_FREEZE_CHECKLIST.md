# DESIGN_FREEZE_CHECKLIST.md
# Animal Brain Benchmark — Phase 1 Design Freeze Checklist

> **Version:** ABB-0.1-rev1
> **Date:** 2026-09-14
> **Purpose:** This document must be fully checked and signed off before any Phase 2
> implementation code is written.

No simulation code, GNN, environment, or ML dependency installation may occur until
every item in this checklist is marked complete and the document is signed off.

---

## Section A: Cross-Document Consistency Audit

Each item below verifies that a specific cross-document relationship is consistent.

### A1 — Architecture IDs Are Consistent Across All Documents

| Architecture ID | PROJECT_SCOPE §6 | BENCHMARK_DESIGN §2 | ARCHITECTURE §2 | ROADMAP Phase | README |
|---|---|---|---|---|---|
| A0 (Random Policy) | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A1-BIO | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A1-FROZEN | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A1-BIAS | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A1-EDGE | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A1-BOTH | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A2-BIO | ✅ | — | ✅ | 3 ✅ | ✅ |
| A2-TRAIN | ✅ | — | ✅ | 3 ✅ | ✅ |
| A3-ER | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A3-CONFIG | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A3-SBM | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A3-DENSE | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A5-WSHUFFLE | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A5-WSIGN | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A5-ETARGET | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A6 (Simple LIF SNN) | ✅ Phase 3 | — | ✅ | 3 ✅ | ✅ |
| A7 (MLP) | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A8 (LSTM) | ✅ | ✅ | ✅ | 2 ✅ | ✅ |
| A9 (CNN) | ✅ Phase 3 | — | ✅ | 3 ✅ | ✅ |
| A10 (Transformer) | ✅ Phase 4 | — | ✅ | 4 ✅ | ✅ |

**Status:** ✅ All architecture IDs consistent. Barabási–Albert (A4) correctly removed
from all documents. A6 is Phase 3 in all documents (MED-01 resolved).

---

### A2 — Task IDs Are Consistent Across All Documents

| Task ID | PROJECT_SCOPE §7 | BENCHMARK_DESIGN §2 | ROADMAP | README |
|---|---|---|---|---|
| T-001 (D_obs=64) | ✅ | ✅ | Phase 2 ✅ | ✅ |
| T-002 (D_obs=16) | ✅ | ✅ | Phase 3 ✅ | ✅ |
| T-003-GRID (D_obs=32) | ✅ | ✅ | Phase 2 ✅ | ✅ |
| T-003-CONT (D_obs=8) | ✅ | ✅ | Phase 3 ✅ | ✅ |
| T-004a (post-training ablation) | ✅ | ✅ | Phase 3 ✅ | ✅ |
| T-004b (trained robustness) | ✅ | ✅ | Phase 3 ✅ | ✅ |
| T-004c (interface damage) | ✅ | ✅ | Phase 3 ✅ | ✅ |

**Status:** ✅ Task IDs consistent. T-004 (monolithic) removed from all documents;
T-004a/b/c used everywhere.

---

### A3 — Assumption IDs Are Consistent and Complete

| Assumption ID | SCIENTIFIC_ASSUMPTIONS.md | Code Labels | Manifest References |
|---|---|---|---|
| SA-001 | ✅ | `[SA-001]` | manifest `assumptions_applied` |
| SA-002 + ablations (ALT/NULL/RAND) | ✅ | `[SA-002]` | ✅ |
| SA-003 | ✅ | `[SA-003]` | ✅ |
| SA-004 | ✅ | `[SA-004]` | ✅ |
| SA-005 | ✅ | `[SA-005]` | ✅ |
| SA-006 | ✅ | `[SA-006]` | ✅ |
| SA-007-GNN (new; split from SA-007) | ✅ | `[SA-007-GNN]` | ✅ |
| SA-007-LIF (new; split from SA-007) | ✅ | `[SA-007-LIF]` | ✅ |
| SA-007 (original monolithic) | ❌ REMOVED — split into SA-007-GNN + SA-007-LIF | — | — |
| SA-008 | ✅ | `[SA-008]` | ✅ |
| SA-009 (new; custom operator) | ✅ | `[SA-009]` | ✅ |
| SA-009-GCN (ablation variant) | ✅ | `[SA-009-GCN]` | ✅ |
| SA-009-GAT (ablation variant) | ✅ | `[SA-009-GAT]` | ✅ |
| SA-010 | ✅ | `[SA-010]` | ✅ |
| SA-011 | ✅ | `[SA-011]` | ✅ |
| SA-012 | ✅ | `[SA-012]` | ✅ |

**Status:** ✅ Assumption IDs consistent. Original SA-007 removed and replaced by split
pair. SA-009 added for custom operator.

---

### A4 — Metrics Are Consistent and Correctly Applied

| Metric ID | Defined In | Task Applicability (BENCHMARK_DESIGN §6) | Applies to T-002 RL? |
|---|---|---|---|
| M-001 (Mean Episode Reward) | ✅ | T-003-GRID, T-003-CONT ONLY | ❌ No (T-002 is supervised) |
| M-002 (Classification Accuracy) | ✅ | T-001, T-002 | — |
| M-003 (AUC-ROC) | ✅ | T-001 | — |
| M-004 (Steps to Goal) | ✅ | T-003-GRID | — |
| M-005 (Success Rate) | ✅ | T-003-GRID, T-003-CONT | — |
| M-006 (Memory Duration) | ✅ | T-002 | — |
| M-006-BUFFER (Buffer Ceiling) | ✅ | T-002 | — |
| M-007 (AULC) | ✅ | All tasks | — |
| M-008 (Lesion Robustness AUC) | ✅ | T-004a, T-004b | — |
| M-009 (Inference FLOPs) | ✅ | All | — |
| M-010 (Wall Time) | ✅ | All | — |
| M-011 (Peak Memory) | ✅ | All | — |
| M-012 (Training FLOPs) | ✅ | All (compute budget axis) | — |
| M-013 (Pareto Efficiency) | ✅ | All | — |

**Status:** ✅ M-001 no longer incorrectly applied to T-002 (HIGH-03 resolved). Circular
M-007 definition replaced with AULC (LOW-04 resolved). M-006-BUFFER added (CRIT-06).

---

### A5 — Research Questions Are Answered by Specific Experimental Conditions

| Research Question | Answered By | Controls Present? |
|---|---|---|
| PRQ-1 (Inductive bias via gradient training) | A1-BIAS/EDGE/BOTH vs. A3-ER/CONFIG/SBM/DENSE/A7/A8 | ✅ |
| PRQ-2 (Structural prior, no training) | A1-BIO vs. A1-FROZEN vs. A0 | ✅ |
| PRQ-3 (Topology vs. degree sequence) | A1-EDGE vs. A3-CONFIG | ✅ |
| PRQ-4 (Robustness) | T-004a/b/c on A1 vs. A7/A8 | ✅ |
| PRQ-5 (Compute cost) | M-009/M-010/M-011 + Pareto frontier | ✅ |

**Status:** ✅ Every research question has dedicated experimental conditions.

---

### A6 — Fairness Protocol Is Consistent Across Documents

| Fairness Rule | PROJECT_SCOPE §5 | BENCHMARK_DESIGN §3.1 | ARCHITECTURE §7.1 |
|---|---|---|---|
| Three-axis budget comparison | ✅ | ✅ | ✅ |
| Pareto frontier reporting | ✅ | ✅ | ✅ |
| Hyperparameter search (20 trials, dev split) | ✅ | ✅ | ✅ |
| No universal lr default | ✅ | ✅ | ✅ |
| 10 seeds for RL, 5 for supervised | ✅ | ✅ | — |
| Bootstrap CI, B=2000 | — | ✅ | — |

**Status:** ✅ Three-axis protocol consistent. No document retains the universal
Adam lr=1e-3 as the evaluation default (HIGH-05 resolved).

---

### A7 — Prohibited Claims Are Consistent Across Documents

| Prohibited Claim | PROJECT_SCOPE §4 | DATA_PROVENANCE §7 | RISKS R-012 |
|---|---|---|---|
| "Simulates a living fly" | ✅ Prohibited | ✅ | ✅ |
| "A1-BIO is a biological simulation" | ✅ Prohibited | ✅ | ✅ |
| "Better score = fly brain superior" | ✅ Prohibited | — | ✅ |
| "T-004 measures biological robustness to cellular damage" | ✅ Prohibited | — | ✅ |
| "Consciousness / sentience" | ✅ Prohibited | ✅ | — |
| "Results generalize to vertebrate brains" | ✅ Prohibited | ✅ | — |

**Status:** ✅ Prohibited claims consistent. A1-BIO described as "most data-direct,
least-trained condition" in all documents.

---

### A8 — Reconstruction Error Asymmetry Is Consistently Documented

| Location | Correctly States Asymmetric Confound? |
|---|---|
| DATA_PROVENANCE.md §5.1 | ✅ |
| RISKS.md R-004 | ✅ (with CAUTION alert) |
| SCIENTIFIC_ASSUMPTIONS.md BF-001 | ✅ |

**Status:** ✅ The incorrect claim that "relative comparisons are valid because errors
affect all models uniformly" is removed from all documents (MED-05 resolved).

---

### A9 — Observation Dimension (D_obs) Is Architecture-Independent

| Location | D_obs Stated as Task Constant? |
|---|---|
| BENCHMARK_DESIGN.md §2.1 (design invariant) | ✅ |
| T-001 spec (D_obs=64) | ✅ |
| T-002 spec (D_obs=16) | ✅ |
| T-003-GRID spec (D_obs=32) | ✅ |
| T-003-CONT spec (D_obs=8) | ✅ |
| ARCHITECTURE.md AbstractAgent observe() docstring | ✅ |
| PROJECT_SCOPE.md §8 interface spec | ✅ |
| README.md task table | ✅ |

**Status:** ✅ D_obs is consistently a task constant in all documents (CRIT-05 resolved).

---

### A10 — neuPrint Token Policy Is Consistent

| Location | Uses env var only? | Explicit error on missing token? |
|---|---|---|
| DATA_PROVENANCE.md §3.1 | ✅ | ✅ |
| ARCHITECTURE.md data layer description | ✅ (references DATA_PROVENANCE) | ✅ |
| RISKS.md (no token policy needed) | — | — |

**Status:** ✅ Token policy consistent. "Secrets manager" option removed (MED-06 resolved).

---

### A11 — Benchmark Versioning Is Defined and Referenced

| Location | Version system defined/referenced? |
|---|---|
| BENCHMARK_DESIGN.md §1.4 | ✅ Defined |
| RISKS.md R-011 | ✅ References BENCHMARK_DESIGN §1.4 |
| All document headers | ✅ Version string "ABB-0.1-rev1" |
| Manifest schema v1.1 | ✅ `benchmark_version` field |

**Status:** ✅ Benchmark versioning defined (LOW-03 resolved). Dangling reference in
RISKS.md now resolves correctly.

---

### A12 — Docker Policy Is Consistent

| Location | Docker placement |
|---|---|
| ARCHITECTURE.md repository layout | ✅ `docker/Dockerfile` in layout |
| ARCHITECTURE.md §8 dependencies | ✅ Listed |
| ROADMAP.md Phase 2 deliverables | ✅ Deliverable 2.2 |
| ROADMAP.md Phase 4 deliverables | ✅ Removed from Phase 4 |

**Status:** ✅ Docker is Phase 2 deliverable (LOW-02 resolved).

---

## Section B: Prohibited Content Scan

The following phrases must not appear in any ABB document in an uncaveated affirmative context.
Search result: ❌ = found and not corrected; ✅ = not found or only found with explicit correction.

| Prohibited Phrase | Scan Result | Notes |
|---|---|---|
| "simulate a living fly" | ✅ Not found uncaveated | Appears only in prohibition lists |
| "simulated a fly brain" | ✅ Not found uncaveated | Appears only in prohibition lists |
| "relative comparisons are still valid" (re: reconstruction errors) | ✅ Removed | Was in RISKS.md R-004; now corrected |
| "Adam lr = 1e-3" (as universal default) | ✅ Removed | Only in hyperparameter search config examples |
| "80% of final performance" (circular M-007) | ✅ Removed | Replaced with AULC |
| "fair comparison" (singular, unqualified) | ✅ Removed | Replaced with "three-axis" language everywhere |
| "type-stratified random sample" (as primary subgraph) | ✅ Demoted | Only appears as secondary/cross-validation use |
| "GCNConv" (as primary A1 operator) | ✅ Demoted | Labeled as ablation variant only |
| "GATConv" (as primary A1 operator) | ✅ Demoted | Labeled as ablation variant only |
| "T-002" listed under M-001 applicability | ✅ Removed | M-001 now correctly T-003 only |
| "Barabási–Albert" (as A4 control) | ✅ Removed | Superseded by A3-CONFIG |

---

## Section C: Outstanding Actions Before Phase 2

The following items require external actions before Phase 2 begins:

| # | Action | Owner | Status |
|---|---|---|---|
| C1 | Obtain neuPrint API token from https://neuprint.janelia.org | Project lead | ✅ **DONE** |
| C2 | Verify `male-cns:v1.0` is queryable with token | Engineer | ✅ **PASS** — confirmed live 2026-09-14; 10 datasets found; `male-cns:v1.0` accessible |
| C3 | Run minimal test query: fetch 100 neurons + adjacencies | Engineer | ✅ **PASS** — 100 neurons, 1,598 connections, 30,613 synapse contacts; both schemas clean; `predictedNt` coverage 99.7% |
| C4 | Confirm NT accuracy specification from MaleCNS *Cell* (2026) methods section | Scientist | ✅ **RESOLVED** — 87%/94% = FAFB/FlyWire (TEM, same modality as MaleCNS); 78%/91% = Hemibrain (FIB-SEM). MaleCNS-specific figures: 🚫 CLOSED-AS-UNAVAILABLE (not publicly reported). DATA_PROVENANCE §4.1 updated. |
| C5 | All team members review and sign off on SCIENTIFIC_ASSUMPTIONS.md | All | ⬜ Pending — team action |
| C6 | All team members acknowledge prohibited claims list (PROJECT_SCOPE.md §4) | All | ⬜ Pending — team action |
| C7 | Identify reference hardware for benchmark runs (GPU model, CPU model, RAM) | Engineer | ✅ **DONE** — see Section E below |
| C8 | Confirm Bayati-Kim-Saberi algorithm implementation is available for A3-CONFIG | Engineer | ✅ **PASS** — `networkx 3.6.1` installed; `networkx.generators.degree_seq.directed_configuration_model` confirmed available 2026-09-14 |
| C9 | Confirm whether NT accuracy breakdown by cell type is available in MaleCNS supplementary | Scientist | 🚫 **CLOSED-AS-UNAVAILABLE** — MaleCNS-specific per-class breakdown not publicly reported. Per-NT-class data (FAFB) exists in Eckstein 2024 supplementary (Zenodo). `predictedNtConfidence` available per neuron (99.2% coverage) for use as empirical quality indicator only, NOT accuracy. |

---

## Section D: Sign-Off

Phase 2 implementation may begin only after all of the following are signed:

| Signoff | Name | Date | Notes |
|---|---|---|---|
| All Section A items verified | *(automated audit)* | 2026-09-14 | Cross-document consistency audit — all items ✅ |
| All Section B prohibited phrases cleared | *(automated grep scan)* | 2026-09-14 | Zero uncaveated prohibited phrases found |
| C1–C4, C7–C9 completed (engineering items) | *(automated tests)* | 2026-09-14 | See Section C above |
| C5: SCIENTIFIC_ASSUMPTIONS.md reviewed by project scientist | *(pending)* | — | Team action required before Phase 2 |
| C6: PROJECT_SCOPE.md §4 prohibited claims acknowledged | *(pending)* | — | Team action required before Phase 2 |
| SCIENTIFIC_ASSUMPTIONS.md reviewed by project scientist | *(pending)* | — | |
| PROJECT_SCOPE.md §4 prohibited claims acknowledged by all contributors | *(pending)* | — | |

> [!NOTE]
> C5 and C6 are the only remaining open items for Phase 2 approval.
> All engineering, data, and documentation prerequisites (C1–C4, C7–C9) are complete.

---

## Section E: Reference Hardware (C7)

**Recorded:** 2026-09-14 — ABB Phase 1 finalization

| Component | Specification |
|---|---|
| **CPU** | Intel Core i5-13450HX (13th gen, 10-core, 16-thread) |
| **RAM** | 24 GB DDR5 4800 MHz |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **VRAM** | 6 GB GDDR6 |
| **OS** | Windows 11 Pro |
| **Form factor** | Laptop |
| **Hardware ID** | `ABB-REF-HW-001` |

> [!IMPORTANT]
> This is the **reference measurement hardware** for ABB Phase 1–2 wall-clock (M-010)
> and peak memory (M-011) reproducibility measurements. It is NOT a minimum requirement.
> Results from other hardware must report their hardware configuration and must not be
> directly compared to ABB-REF-HW-001 timings without normalization.
>
> This hardware specification is **NOT embedded in model logic or hyperparameters.**
> It is a reproducibility metadata record only.

---

## Revision History

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1-rev1 | Initial creation after adversarial design review and documentation revision |
| 2026-09-14 | ABB-0.1-rev1 (Phase 1 freeze) | Section C updated with final preflight results (C1–C4 ✅, C7 ✅, C8 ✅, C9 🚫 CLOSED-AS-UNAVAILABLE); Section E added with reference hardware (ABB-REF-HW-001); Section D sign-off table updated to reflect engineering items complete, C5/C6 pending team action |
