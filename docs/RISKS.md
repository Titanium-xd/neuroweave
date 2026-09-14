# RISKS.md
# Animal Brain Benchmark — Risk Register

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

Risks are classified by:
- **Likelihood:** Low / Medium / High
- **Impact:** Low / Medium / High / Critical
- **Type:** Scientific / Engineering / Data / Reproducibility / Ethical

---

## Part I: Scientific / Validity Risks

### R-001 — Results Reflect Assumptions, Not Topology
**Type:** Scientific
**Likelihood:** High
**Impact:** Critical
**Description:** Any observed advantage (or disadvantage) of the biological connectome
architecture may be an artifact of the engineering assumptions (SA-001 through SA-012),
not a property of the biological topology itself.

Examples:
- If 15% of edges have the wrong sign (SA-002), we are testing a corrupted topology.
- If uniform LIF parameters are a poor fit (SA-003), we are not testing what the wiring can do.
- If the custom sparse linear operator has subtle numerical issues, results reflect
  implementation, not biology.

**Mitigation:**
1. Sensitivity analysis: vary each assumption independently; measure effect on results
2. Every result report includes a mandatory assumption sensitivity table
3. Ablation experiments for SA-001, SA-002, SA-009 are Phase 2 deliverables
4. Results are qualified with the specific assumption set in use; never stated as properties
   of "biological topology" in general

---

### R-002 — Subgraph Selection Bias
**Type:** Scientific
**Likelihood:** Medium
**Impact:** High
**Description:** If the selected subgraph happens to be particularly well-suited (or ill-suited)
to the task by coincidence, results will not generalize.

**Mitigation:**
1. Minimum 3 independent subgraph selections per experiment; variance across subgraphs reported
2. Both functional and connectivity-respecting random subgraphs used
3. Subgraph connectivity statistics required before use (BENCHMARK_DESIGN.md §4.1)
4. Subgraph identity recorded by hash in every manifest

---

### R-003 — Task Selection Does Not Probe Any Biological Advantage
**Type:** Scientific
**Likelihood:** Medium
**Impact:** High
**Description:** The biological connectome evolved for specific *Drosophila* functions, not
our benchmark tasks. A null result on ABB tasks does NOT mean biological topology is
computationally uninformative.

**Mitigation:**
1. Task suite includes structurally varied tasks (classification, memory, navigation, control)
2. T-003-CONT is selected for structural similarity to VNC motor function
3. All null results are framed as "no advantage on this task suite under these assumptions,"
   not as "biological topology is useless"
4. Phase 4 includes tasks more directly inspired by *Drosophila* behavior

---

### R-004 — Reconstruction Errors in MaleCNS Data
**Type:** Scientific / Data
**Likelihood:** Medium
**Impact:** Medium
**Description:** Residual merge and split errors in MaleCNS v1.0 corrupt the true connectivity
graph.

> [!CAUTION]
> **Revised mitigation statement (ABB-0.1-rev1):** Reconstruction errors affect
> **only the biological models (A1, A2)**, not the synthetic control models (A3–A10).
> The control graphs are clean synthetic graphs with no reconstruction noise. This creates
> an asymmetric confound: any observed **disadvantage** of A1/A2 relative to controls could
> partly reflect reconstruction noise in the input topology, not biological topology properties.
> This asymmetry **cannot** be resolved without error-corrected data. It must be stated as a
> fundamental limitation in all result publications.
>
> **The previous statement "relative comparisons are still valid because reconstruction
> errors affect all models uniformly" was INCORRECT and has been removed.**

**Mitigation:**
1. Accept v1.0 as-is; document asymmetric confound explicitly in every result report
2. Track dataset version in all experiments; re-run if a corrected version is released
3. If correction annotations become available, test with cleaned subset and report delta

---

### R-005 — Neurotransmitter Prediction Errors
**Type:** Scientific
**Likelihood:** Medium
**Impact:** Medium
**Description:** NT predictions have published accuracy of ~87% per synapse / ~94% per neuron
on FAFB/FlyWire (serial-section TEM, same modality as MaleCNS) and ~78% per synapse / ~91%
per neuron on Hemibrain (FIB-SEM). The accuracy for MaleCNS v1.0 specifically is not
independently validated (CLOSED-AS-UNAVAILABLE). Using the FAFB proxy: ~6–13% of edges
may have the wrong sign.

**Mitigation:**
1. SA-002-ALT ablation: treat all edge signs as learnable parameters; compare performance
2. SA-002-NULL ablation: remove sign information entirely; compare performance
3. A5-WSIGN control: compare against shuffled-sign version of the same topology
4. Report sensitivity of results to sign assignment rule

---

### R-006 — Scale Infeasibility for LIF Simulation
**Type:** Engineering
**Likelihood:** High
**Impact:** High
**Description:** Simulating LIF at scale (thousands of neurons × millions of synapses × ms
time steps) is computationally expensive.

**Mitigation:**
1. Phase 2 uses GNN approximation (SA-008/SA-009); LIF reserved for Phase 3
2. Phase 3 LIF limited to ~5,000-neuron subgraphs
3. Efficient LIF backends evaluated: custom PyTorch sparse, Brian2CUDA, GeNN
4. Phase 4 stretch: Intel Loihi 2 for larger-scale LIF (hardware deployment, not a Phase 2 claim)

---

### R-007 — Comparison Fairness Cannot Be Reduced to a Single Axis
**Type:** Scientific
**Likelihood:** High (this is known by design)
**Impact:** Medium
**Description:** No single training budget axis (sample, compute, or wall-clock) produces
a fully "fair" comparison. Different axes will favor different architectures.

**Mitigation:**
1. Three-axis comparison protocol is the primary design response (see BENCHMARK_DESIGN.md §3.1)
2. Pareto frontier over parameter counts reported per architecture family
3. All three axes are always reported; no single axis is declared "the" fair comparison
4. The limitation is stated explicitly in every comparison report

---

### R-008 — Gradient-Based Training Is Not Biologically Realistic
**Type:** Scientific
**Likelihood:** High (known by design)
**Impact:** Interpretive
**Description:** Training with gradient descent has no biological analogue.

**Mitigation:**
1. ABB asks "can this topology learn X?" not "does the fly brain do X?" — stated in every report
2. A1-BIO and A1-FROZEN conditions evaluate topology without any gradient training
3. Fixed-weight conditions vs. trained-weight conditions are always compared and labeled separately

---

### R-009 — neuPrint API Changes or Downtime
**Type:** Engineering / Data
**Likelihood:** Low
**Impact:** Medium
**Description:** API schema changes or unavailability would break data access.

**Mitigation:**
1. All data cached locally before experiments begin; experiments run from cache
2. `scripts/data/download_malecns.py` reproduces cache from scratch
3. Exact `neuprint-python` version pinned per experiment
4. If API unavailable, experiments run from local cache with no degradation

---

### R-010 — Reproducibility Failure Due to GPU Non-Determinism
**Type:** Reproducibility
**Likelihood:** Medium
**Impact:** Medium
**Description:** GPU scatter/gather ops (used in sparse message passing) may be non-deterministic
across runs with the same seed.

**Mitigation:**
1. `torch.use_deterministic_algorithms(True)` enforced in all evaluation runs
2. Non-deterministic ops replaced with deterministic alternatives during implementation
3. CPU evaluation mode available for strict reproducibility verification
4. Each seed evaluated 2× in CI; flag if results differ beyond float32 precision

---

### R-011 — Overfitting to Benchmark Tasks
**Type:** Scientific
**Likelihood:** Low (Phase 2 is novel)
**Impact:** High
**Description:** Using the same tasks to select hyperparameters AND to report final performance
produces optimistic results.

**Mitigation:**
1. Hyperparameter search uses development split (not evaluation split) — enforced by design
2. Evaluation split is held out and used only for final reported numbers
3. Task suite updates require a MAJOR benchmark version bump (BENCHMARK_DESIGN.md §1.4)
4. All hyperparameter search logs are recorded in the manifest

---

### R-012 — Misuse: Overclaiming Biological Relevance
**Type:** Ethical / Scientific Communication
**Likelihood:** Medium
**Impact:** Critical
**Description:** ABB results may be misrepresented as proving "the fly brain is smarter
than AI" or "we have simulated a fly brain."

**Mitigation:**
1. PROJECT_SCOPE.md §4 (prohibited claims table) governs all communications
2. All result reports have a mandatory "Limitations and Assumptions" section
3. A1-BIO is described as "most data-direct, least-trained connectome-derived condition" —
   never as "biological brain simulation"
4. Lesion results are described as "post-training sensitivity to architectural perturbation" —
   never as "biological robustness to cellular damage"

---

### R-013 — Single Specimen Data
**Type:** Scientific
**Likelihood:** High (fact, not risk)
**Impact:** Medium
**Description:** MaleCNS v1.0 is from one individual. Our "biological architecture" may be
idiosyncratic to this specimen.

**Mitigation:**
1. Stated in all publications and DATA_PROVENANCE §1.2
2. Phase 4: FlyWire (female) cross-comparison to test inter-individual generality
3. Phase 4 stretch: MANC VNC-only as intra-sex structural variant

---

### R-014 — Software Dependency Decay
**Type:** Engineering
**Likelihood:** Medium (multi-year project)
**Impact:** Medium
**Description:** PyTorch Geometric, Gymnasium, and neuprint-python evolve and may break.

**Mitigation:**
1. All experiments pin exact library versions
2. `uv.lock` for reproducible environments
3. Docker container created in Phase 2 (not deferred to Phase 4)
4. Dependency audit at start of each phase

---

### R-015 — Custom Sparse Operator Has Hidden Performance Costs
**Type:** Engineering
**Likelihood:** Medium
**Impact:** Medium
**Description:** The custom signed sparse linear operator (SA-009) bypasses optimized
cuSPARSE/PyG kernels. It may be slower than GCNConv/GATConv on GPU due to less aggressive
kernel fusion and less cache-aware memory access patterns.

**Mitigation:**
1. Benchmark custom operator against PyG standard ops in Phase 2 infrastructure work
2. If performance gap is large (>10× per step), investigate sparse CUDA kernel optimization
3. Wall-clock time is reported as M-010 for all architectures; performance vs. time tradeoff
   is transparent to readers
4. If custom operator is unacceptably slow, use GATConv with forced weight initialization
   from synapse counts and frozen attention — a documented compromise labeled [SA-009-FALLBACK]

---

## Part II: Risk Summary Matrix

| ID | Risk | Likelihood | Impact | Priority |
|---|---|---|---|---|
| R-001 | Results reflect assumptions, not topology | High | Critical | 🔴 Critical |
| R-002 | Subgraph selection bias | Medium | High | 🔴 High |
| R-003 | Task selection misses biological advantage | Medium | High | 🔴 High |
| R-004 | Reconstruction errors (asymmetric confound) | Medium | Medium | 🟡 Medium |
| R-005 | NT prediction errors | Medium | Medium | 🟡 Medium |
| R-006 | LIF simulation infeasibility | High | High | 🔴 High |
| R-007 | No single fair comparison axis | High (design) | Medium | 🟡 Medium |
| R-008 | Training not biologically realistic | High (design) | Interpretive | 🟡 Medium |
| R-009 | neuPrint API unavailability | Low | Medium | 🟢 Low |
| R-010 | GPU non-determinism | Medium | Medium | 🟡 Medium |
| R-011 | Overfitting to benchmark tasks | Low | High | 🟡 Medium |
| R-012 | Overclaiming biological relevance | Medium | Critical | 🔴 Critical |
| R-013 | Single specimen limitation | High (fact) | Medium | 🟡 Medium |
| R-014 | Software dependency decay | Medium | Medium | 🟡 Medium |
| R-015 | Custom operator performance costs | Medium | Medium | 🟡 Medium |

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | MED-05: R-004 mitigation completely rewritten — the previous claim that "reconstruction errors affect all models uniformly so relative comparisons are valid" was incorrect; errors affect only biological models; asymmetric confound documented with CAUTION alert; R-007 updated to reflect three-axis protocol; R-008 updated to reference A1-BIO/A1-FROZEN fixed-weight conditions; R-012 mitigation updated with new A1-BIO and T-004 prohibited claim language; R-015 added for custom sparse operator performance risk |
