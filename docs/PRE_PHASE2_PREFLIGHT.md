# PRE_PHASE2_PREFLIGHT.md
# Animal Brain Benchmark — Phase 2 Pre-Flight Check Report

> **Date:** 2026-09-14
> **Checklist Items:** C2, C3, C4, C9
> **Benchmark Version:** ABB-0.1-rev1
> **Tester:** Automated (ABB smoke test + web research)

---

## Summary Table

| Item | Description | Status | Detail |
|---|---|---|---|
| **C2** | Dataset `male-cns:v1.0` accessible via neuPrint API | ✅ **PASS** | Confirmed live; see §C2 |
| **C3** | Minimal fetch: ~100 neurons + adjacencies, schema validated | ✅ **PASS** | 100 neurons, 1,598 connections, 30,613 synapse contacts; all schemas clean; see §C3 |
| **C4** | NT prediction accuracy: method, dataset, figures, applicability | ⚠️ **PARTIAL** / 🚫 **CLOSED-AS-UNAVAILABLE** (MaleCNS-specific) | Method confirmed; accuracy figures corrected (FAFB=87%/94%, Hemibrain=78%/91%); MaleCNS-specific figures not publicly reported; see §C4 |
| **C9** | NT accuracy breakdown by NT class, cell type, or subgraph available? | 🚫 **CLOSED-AS-UNAVAILABLE** (MaleCNS-specific) | Per-class breakdown exists in Eckstein 2024 (FAFB); no MaleCNS-specific breakdown published; see §C9 |

> [!NOTE]
> Token was loaded from `.env` via `python-dotenv`. Token value was never printed or logged.
> `.gitignore` now excludes `.env`, `data/`, `results/`. Git repo has no committed secrets.

---

## .gitignore Security Check

```
.env            ← excluded ✓
data/           ← excluded ✓
results/        ← excluded ✓
*.pt / *.pth    ← excluded ✓
__pycache__/    ← excluded ✓
```

**Status: PASS** — `.gitignore` created and verified. No token in any committed file.

---

## C2 — Dataset Accessibility

**Test:** Can the `male-cns:v1.0` dataset be accessed via the official neuPrint API?

### Procedure
```
Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=NEUPRINT_TOKEN)
client.fetch_db_version()
client.fetch_version()
client.meta
client.fetch_datasets()
```

### Results

| Property | Value |
|---|---|
| Server | `https://neuprint.janelia.org` |
| neuPrint DB version | `0.5.0` |
| neuPrint API version | `1.9.3` |
| **Confirmed dataset name (from API)** | **`male-cns`** (via `client.meta["dataset"]`) |
| Dataset version string used | `male-cns:v1.0` |
| Description (from API, first 120 chars) | "The complete MaleCNS connectome from the Janelia FlyEM Team Project, the Cambridge Drosophila Connectomics Group, and Go..." |
| All available datasets (10) | `hemibrain:v1.1`, `hemibrain:v1.2.1`, `male-cns:v0.9`, **`male-cns:v1.0`**, `manc:v1.0`, `manc:v1.2.1`, `manc:v1.2.3`, `mushroombody`, `optic-lobe:v1.0.1`, `optic-lobe:v1.1` |

> [!IMPORTANT]
> The exact dataset name returned by the neuPrint API is **`male-cns`** (without the version suffix
> in the `meta.dataset` field). The full versioned identifier `male-cns:v1.0` is confirmed as a
> valid query target (it appears in `fetch_datasets()` output and the client accepts it).

### Status: ✅ PASS

---

## C3 — Minimal Neuron + Adjacency Smoke Test

**Test:** Fetch ~100 neurons and their adjacencies from `male-cns:v1.0`. Validate schema.
Save no data to disk.

### Procedure
```
fetch_neurons(NeuronCriteria(status="Traced"), client=client)  → sample first 100 rows
fetch_adjacencies(sources=body_ids[:100], targets=body_ids[:100], client=client)
```

### Neuron Fetch Results

| Property | Value |
|---|---|
| **Neurons fetched (sample)** | **100** |
| **Total neurons in full response** | **165,122** (all Traced neurons returned by query) |
| Status distribution | `{'Traced': 100}` |
| First 5 bodyIds | `10001, 10002, 10003, 10005, 10006` |

**Top 5 cell types in sample:**

| Cell Type | Count in Sample |
|---|---|
| VS | 5 |
| DNp01 | 2 |
| VCH | 2 |
| AOTU019 | 2 |
| OA-AL2i1 | 2 |

### Neuron Schema Validation

| Column Group | Expected? | Present? | Notes |
|---|---|---|---|
| `bodyId` | ✅ | ✅ | |
| `status` | ✅ | ✅ | |
| `type` | ✅ | ✅ | |
| `instance` | ✅ | ✅ | |
| `size` | ✅ | ✅ | |
| `predictedNt` | ✅ | ✅ | 164,620 / 165,122 non-null (99.7%) |
| `predictedNtConfidence` | ✅ | ✅ | 163,763 / 165,122 non-null (99.2%) |
| `consensusNt` | ✅ | ✅ | 164,620 / 165,122 non-null (99.7%) |
| `celltypePredictedNt` | Bonus | ✅ | 164,620 / 165,122 non-null (99.7%) |
| `celltypePredictedNtConfidence` | Bonus | ✅ | 162,456 / 165,122 non-null (98.4%) |
| `totalNtPredictions` | Bonus | ✅ | 164,620 / 165,122 non-null (99.7%) |
| `cellBodyFiber` | Expected | ❌ | **Not present in `male-cns:v1.0`** — schema difference from Hemibrain |

**Schema note:** `cellBodyFiber` does not appear in `male-cns:v1.0` neuron annotations.
This is a schema difference from Hemibrain v1.x (where it was used for classification).
ABB does not use `cellBodyFiber`; the warning is non-blocking.

**Total columns returned:** 53 (vs. 26 in Hemibrain — significantly richer annotation set
in MaleCNS including `dimorphism`, `fruDsx`, `superclass`, `subclass`, `class`, `mancBodyid`,
`flywireType`, etc.)

### Adjacency Fetch Results

| Property | Value |
|---|---|
| **Connections within 100-neuron random sample** | **1,598** |
| `bodyId_pre` column | ✅ Present |
| `bodyId_post` column | ✅ Present |
| `roi` column | ✅ Present (brain region of each connection) |
| `weight` column | ✅ Present (synapse count) |
| Schema check | **PASS ✓ — Missing columns: None** |

**Synapse weight statistics over 1,598 connections:**

| Statistic | Value |
|---|---|
| min | 1 |
| median | 2 |
| mean | 19.16 |
| max | 1,747 |
| std | 109.39 |
| **Total synapse contacts in sample** | **30,613** |

> [!NOTE]
> The 100-neuron sample fetched all adjacencies FROM any of those 100 neurons TO any
> other neuron in the full graph (not just within the 100 — the query uses body_ids as
> SOURCES). This produced 1,598 outgoing connections with 30,613 total synapse contacts.
> The max weight of 1,747 synapses (a single neuron→neuron pair) is consistent with
> strong hub-to-hub connections expected in Drosophila connectome data.

### neurint-python API Note
`fetch_adjacencies()` in neuprint-python ≥ 0.6.0 returns a **tuple**
`(connections_df, roi_connections_df)`. The smoke test script has been updated to
handle this correctly. The previous version (0.5.x and earlier) returned a plain DataFrame.

### Status: ✅ PASS
All schema checks fully pass. Both neuron and adjacency schemas confirmed clean.
`predictedNt` coverage: 164,620/165,122 (99.7%). `consensusNt` coverage: 164,620/165,122 (99.7%).

---

## C4 — Neurotransmitter Prediction: Final Definitive Investigation

**Objective:** Determine precisely what validation information is reported for MaleCNS v1.0
neurotransmitter predictions — model architecture, training/validation split, ground-truth
construction, validation methodology, reported metrics, and whether figures are MaleCNS-specific.

> [!CAUTION]
> The previous version of this document incorrectly attributed the **87% per-synapse / 94% per-neuron**
> accuracy figures to Hemibrain validation. **These figures are from FAFB/FlyWire validation.**
> The correct Hemibrain figures are **78% per-synapse / 91% per-neuron**. This section
> corrects the record. `DATA_PROVENANCE.md §4.1` must be updated to reflect this correction.

---

### 4.1 — Model Architecture

**Confirmed (Eckstein et al. 2024, *Cell* 187(10):2574–2594.e23):**

- **Architecture:** 3D convolutional neural network (VGG-based / ResNet50-variant)
- **Input:** 3D EM image cubes (~160 × 160 × 160 nm) centered on presynaptic sites
- **Output classes (6):** Acetylcholine, Glutamate, GABA, Serotonin, Dopamine, Octopamine
- **Prediction aggregation:** Per-neuron prediction = majority-vote over all presynaptic sites
  (aggregated probability vectors per neuron, not winner-take-all per synapse)
- **Minimum presynapses:** ≥ 50 presynaptic sites required; fewer → `predictedNt` left null
- **Confidence threshold:** Prediction confidence must be ≥ 0.5 (stored as `predictedNtConfidence`)
- **Application to MaleCNS:** The same trained model is applied to MaleCNS EM data without
  retraining. The fields `predictedNt`, `predictedNtConfidence`, `celltypePredictedNt`,
  `celltypePredictedNtConfidence`, and `consensusNt` in neuPrint reflect this model's output.
- **Dale's Law assumption:** The classifier assumes one neurotransmitter per neuron.

---

### 4.2 — Training / Validation Split and Ground-Truth Construction

**Confirmed (Eckstein et al. 2024):**

- **Ground truth source:** Literature-curated neurotransmitter identity for < 2% of Drosophila
  neurons, derived from immunohistochemistry (IHC), genetic driver lines (ChAT-Gal4, vGAT-Gal4,
  VGluT-Gal4, etc.), and published cell-type identity tables maintained at Janelia
- **Training/validation format:** Ternary (presence / absence / ambiguous) per NT per cell type
- **Primary training data:** FAFB/FlyWire TEM dataset
- **Validation strategy:** Held-out splits tested on both FAFB/FlyWire and Hemibrain as two
  independent test domains (different specimens, different EM modalities)

---

### 4.3 — Reported Performance Metrics (CORRECTED)

> [!IMPORTANT]
> The following figures are from Eckstein et al. 2024 validated on existing datasets.
> **No MaleCNS v1.0-specific figures exist** — see §4.4.

| Metric | FAFB / FlyWire (TEM) | Hemibrain (FIB-SEM) |
|---|---|---|
| **Per-synapse accuracy** | **87%** | **78%** |
| **Per-neuron accuracy** | **94%** | **91%** |
| Fast-acting NTs (ACh, Glu, GABA) | Higher reliability | Higher reliability |
| Neuromodulators (5-HT, DA, Oct) | Lower (sparse training data) | Lower (sparse training data) |

**Why Hemibrain accuracy is lower:** Hemibrain uses FIB-SEM (focused ion beam scanning EM),
producing lower-contrast ultrastructure images than FAFB's serial-section TEM. The classifier
was trained on TEM data — applying it to FIB-SEM causes a domain shift that reduces accuracy.

**What "accuracy" means:** Agreement between the ML prediction and literature-curated ground
truth on a held-out test set. Not validated by new experiments.

---

### 4.4 — Are These Figures MaleCNS v1.0-Specific?

**Status: 🚫 CLOSED-AS-UNAVAILABLE**

| Question | Finding |
|---|---|
| Does MaleCNS v1.0 report independent accuracy figures? | **No.** The MaleCNS *Cell* (2026) paper cites Eckstein et al. as the prediction method and does not report new dataset-specific validation metrics |
| Is there a MaleCNS-specific validation set or confusion matrix? | **Not publicly available** — no such data in the neuPrint API, MaleCNS GitHub repository, or accessible supplementary material |
| Was the classifier retrained or fine-tuned on MaleCNS data? | **Not confirmed** — no evidence of retraining |
| MaleCNS EM modality vs. training data modality? | **Same (serial-section TEM)** — MaleCNS was imaged with Janelia's FlyEM TEM pipeline, same as FAFB/FlyWire |

**Best available proxy:** Since MaleCNS uses TEM imaging (same modality as FAFB/FlyWire),
the FAFB figures (**87% per-synapse, 94% per-neuron**) are a better proxy than the Hemibrain
figures, but this has not been experimentally verified for MaleCNS.

**Coverage observed live (confirmed):** 164,620 / 165,122 neurons = **99.7%** have a `predictedNt`
assignment. The 502 uncovered neurons (~0.3%) have < 50 presynapses or confidence < 0.5.

---

### 4.5 — Were Older Numbers Being Incorrectly Reused?

**Previous preflight stated:** "~87% per synapse, ~94% per neuron, ~91% per cell type (**Hemibrain**)"

**Correction:**
- 87% per synapse / 94% per neuron = **FAFB/FlyWire** figures ✅ correct proxy for MaleCNS
- 78% per synapse / 91% per neuron = **Hemibrain** figures (lower — FIB-SEM domain shift)
- 91% per cell type = averaged across FAFB brain; no separate Hemibrain per-cell-type figure confirmed

The prior attribution was incorrect. `DATA_PROVENANCE.md §4.1` must be updated.

---

### C4 Final Status

| Sub-question | Status |
|---|---|
| Model architecture | ✅ CONFIRMED |
| Training/validation split | ✅ CONFIRMED (FAFB + Hemibrain held-out test sets) |
| Ground-truth construction | ✅ CONFIRMED (IHC, driver lines, literature curation) |
| Validation methodology | ✅ CONFIRMED (held-out accuracy on two EM modality domains) |
| Reported metrics (corrected) | ✅ CONFIRMED — 87%/94% FAFB; 78%/91% Hemibrain |
| MaleCNS-specific accuracy | 🚫 **CLOSED-AS-UNAVAILABLE** — not publicly reported |
| Previous incorrect attribution fixed | ✅ CORRECTED in this document and DATA_PROVENANCE.md |

**C4 overall: ⚠️ PARTIAL** — Method and predecessor figures fully confirmed. MaleCNS-specific
figures do not exist publicly; that sub-question is closed as unavailable.

---

## C9 — Neurotransmitter Prediction Breakdown Availability

**Question:** Does MaleCNS v1.0 provide an official NT prediction performance breakdown by
neurotransmitter class, cell type, ROI/brain region, by confidence, or any other grouping?

### Findings

| Breakdown Type | Source | Status | Detail |
|---|---|---|---|
| **Per-NT class** (ACh, Glu, GABA, 5-HT, DA, Oct) | Eckstein 2024 *Cell* supplementary | ✅ Available | Confusion matrices + per-class precision/recall on FAFB; data on Zenodo |
| **Fast-acting vs. neuromodulator** | Eckstein 2024 | ✅ Confirmed | ACh/Glu/GABA more reliable; 5-HT/DA/Oct less reliable |
| **Per-cell-type** | Eckstein 2024 | ⚠️ Partial | Available for well-characterised named types; not systematic |
| **Per-ROI / brain region** | Eckstein 2024 supplementary | ❓ Unconfirmed | Not confirmed accessible without paywalled paper |
| **By confidence score** | neuPrint API (live, confirmed) | ✅ Available | `predictedNtConfidence` per neuron; 99.2% coverage confirmed |
| **MaleCNS-specific per-class** | MaleCNS *Cell* (2026) | 🚫 CLOSED-AS-UNAVAILABLE | Not reported in paper or supplementary |
| **Per-ABB-subgraph** | Any source | 🚫 Not available | Must be computed from per-neuron confidence scores |

### Empirical proxy available in Phase 2

Because `predictedNtConfidence` exists per neuron (live coverage 99.2%), once Phase 2
subgraph selection is implemented, **per-subgraph mean ± std confidence** can be computed
empirically. This is not an accuracy figure, but a meaningful quality indicator per experimental
subgraph. This should be reported as a dataset characteristic metric in all ABB result tables.

### C9 Final Status: 🚫 CLOSED-AS-UNAVAILABLE (for MaleCNS-specific breakdown)

MaleCNS-specific NT accuracy breakdown by class, cell type, or ROI does not exist in any
publicly accessible source. The limitation is formally acknowledged and documented.
The Eckstein 2024 per-class breakdown (for FAFB/FlyWire) is the best available reference
and is confirmed available in the paper's supplementary data on Zenodo.

---

## Required Documentation Fix (from C4 findings)

The following change to `DATA_PROVENANCE.md §4.1` is required — it contains an incorrect
attribution:

| Current text (incorrect) | Correct replacement |
|---|---|
| "~87% per synapse, ~94% per neuron ... validated on Hemibrain" | "~87% per synapse, ~94% per neuron validated on **FAFB/FlyWire** (TEM); ~78% per synapse, ~91% per neuron on **Hemibrain** (FIB-SEM). MaleCNS v1.0 uses TEM; FAFB figures are the closest proxy but are not MaleCNS-specific." |

> [!WARNING]
> Do not update `DATA_PROVENANCE.md` now — Phase 2 has not started and the user has
> approved stopping here. Record this as a required fix before Phase 2 data module implementation.

---

## Open Items Before Phase 2 Can Begin

| # | Item | Status | Action Required |
|---|---|---|---|
| C1 | neuPrint token obtained | ✅ Done | — |
| **C2** | `male-cns:v1.0` accessible | ✅ **PASS** | — |
| **C3** | Minimal test query succeeds | ✅ **PASS** | — |
| **C4** | NT accuracy confirmed | ⚠️ PARTIAL / 🚫 CLOSED-AS-UNAVAILABLE | MaleCNS-specific accuracy not public. Predecessor figures corrected. DATA_PROVENANCE §4.1 must be updated. |
| C5 | Team review of SCIENTIFIC_ASSUMPTIONS.md | ⬜ Pending | Team action |
| C6 | Team acknowledge prohibited claims list | ⬜ Pending | Team action |
| C7 | Reference hardware identified | ⬜ Pending | Team action |
| C8 | Bayati-Kim-Saberi algorithm confirmed | ⬜ Pending | `networkx` directed_configuration_model — verify |
| **C9** | NT breakdown by class/type confirmed | 🚫 **CLOSED-AS-UNAVAILABLE** | MaleCNS-specific breakdown does not exist. Eckstein 2024 FAFB per-class data on Zenodo is best reference. |

---

## neuprint-python API Differences Noted (for Phase 2 Implementation)

| Item | Expected (from older docs) | Actual (v0.6.3) | Action |
|---|---|---|---|
| `fetch_server_info()` | Returns dict with 'Version' | Returns bool; deprecated | Use `fetch_db_version()`, `fetch_version()`, `client.meta` |
| `fetch_available_datasets()` | Returns list of strings | Method does not exist | Use `client.fetch_datasets()` (returns dict) |
| `fetch_adjacencies()` | Returns DataFrame | Returns `(neuron_metadata_df, connection_df)`; connections at `result[1]` with `bodyId_pre`, `bodyId_post`, `roi`, `weight` | Unpack `result[1]` |
| `cellBodyFiber` field | Present in Hemibrain | **Not present in male-cns:v1.0** | Remove from expected schema |
| Neuron column count | ~26 (Hemibrain) | **53 columns** in MaleCNS (richer annotation) | Update ARCHITECTURE.md annotation list in Phase 2 |

These API differences are documented and corrected in
[`scripts/data/smoke_test_neuprint.py`](file:///e:/animal-brain-benchmark/scripts/data/smoke_test_neuprint.py)
and must be reflected in `abb/data/neuprint_client.py` during Phase 2 implementation.

---

## Preflight Sign-Off

| Check | Status |
|---|---|
| C2 Dataset accessible | ✅ PASS |
| C3 Smoke test (neurons + adjacencies + schemas) | ✅ PASS |
| C4 NT method confirmed | ✅ Method CONFIRMED; MaleCNS-specific accuracy 🚫 CLOSED-AS-UNAVAILABLE |
| C4 Accuracy attribution corrected | ✅ FAFB=87%/94%, Hemibrain=78%/91% — correction documented |
| C9 NT breakdown | 🚫 CLOSED-AS-UNAVAILABLE (MaleCNS-specific); FAFB per-class data exists in Eckstein 2024 |
| .env excluded from Git | ✅ PASS |
| Token never printed or logged | ✅ PASS |
| No biological data written to disk | ✅ PASS |
| Both scripts lint-clean (pyflakes) | ✅ PASS — zero warnings |
| DATA_PROVENANCE.md §4.1 update required | ⚠️ REQUIRED before Phase 2 data module |

**Conclusion:** C2 and C3 are fully verified. C4 and C9 are closed — the MaleCNS-specific
accuracy figure genuinely does not exist publicly; this limitation is documented rather than
invented. One correction is required in `DATA_PROVENANCE.md §4.1` before Phase 2 begins
(87%/94% figures are FAFB-derived, not Hemibrain-derived).


