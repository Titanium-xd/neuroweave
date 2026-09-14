# DATA_PROVENANCE.md
# Animal Brain Benchmark — Data Provenance and Attribution

> **Status:** Phase 1 — Revised Specification (ABB-0.1-rev1)
> **Last Updated:** 2026-09-14
> **Benchmark Version:** ABB-0.1-rev1

---

## 1. Primary Dataset: MaleCNS v1.0

### 1.1 Identity and Release

| Field | Value |
|---|---|
| **Dataset Name** | MaleCNS (Male Central Nervous System Connectome) |
| **Version** | v1.0 |
| **Release Date** | June 8, 2026 (dataset); September 3, 2026 (*Cell* paper) |
| **Producing Institution** | HHMI Janelia Research Campus — FlyEM Project Team |
| **Collaborating Institutions** | University of Cambridge; MRC Laboratory of Molecular Biology; Google Research |
| **Official Project URL** | https://male-cns.janelia.org |
| **Primary Access Platform** | neuPrint (https://neuprint.janelia.org) |
| **Dataset Identifier (API)** | `male-cns:v1.0` |

### 1.2 Primary Publication (Required Citation)

> **Title:** "Sexual dimorphism in the complete connectome of the Drosophila male
> central nervous system"
> **Journal:** *Cell* (September 3, 2026)
> **Authors:** FlyEM Project Team et al. (full author list in paper)
> **DOI:** To be confirmed from the published paper; recorded in `CITATIONS.bib` before
> any public release of ABB results.

All ABB results derived from MaleCNS data MUST cite this paper and the dataset.

### 1.3 License

| Field | Value |
|---|---|
| **License** | Creative Commons Attribution 4.0 International (CC-BY 4.0) |
| **License URL** | https://creativecommons.org/licenses/by/4.0/ |
| **Attribution Required** | Yes — cite primary publication; acknowledge HHMI Janelia FlyEM |
| **Commercial Use** | Permitted under CC-BY 4.0 terms |
| **Redistribution of raw data** | Permitted with attribution; verify neuPrint current ToS before bulk redistribution |

> **Engineering Note (ABB-PROV-001):** ABB does not redistribute raw connectome data files.
> Data is accessed via the official neuPrint API and cached locally as derived sparse
> adjacency representations. All cached artefacts derived from MaleCNS data inherit the
> CC-BY 4.0 obligation and carry a machine-readable provenance header.

---

## 2. Related and Predecessor Datasets

### 2.1 MANC — Male Adult Nerve Cord

| Field | Value |
|---|---|
| **Scope** | VNC only (no brain) |
| **Primary Paper** | Takemura et al., *eLife* (2024) |
| **Annotations Paper** | Marin et al., *eLife* (2024) |
| **neuPrint identifier** | `manc:v1.0` |
| **Relationship** | Superseded by MaleCNS v1.0 for whole-CNS work; useful for VNC-specific cross-validation |

### 2.2 FlyWire — Adult Female Brain

| Field | Value |
|---|---|
| **Scope** | Brain only; ~140,000 neurons |
| **License** | CC-BY 4.0 |
| **Relationship** | Female reference; planned Phase 4 cross-connectome comparison |

### 2.3 Hemibrain v1.2.1

| Field | Value |
|---|---|
| **neuPrint identifier** | `hemibrain:v1.2.1` |
| **Relationship** | Older reference; API patterns used in documentation examples. **Neurotransmitter prediction accuracy figures differ by EM modality:** FAFB/FlyWire (serial-section TEM, same modality as MaleCNS) = 87% per synapse / 94% per neuron; Hemibrain (FIB-SEM, lower contrast) = 78% per synapse / 91% per neuron. These figures may not directly apply to MaleCNS v1.0 subgraphs (see §4.1). |

---

## 3. Access Methods

### 3.1 neuPrint Python API (Primary Access Method)

**Library:** `neuprint-python`
**Install:** `pip install neuprint-python`
**Authentication:** Personal API token obtained from https://neuprint.janelia.org → Account → API Token

```python
# CANONICAL INITIALIZATION — tokens from environment variable only
import os
from neuprint import Client

token = os.environ.get('NEUPRINT_TOKEN')
if token is None:
    raise RuntimeError(
        "NEUPRINT_TOKEN environment variable is not set.\n"
        "Obtain a token at https://neuprint.janelia.org (Account > API Token),\n"
        "then set it with:\n"
        "  export NEUPRINT_TOKEN=<your_token>   # Linux/macOS\n"
        "  $env:NEUPRINT_TOKEN='<your_token>'   # Windows PowerShell"
    )

client = Client(
    'https://neuprint.janelia.org',
    dataset='male-cns:v1.0',
    token=token
)
```

> **Security Rule (ABB-PROV-002):** API tokens MUST NOT appear in source code, committed
> configuration files, or log outputs. The ONLY permitted method is the `NEUPRINT_TOKEN`
> environment variable. Do not use a secrets manager that may log the token value.

**Key API functions:**

| Function | Purpose |
|---|---|
| `fetch_neurons(NeuronCriteria)` | Neuron metadata, type annotations, per-ROI synapse counts |
| `fetch_adjacencies(sources, targets)` | Directed weighted edge list; `weight` = synapse count |
| `connection_table_to_matrix()` | Convert edge list to dense matrix (small subgraphs only) |
| `fetch_custom(cypher_query)` | Raw Neo4j Cypher queries for custom data extraction |
| `fetch_synapses(NeuronCriteria)` | Individual synapse locations |

**`fetch_adjacencies()` output schema:**

| Column | Type | Description |
|---|---|---|
| `bodyId_pre` | int64 | Presynaptic neuron unique ID [DATASET-FACT] |
| `bodyId_post` | int64 | Postsynaptic neuron unique ID [DATASET-FACT] |
| `weight` | int | Synapse contact count; our connection strength proxy [DATASET-FACT + ASSUMPTION SA-001] |
| `roi` | str | Brain region of interest |

> **Assumption Note (ABB-PROV-003 — SA-001):** `weight` (synapse count) is used as the
> initial proxy for connection strength. This is an engineering assumption. Synapse count
> correlates with synaptic area, which correlates with postsynaptic current amplitude
> (Rees et al., 2017), but the precise functional weight of every synapse is NOT
> experimentally measured in this dataset.

### 3.2 navis — Morphology and Skeleton Access (Future Phases)

**Library:** `navis`
**Purpose:** Neuron skeleton trees (SWC format), meshes, spatial coordinates.
ABB Phase 1–2 does not require skeleton data.

### 3.3 natverse / malecns R Package (Reference Authority)

**Package:** `malecns` (R; natverse ecosystem)
**GitHub:** https://github.com/flyconnectome/malecns

ABB is Python-based. If annotation discrepancies arise between Python and R API outputs,
the `malecns` R package output is treated as the authoritative reference, as it is
maintained by the dataset producers.

---

## 4. Neuron Annotation Schema

| Field | neuPrint Property | Status | Notes |
|---|---|---|---|
| **bodyId** | `bodyId` | [DATASET-FACT] | Reconstruction-assigned integer; not a biological property |
| **Cell type** | `type` | [DATASET-FACT] (with judgment) | Expert-assigned; ~11,710 types; some assignments involve judgment |
| **Cell instance** | `instance` | [DATASET-FACT] (with judgment) | More specific than type; includes laterality (L/R) |
| **Neurotransmitter** | `predictedNt` | [COMPUTATIONAL MODEL] | ML-predicted from EM features; NOT physiologically measured in this specimen |
| **Synapse count** | edge `weight` | [DATASET-FACT] | Subject to reconstruction error |
| **Brain region (ROI)** | ROI fields | [DATASET-FACT] | Expert atlas with judgment-based boundaries |
| **Sex** | Fixed: male | [BIO-FACT] | Single adult male specimen |
| **Developmental stage** | Fixed: adult | [BIO-FACT] | |

### 4.1 Neurotransmitter Accuracy — Precise Statement

> [!IMPORTANT]
> The neurotransmitter predictor used in MaleCNS v1.0 is the Eckstein et al. (2024, *Cell*
> 187(10):2574–2594.e23) 3D CNN classifier applied to EM synaptic ultrastructure images.
>
> **Corrected accuracy attribution (ABB-0.1-rev1, finalized 2026-09-14):**
>
> | Dataset | EM Modality | Per-Synapse Accuracy | Per-Neuron Accuracy |
> |---|---|---|---|
> | FAFB / FlyWire | Serial-section TEM | **87%** | **94%** |
> | Hemibrain v1.x | FIB-SEM (lower contrast) | **78%** | **91%** |
>
> **MaleCNS v1.0** uses serial-section TEM (same modality as FAFB/FlyWire). Therefore the
> FAFB figures (87% per synapse, 94% per neuron) are the closest available proxy. However,
> these figures were validated on FAFB/Hemibrain data; **no independent MaleCNS-specific
> validation has been publicly reported.** The MaleCNS *Cell* (2026) paper cites Eckstein
> et al. without reporting new dataset-specific accuracy metrics.
>
> **Classification rules:**
> - Prediction requires ≥ 50 presynaptic contacts; otherwise `predictedNt` is null
> - Prediction accepted only if confidence (`predictedNtConfidence`) ≥ 0.5
> - Coverage confirmed live: 164,620 / 165,122 neurons = 99.7% have a `predictedNt` assignment
> - Neuromodulators (5-HT, DA, Oct) are less reliably predicted than fast-acting NTs (ACh, Glu, GABA)
>
> Until a MaleCNS-specific validation study is published, ABB documents state:
> *"NT predictions are computational estimates. The closest published accuracy proxy is 87%
> per synapse / 94% per neuron (FAFB/FlyWire, same TEM modality); Hemibrain accuracy was 78%/91%
> (FIB-SEM, different modality). MaleCNS-specific accuracy has not been independently validated
> and is formally CLOSED-AS-UNAVAILABLE from public sources."*
>
> Sign assignments derived from NT predictions are labeled [ASSUMPTION SA-002] throughout.

### 4.2 Neurotransmitter-to-Sign Mapping (SA-002)

| Predicted NT | Sign Applied | Justification | Confidence |
|---|---|---|---|
| Acetylcholine (ACh) | +1 | Dominant excitatory NT in insect CNS | High [BIO-FACT] |
| GABA | −1 | Dominant inhibitory NT | High [BIO-FACT] |
| Glutamate | −1 | Predominantly inhibitory in *Drosophila* CNS | Medium (context-dependent) |
| Dopamine | 0 or learnable | Neuromodulator; receptor-subtype-dependent | Low [ASSUMPTION] |
| Serotonin | 0 or learnable | Neuromodulator | Low [ASSUMPTION] |
| Octopamine | 0 or learnable | Insect noradrenaline analogue | Low [ASSUMPTION] |
| Unknown | 0 or learnable | Insufficient information | Very Low [ASSUMPTION] |

Full discussion: SCIENTIFIC_ASSUMPTIONS.md SA-002.

---

## 5. Data Integrity and Versioning

### 5.1 Reconstruction Errors

Despite extensive proofreading, MaleCNS v1.0 contains residual errors:
- **Merge errors:** Two neurons incorrectly joined
- **Split errors:** One neuron incorrectly fragmented into multiple bodies
- **Synapse detection errors:** False positive or negative synapse identifications

> **ABB Policy (ABB-PROV-006):** MaleCNS v1.0 is used as-is. Reconstruction errors are
> NOT corrected by ABB.
>
> **Important:** Reconstruction errors affect only A1 and A2 (biological topology models),
> not synthetic control graphs A3/A4/A5. This means any observed DISADVANTAGE of A1/A2
> relative to controls could partly reflect reconstruction noise in the input topology rather
> than a property of biological topology itself. This is a fundamental confound that must be
> stated in all result publications. It cannot be resolved without error-corrected data.

### 5.2 Dataset Version Pinning

All data downloads record the exact dataset version. The version string `male-cns:v1.0`
is embedded in every experiment manifest. If a corrected or updated version is released,
all affected experiments are re-run and the delta is documented.

### 5.3 Local Cache Policy

- Raw adjacency edge lists are cached in `data/raw/malecns_v1_0/` as compressed Parquet files
- Each cache file has a `.provenance.json` sidecar (download date, API version, query
  parameters, sha256 checksum)
- Cached data is NOT committed to Git (listed in `.gitignore`)
- `scripts/data/download_malecns.py` reproduces the full cache from scratch
- If the cache is absent, experiments that require connectome data refuse to start with
  a clear error message directing the user to run the download script first

---

## 6. Scale Constraints

| Property | Value | ABB Implication |
|---|---|---|
| Total neurons | ~166,700 | Full-graph simulation infeasible on single GPU |
| Total synapses | ~125,000,000 | Full adjacency ≈ 220 GB float32; sparse representations mandatory |
| Cell types | ~11,710 | Enables type-aware subgraph construction and lesion strategies |

> **ABB-PROV-007 (SA-006):** Phase 1–2 experiments use subgraphs of ~5,000 neurons. Results
> are specific to the selected subgraph. Cross-subgraph variance is reported over ≥ 3
> independent subgraphs per experiment (BENCHMARK_DESIGN.md §4.4).

---

## 7. Responsible Use Statement

- MaleCNS v1.0 is from a single adult male *Drosophila melanogaster* specimen — not a
  population average.
- ABB makes no claims about consciousness, cognition, subjective experience, or any
  non-structural property of the biological organism.
- ABB does not claim the computational model is a simulation of a living fly.
- The A1-BIO condition is described as the "most data-direct, least-trained connectome-
  derived condition," not as a biological simulation.
- Results must not be extrapolated to vertebrate or human neuroscience without strong
  independent justification.
- All modeling assumptions are disclosed and labeled. Results that are artifacts of
  undisclosed assumptions will not be published.

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-09-14 | ABB-0.1 | Initial document |
| 2026-09-14 | ABB-0.1-rev1 | HIGH-02: corrected NT accuracy claim to specify Hemibrain validation conditions; added explicit statement that MaleCNS-specific validation must be confirmed from paper methods; MED-05: corrected §5.1 reconstruction error note to state that errors affect only biological models, not synthetic controls; MED-06: replaced neuPrint token code example with explicit RuntimeError and corrected PowerShell syntax |
| 2026-09-14 | ABB-0.1-rev1 (C4-final) | §4.1 accuracy figures corrected: 87%/94% = FAFB/FlyWire (TEM) validation; 78%/91% = Hemibrain (FIB-SEM) validation. Previous Hemibrain attribution was incorrect. MaleCNS-specific accuracy formally CLOSED-AS-UNAVAILABLE. §2.3 Hemibrain entry updated with corrected EM-modality note. |
