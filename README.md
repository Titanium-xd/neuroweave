# NeuroWeave
## Fruit Fly Brain vs AI — Animal Brain Benchmark

A benchmark for comparing connectome-topology-constrained neural architectures
derived from a real *Drosophila* nervous system against conventional AI models
on controlled computational tasks.

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data: MaleCNS v1.0 CC-BY 4.0](https://img.shields.io/badge/data-MaleCNS%20v1.0%20CC--BY%204.0-blue.svg)](LICENSE-DATA.md)
[![Security Policy](https://img.shields.io/badge/security-policy-lightgrey.svg)](SECURITY.md)
[![Live Demo](https://img.shields.io/badge/demo-live%20on%20Cloudflare-orange.svg)](https://e8fb2611.neuroweave.pages.dev/)

---

## Why this exists

Most comparisons between "bio-inspired" and conventional AI architectures use
approximate or synthetic connectivity data. NeuroWeave asks what happens when
you use the actual wiring of a real nervous system as a structural constraint
on a neural network — and test it rigorously against conventional AI under
identical conditions.

It is an engineering experiment, not a neuroscience simulation. The goal is to
find out whether real biological connectivity contains computationally useful
structure — and to be honest about when it does not.

---

## The question

> Does using the real wiring diagram of a fruit fly nervous system as a neural
> network architectural constraint produce measurable computational advantages
> over conventional AI baselines on controlled tasks?

The answer so far: **it depends on the task.**

---

## What MaleCNS is

**MaleCNS v1.0** is the first complete connectome of the adult male
*Drosophila melanogaster* central nervous system, produced by the FlyEM Project
Team at HHMI Janelia Research Campus.

- ~21,000 neurons mapped
- Tens of millions of synaptic connections
- Neurotransmitter predictions per neuron
- Publicly released under CC-BY 4.0

A connectome is a structural wiring diagram — it records which neuron connects
to which, how many synapses join them, and in which direction signals travel.
It is not a model of what the neurons compute; it is a measurement of how they
are connected.

NeuroWeave uses this connectivity data as an architectural mask on a neural
network layer: only biologically real connections are allowed to carry
learned weights. All other weights are structurally zeroed.

---

## How NeuroWeave works

```
MaleCNS v1.0 connectome
        │
        ▼
  Extract subgraph
  (150 neurons, 3,029 edges)
        │
        ▼
  Build connectivity mask
  (sparse adjacency from real data)
        │
        ▼
  Constrained neural network layer
  (weights only where biology says connections exist)
        │
        ▼
  Train on benchmark tasks
        │
        ▼
  Compare against:
  ├── Random graph (same density)
  ├── Dense MLP
  ├── LSTM
  └── Random baseline
        │
        ▼
  Evaluate on held-out test set
  (5 seeds, bootstrap 95% CI)
```

---

## System architecture

```
e:\animal-brain-benchmark\
├── abb/                     # Python research engine
│   ├── data/                # MaleCNS data loading, graph construction
│   ├── models/              # Architecture implementations (GNN, MLP, LSTM, SA-010)
│   ├── tasks/               # T-001, T-002 task definitions
│   ├── sim/                 # Simulation / evaluation loop
│   └── config/              # YAML experiment configs
├── scripts/
│   ├── tasks/               # Experiment runner scripts
│   ├── data/                # Data pipeline utilities
│   └── extract_graph_stats.py
├── artifacts/               # Real experiment outputs (JSON)
│   ├── t001_campaign/
│   └── t002_confirmation/
├── configs/                 # YAML experiment configurations
├── data/                    # Raw MaleCNS parquet data
├── docs/                    # Methodology, scope, assumptions documents
├── frontend/                # React/TanStack web interface
│   └── src/
│       ├── data/            # benchmark.ts — data layer reading artifacts/
│       ├── routes/          # Page routes (homepage, benchmark, arena, etc.)
│       └── components/
└── tests/
```

---

## Model families

### MaleCNS-derived (biological)

| ID | Description | Trainable params | Notes |
|---|---|---|---|
| A1-BIO | Connectome-constrained, trained | 6,514 | Primary model |
| A1-FROZEN | Connectome-constrained, frozen weights | 0 | Topology-only ablation |

Both models use the same 150-neuron subgraph with 3,029 directed edges.
In Phase 3, SA-010 leaky/decay temporal dynamics were added.

**SA-010 assumptions (engineering, not biological fact):**
- Decay rate: 0.05 (configurable)
- Leak rate: 0.02 (configurable)
- Selected via exploratory sensitivity analysis on T-002
- Not derived from membrane potential measurements

### Topology controls

| ID | Description |
|---|---|
| A3-ER | Erdős–Rényi random graph (same edge density) |

### Conventional AI baselines

| ID | Description | Trainable params |
|---|---|---|
| A0-RANDOM | Untrained random weights | 0 |
| A7-MLP | Multi-layer perceptron | varies |
| A8-LSTM | Long short-term memory | 266,626 |

---

## Benchmark tasks

### T-001 — Binary Pattern Discrimination

**Type:** Static supervised classification  
**D_obs:** 16  
**Memory required:** No  

A single binary pattern (sparse or dense) is presented. The model classifies
it in one forward pass. This is a baseline sanity check — it confirms that
architectures are learning, and establishes a ceiling.

### T-002 — Temporal Sequence Memory (Delayed Recall)

**Type:** Working memory / temporal classification  
**D_obs:** 16  
**Memory required:** Yes  

A binary stimulus is shown at time step 0. Several gap steps follow with no
relevant signal. At the end of the gap, the model must recall which stimulus
it originally saw. A feedforward network cannot solve this without a memory
mechanism.

---

## Key results

### T-001 — No unique MaleCNS advantage

| Architecture | Accuracy |
|---|---|
| MaleCNS-derived | ~99% |
| Random graph (ER) | ~99% |
| MLP | ~99% |
| LSTM | ~99% |
| Random baseline | ~50% |

**Finding:** All trained architectures approach ceiling accuracy. T-001 does not
discriminate between architectural families. The connectome-derived topology
provides no measurable advantage on static pattern discrimination.

This is an expected and scientifically valid negative result. The task is too
easy to reveal structural differences.

### T-002 — MaleCNS-derived + SA-010 performs strongly on delayed recall

*Confirmed results — 5-seed evaluation on held-out test set:*

| Architecture | Mean accuracy | 95% CI | Trainable params |
|---|---|---|---|
| **MaleCNS-derived + SA-010** | **94.8%** | ±6.2% | 6,514 |
| MaleCNS-derived (frozen) + SA-010 | 72.2% | ±11.4% | 0 |
| LSTM | 57.6% | ±8.9% | 266,626 |
| Random baseline | 50.0% | — | 0 |

**Finding:** On the temporal memory task, the MaleCNS-derived model with SA-010
dynamics achieves 94.8% accuracy using 41× fewer trainable parameters than the
LSTM (57.6%).

> ⚠️ **Scientific caveats — read before citing:**
>
> - The SA-010 configuration (decay 0.05 / leak 0.02) was identified through
>   exploratory sensitivity analysis, then confirmed on a separate held-out
>   test set. The exploratory sweep was not used for final evaluation.
> - Confidence intervals are wide (±6–11%). The result is statistically
>   meaningful but not yet decisive at large scale.
> - This result is task-specific. It does not generalise to other tasks,
>   other subgraphs, or other organisms.
> - The advantage may be partly attributable to SA-010 dynamics (an engineering
>   assumption), not purely to the connectome topology.
> - This is **not** evidence that the fly brain is computationally superior.
>   It is evidence that this specific computational structure performs well on
>   this specific temporal task under these specific experimental conditions.

---

## Live Arena

The **[Live Arena](https://e8fb2611.neuroweave.pages.dev/arena)** (`/arena`) is an interactive benchmark replay.

**What it is:** A browser demonstration that runs trial-by-trial replays of
the T-002 delayed-recall task. Each trial draws its outcome from the measured
benchmark accuracy probabilities:

- MaleCNS-derived + SA-010: 94.8% correct per trial
- LSTM: 57.6% correct per trial

**What it is not:** The trained PyTorch models are not running in the browser.
No model weights are loaded client-side. The demo uses a seeded deterministic
PRNG (mulberry32) to generate trial outcomes consistent with the real
benchmark statistics.

The caveat is permanently displayed on the arena page.

---

## Limitations and scientific caveats

| Limitation | Detail |
|---|---|
| Prototype scale | 150-neuron subgraph is a small sample of ~21,000 neurons |
| Single organism | Results apply to one male *Drosophila*; not generalisable to other species |
| Task scope | Only T-001 and T-002 completed; T-003/T-004 not yet run |
| SA-010 assumptions | Decay/leak parameters are engineering choices, not measured biology |
| Wide CIs | 5 seeds; 95% CI ≈ ±6–11% — results need larger-scale confirmation |
| No ablation of topology vs. dynamics | Cannot yet isolate how much of the T-002 advantage comes from the connectome wiring vs. the SA-010 dynamics |
| Single subgraph | Only one 150-neuron subgraph tested; results may vary across subgraphs |

This project **does not** claim:
- The model simulates a living fly brain
- Biological intelligence has been reproduced
- Consciousness or cognition has been modelled
- Results generalise to vertebrate or human brains
- The MaleCNS topology is universally superior to conventional architectures

---

## Reproducibility — how to run

### Prerequisites

```bash
python >= 3.10
pytorch >= 2.0
torch-geometric
neuprint-python   # for data pipeline only; raw parquet included in data/
```

### Install

```bash
git clone https://github.com/Titanium-xd/neuroweave.git
cd neuroweave
pip install -e .
```

### Run the T-002 confirmation experiment

```bash
# Set Python path
$env:PYTHONPATH='.'  # PowerShell
# or
export PYTHONPATH=.  # bash

python scripts/tasks/run_confirmation_t002.py
```

Results are written to `artifacts/t002_confirmation/`.

### Run the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:8080
```

### Run tests

```bash
pytest tests/
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Graph data | MaleCNS v1.0 parquet via neuprint-python |
| GNN | PyTorch Geometric — custom sparse linear operator |
| Temporal dynamics | Custom SA-010 leaky/decay layer (PyTorch) |
| Evaluation | 5-seed, held-out test split, bootstrap 95% CI |
| Frontend | React 19 + TanStack Router + TanStack Start |
| Data layer | JSON artifact files read by `src/data/benchmark.ts` |
| Styling | Tailwind CSS (inline), Inter Tight + Instrument Serif |
| Deployment | Vite + Nitro (Cloudflare-compatible output) |

---

## Project structure

```
abb/
  data/loader.py          MaleCNS parquet → PyG graph
  models/
    operator.py           Custom signed sparse linear operator
    malecns_gnn.py        Connectome-constrained GNN (stateful)
    sa010.py              SA-010 leaky/decay temporal dynamics
    baselines.py          MLP, LSTM, random policy
  tasks/
    t001.py               Binary pattern discrimination
    t002.py               Temporal delayed recall
  sim/evaluator.py        Multi-seed evaluation loop

scripts/
  tasks/
    run_t001_campaign.py
    run_confirmation_t002.py
    run_sensitivity_t002.py
  extract_graph_stats.py  Subgraph statistics from raw parquet

artifacts/
  campaign_1/                 T-001 multi-architecture results (10 architectures)
  campaign_2/                 T-002 stateful architecture results
  campaign_3/                 T-002 SA-010 5-seed results
  confirmation_t002/          T-002 confirmed results (5-seed, held-out)
  sensitivity_t002/           T-002 sensitivity sweep across SA-010 params

frontend/
  src/
    data/benchmark.ts     Data layer — reads artifact JSONs
    routes/               Page routes (index, benchmark, arena, etc.)
    components/site/      Nav, Footer, TaskResults, ConnectomeExplorer

docs/
  HOW_IT_WORKS.md         Plain-language explanation (9 chapters)
  RESULTS_REVIEW_CURRENT.md   Adversarial review of current results
```

---

## MaleCNS attribution

This project uses **MaleCNS v1.0** produced by the FlyEM Project Team at HHMI
Janelia Research Campus, released under CC-BY 4.0.

```
FlyEM Project Team et al.
"Sexual dimorphism in the complete connectome of the Drosophila
male central nervous system"
Cell, September 2026
https://male-cns.janelia.org
https://neuprint.janelia.org  (dataset: male-cns:v1.0)
License: CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/
```

Any use of NeuroWeave that reproduces or derives from MaleCNS data must
include the above attribution per the CC-BY 4.0 license terms.

---

## Future work

1. **Scale:** Test larger subgraphs (500, 2000 neurons) — does the T-002 advantage persist?
2. **Task diversity:** T-003 (navigation), T-004 (lesion robustness), continuous control tasks
3. **SA-010 ablation:** Isolate the contribution of connectome topology vs. leaky/decay dynamics
4. **Baseline breadth:** Add transformers, state-space models (S4, Mamba), liquid neural networks
5. **Multiple subgraphs:** Test ≥3 subgraphs per experiment to assess subgraph sensitivity
6. **Reproducibility package:** Docker image, locked dependency manifest, full seed logs

---

## License

- **NeuroWeave code:** MIT License
- **MaleCNS-derived data products:** CC-BY 4.0 (attribution required — see above)

---

## Case Study by

**Parva Trivedi**
- GitHub: [github.com/Titanium-xd](https://github.com/Titanium-xd)
- Discord: titanium.dc
- LinkedIn: [linkedin.com/in/parva-trivedi](https://www.linkedin.com/in/parva-trivedi/)

---

*NeuroWeave — Animal Brain Benchmark v1.0*
