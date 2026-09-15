# How It Works

*A plain-language explanation of NeuroWeave — from a real nervous system to a controlled AI benchmark.*

---

## Chapter 01 — Start with the wiring

In 2023–2024, a large international collaboration published the first complete
connectome of the adult male *Drosophila melanogaster* (fruit fly) central
nervous system — **MaleCNS v1.0**.

A connectome is a wiring diagram: it records which neuron connects to which,
how many synapses join them, and in which direction the signal travels.

> **MaleCNS v1.0**
> - ~21,000 neurons
> - Tens of millions of synaptic connections
> - Real, empirically measured connectivity — not a model or approximation

This is not a simulation of a brain. It is a structural map of one.

---

## Chapter 02 — Shrink the problem

Running 21,000 neurons as a computational graph is expensive for a first
benchmark. NeuroWeave instead extracts a **prototype subgraph** — a connected
subset of 150 neurons drawn directly from the real MaleCNS parquet data.

| Property | Value |
|---|---|
| Prototype neurons | 150 |
| Directed edges | 3,029 |
| Graph density | 0.136 |
| Source | MaleCNS v1.0 parquet (real data) |

---

## Chapter 03 — Turn wiring into computation

The connectome's connectivity is used as an **architectural constraint** on a
neural network layer. Only connections that exist in the real connectome are
allowed to carry signals. All other weights are masked to zero.

**What this is:** A mathematical graph with learnable parameters, constrained
to match real biological connectivity.

**What this is not:** A simulation of neural firing, membrane potentials, or
biological dynamics.

---

## Chapter 04 — Make the competition fair

```
              SAME TASK
                 |
      +----------+-----------+
      v                      v
 MaleCNS-derived         AI models
 (connectome-constrained) (conventional)
      |                      |
      +----------+-----------+
                 v
           SAME EVALUATION
```

### Architectures compared

| ID | Description |
|---|---|
| A0-RANDOM | Untrained random weights (baseline) |
| A1-BIO | MaleCNS-derived, trained |
| A1-FROZEN | MaleCNS-derived, frozen (ablation) |
| A3-ER | Erdos-Renyi random graph (topology control) |
| A7-MLP | Multi-layer perceptron |
| A8-LSTM | Long short-term memory network |

5 seeds minimum per architecture. Same train/test split. Same evaluation metric.

---

## Chapter 05 — The easy test wasn't enough (T-001)

**Task:** Binary pattern discrimination. One forward pass. No memory required.

**Result:** Almost every architecture approaches perfect accuracy (~99%).
T-001 does not discriminate between architectures. It confirms correctness
but cannot reveal a biological advantage.

---

## Chapter 06 — Make time matter (T-002)

**Task:** Temporal sequence memory (delayed recall).

```
STIMULUS -> WAIT -> WAIT -> WAIT -> RECALL
   *           .       .       .      ?
```

The model sees a binary stimulus, then must ignore several gap steps, then
recall which pattern it originally saw.

This requires carrying information through time — a feedforward pass cannot
solve it. Both the LSTM (via gating) and the MaleCNS-derived model (via
recurrent connectivity + SA-010 leaky-decay dynamics) attempt this.

---

## Chapter 07 — Then something changed

### Confirmed results — T-002 (5-seed, held-out test set)

| Architecture | Mean accuracy | 95% CI | Train params |
|---|---|---|---|
| **MaleCNS-derived + SA-010** | **94.8%** | +/-6.2% | 6,514 |
| MaleCNS-derived (frozen) + SA-010 | 72.2% | +/-11.4% | 0 |
| LSTM | 57.6% | +/-8.9% | 266,626 |
| Random baseline | 50.0% | — | 0 |

> **Methodological note:** SA-010 parameters (decay 0.05, leak 0.02) were
> selected via exploratory sensitivity analysis. CIs are wide. This is a
> task-specific result, not a universal biological claim.

---

## Chapter 08 — Not every test has a winner

| Task | Finding | Verdict |
|---|---|---|
| T-001 | All architectures near ceiling | No advantage |
| T-002 | MaleCNS + SA-010: 94.8%; LSTM: 57.6% | Advantage found |

The T-001 negative result is as important as the T-002 positive result.
The advantage is task-specific, not a generic property of connectome topology.

---

## Chapter 09 — Explore it

- **/benchmark** — full results, CIs, parameter counts, training times
- **/arena** — interactive T-002 benchmark replay
- **/methodology** — architecture families, SA-010 dynamics, scope

---

## Open questions

1. Does the advantage persist with larger subgraphs (>150 neurons)?
2. Which other task families reveal biological wiring advantages?
3. Are the SA-010 parameters stable across different task families?
4. How does the comparison hold against transformers and SSMs (S4, Mamba)?

---

## Technical stack

| Component | Technology |
|---|---|
| Connectome data | MaleCNS v1.0 parquet |
| Model | PyTorch Geometric + custom SA-010 leaky-decay layer |
| Evaluation | 5-seed, held-out test, bootstrap 95% CI |
| Frontend | React + TanStack Router + TanStack Start |
| Data layer | JSON artifact files from Python research engine |

---

## Credits

**Case Study by Parva Trivedi**
- GitHub: https://github.com/Titanium-xd
- Discord: titanium.dc
- LinkedIn: https://www.linkedin.com/in/parva-trivedi/

Source data: MaleCNS v1.0, *Drosophila melanogaster* connectome (public research).

Scope disclaimer: All models are computational. NeuroWeave does not recreate
a living animal, a complete nervous system, or consciousness.

---

*NeuroWeave — Animal Brain Benchmark*
