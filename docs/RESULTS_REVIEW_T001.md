# Adversarial Review: T-001 Binary Pattern Discrimination (Campaign 1)

**Date**: 2026-09-15
**Task**: T-001 Binary Pattern Discrimination
**Scope**: 10 architectures, 5 seeds each, 50 total runs.

## 1. Is T-001 too easy to distinguish architectures?
**Yes.** T-001 is fundamentally a static global density task. The input is a 64-dimensional binary vector where class 0 has ~25% active bits and class 1 has ~75% active bits. Any architecture that can linearly aggregate these bits (such as a simple linear classifier) can solve it. This is proven by the Dense baselines (A7-MLP, A8-LSTM) achieving 100% test accuracy rapidly.

## 2. Does the result actually test topology, or mostly global input statistics?
**Mostly global input statistics.** The task tests whether the network can count "ON" bits across the entire input space. It does not require specific inputs to interact with other specific inputs (e.g., XOR logic), nor does it require spatial or temporal routing. The network simply needs information to not be destroyed.

## 3. Are A1-BIO, A1-FROZEN, A1-BIAS, and A1-EDGE being interpreted correctly?
**Yes.** The variations correctly demonstrate ablation dynamics. Notably, `A1-FROZEN` (where the biological graph weights and biases are untrained) performs at ~97.8% — statistically indistinguishable from fully trained `A1-BIO` (97.4%). This proves that a random sparse projection (via the MaleCNS biological topology) preserves enough input variance for the trainable linear readout head to easily separate the two classes.

## 4. Are the parameter counts and frozen/trainable counts consistent?
**Yes, mathematically exact.** 
The A1-BIO parameter budget is $13,714$.
- `A1-FROZEN`: $10,052$ trainable (Input + Readout) + $3,662$ frozen ($1,681$ edges $\times 2$ layers $+ 150$ biases $\times 2$ layers). Total: $13,714$.
- `A1-BIAS`: $10,352$ trainable + $3,362$ frozen edges. Total: $13,714$.
- `A1-EDGE`: $13,414$ trainable + $300$ frozen biases. Total: $13,714$.
- `A3-CONFIG`: Exactly matches A1-BIO's $1,681$ unique coalesced edges.

## 5. Are the random-graph controls valid?
**Yes.** `A3-ER` (random connectivity) and `A3-CONFIG` (degree-sequence preserved connectivity) both match or marginally exceed `A1-BIO` performance ($99.4\%$ and $98.2\%$ respectively vs $97.4\%$). This confirms that the success of A1-BIO is a generic property of sparse graphs of this size, not a unique property of the MaleCNS connectome.

## 6. Are there any data leakage or training/evaluation problems?
**No.** The dataset generation is deterministic (fixed seed), train/dev/test splits are strictly respected, and recurrent baselines (`A8-LSTM`, `A8-GRU`) were corrected to reset hidden states between independent static batches, eliminating potential sequence leakage.

## 7. Are the reported confidence intervals and multi-seed results being used correctly?
**Yes.** The 5-seed 95% bootstrap CIs ($\pm 1.1\%$ to $2.4\%$) correctly reveal that minor performance gaps between the sparse architectures are statistical noise. Relying on a single seed might have incorrectly implied that one sparse topology was definitively superior to another.

## 8. What conclusions are justified?
- T-001 is a trivial benchmark that is easily solved by dense networks.
- For global density estimation, forced sparsity acts as a bottleneck, reducing accuracy from 100% to ~98%.
- On this specific task, the biological MaleCNS topology provides no measurable computational advantage over random topologies of identical size/sparsity.

## 9. What conclusions are NOT justified?
- **"MaleCNS is an inferior topology."** We cannot conclude this. T-001 does not challenge the network's routing capacity.
- **"Biological brains don't learn."** A1-FROZEN performing well does not mean learning is unnecessary; it just means this task can be solved via random projection mapping.

## 10. What should the next benchmark task test that T-001 does not?
The next benchmark must penalize naive global aggregation and reward **structured routing or temporal integration**. 
Examples:
- **Spatial/Hierarchical Routing:** Tasks where specific input subsets must be routed to specific output subsets without crosstalk.
- **Temporal Memory:** Sequential tasks requiring the network to hold state over time (e.g., navigating a maze, remembering past stimuli), testing if the MaleCNS recurrence provides a memory advantage.
