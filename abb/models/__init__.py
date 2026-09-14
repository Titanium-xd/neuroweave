"""
abb.models — Phase 2 Part 2: GNN Architecture Layer

This package implements the connectome-topology-constrained neural
architectures for the Animal Brain Benchmark.

IMPORTANT FRAMING
-----------------
The architectures in this package are NOT biological simulations.
They are graph neural networks whose sparse connectivity pattern is
derived from MaleCNS v1.0 topology. This is a computational prior,
not a claim about Drosophila cognition, consciousness, or physiology.

Architecture identifiers
------------------------
A0          Random policy (trivial baseline)
A1-BIO      MaleCNS topology, NT-signed weights, trainable
A1-FROZEN   MaleCNS topology, all parameters frozen
A1-BIAS     MaleCNS topology, only biases trainable
A1-EDGE     MaleCNS topology, only edge weights trainable
A1-BOTH     MaleCNS topology, edge weights + biases trainable
A3-ER       Erdős–Rényi random graph control
A3-CONFIG   Configuration model control (degree-matched)
A3-SBM      Stochastic block model control (cell-type blocks)
A3-DENSE    Fully connected baseline
A5-WSHUFFLE MaleCNS topology, shuffled edge weights
A5-WSIGN    MaleCNS topology, randomised edge signs
A5-ETARGET  MaleCNS topology, rewired targets
A7-MLP      Multi-layer perceptron baseline
A8-RNN      LSTM/GRU recurrent baseline

Assumptions active
------------------
SA-001  Synapse count → weight (normalization)
SA-002  NT → sign mapping
SA-006  Subgraph selection (not full connectome)
SA-007  GNN message-passing formulation
SA-008  Input projection design
SA-009  Readout head design
"""

from abb.models.base import AbstractAgent, AgentConfig
from abb.models.factory import create_agent

__all__ = [
    "AbstractAgent",
    "AgentConfig",
    "create_agent",
]
