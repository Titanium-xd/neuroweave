"""
abb.models.factory — Model factory and registry.

Provides a single entry point for creating any ABB model by name:

    agent = create_agent("A1-BIO", adj_coo=adj, obs_dim=8, act_dim=4)

Also provides `list_architectures()` for discovery.
"""

from __future__ import annotations

from typing import Any, Optional

import scipy.sparse as sp

from abb.models.base import AbstractAgent, AgentConfig
from abb.models.baselines import A0Random, A7MLP, A8RNN
from abb.models.bio_variants import A1Bias, A1Bio, A1Both, A1Edge, A1Frozen
from abb.models.topology import (
    A3Config,
    A3Dense,
    A3ER,
    A3SBM,
    A5ETarget,
    A5WShuffle,
    A5WSign,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCH_IDS = [
    "A0",
    "A1-BIO", "A1-FROZEN", "A1-BIAS", "A1-EDGE", "A1-BOTH",
    "A3-ER", "A3-CONFIG", "A3-SBM", "A3-DENSE",
    "A5-WSHUFFLE", "A5-WSIGN", "A5-ETARGET",
    "A7-MLP",
    "A8-LSTM", "A8-GRU",
]


def list_architectures() -> list[str]:
    """Return all registered architecture identifiers."""
    return list(ARCH_IDS)


def create_agent(
    arch_id: str,
    obs_dim: int,
    act_dim: int,
    *,
    # Graph models
    adj_coo: Optional[sp.coo_matrix] = None,
    graph: Optional[Any] = None,
    # Common
    hidden_dim: int = 128,
    n_layers: int = 2,
    seed: int = 42,
    # RNN-specific
    cell_type: str = "lstm",
    # Pass-through kwargs to model constructors
    **kwargs,
) -> AbstractAgent:
    """
    Create an ABB agent by architecture ID.

    Parameters
    ----------
    arch_id : str
        Architecture identifier. See list_architectures().
    obs_dim : int
        Observation dimensionality (task-defined).
    act_dim : int
        Action dimensionality (task-defined).
    adj_coo : sp.coo_matrix | None
        Signed adjacency for graph models. Required for A1-* and A3-*/A5-*.
    graph : ConnectomeGraph | None
        Alternative to adj_coo: pass a ConnectomeGraph and adj is extracted.
    hidden_dim : int
        Hidden dimension for MLP/RNN baselines.
    n_layers : int
        Number of graph layers / MLP hidden layers / RNN layers.
    seed : int
        Initialization seed.
    cell_type : str
        ``"lstm"`` or ``"gru"`` for A8-*.
    **kwargs
        Passed through to model constructors (e.g. input_strategy, activation).

    Returns
    -------
    AbstractAgent
    """
    if arch_id not in ARCH_IDS:
        raise ValueError(
            f"Unknown arch_id {arch_id!r}. Available: {ARCH_IDS}"
        )

    # Resolve graph to adj_coo if needed
    if graph is not None and adj_coo is None:
        from abb.models.operator import adj_from_graph
        adj_coo = adj_from_graph(graph)

    # Helper to assert adj is available for graph models
    def _need_adj():
        if adj_coo is None and graph is None:
            raise ValueError(
                f"arch_id={arch_id!r} requires adj_coo or graph parameter."
            )

    # ── Baselines ──────────────────────────────────────────────────────
    if arch_id == "A0":
        return A0Random.build(obs_dim=obs_dim, act_dim=act_dim, seed=seed)

    if arch_id == "A7-MLP":
        return A7MLP.build(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, seed=seed,
        )

    if arch_id in ("A8-LSTM", "A8-GRU"):
        _ct = "lstm" if arch_id == "A8-LSTM" else "gru"
        return A8RNN.build(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            cell_type=_ct, seed=seed,
        )

    # ── Biological variants ────────────────────────────────────────────
    _need_adj()
    common = dict(obs_dim=obs_dim, act_dim=act_dim, n_layers=n_layers, seed=seed)

    _graph_or_adj = graph  # prefer graph for from_graph() methods

    def _adj_wrap():
        """Return a minimal stand-in object for from_graph() that has adj."""
        # If only adj_coo is provided, build a dummy graph object
        if _graph_or_adj is not None:
            return _graph_or_adj
        # Build a dummy ConnectomeGraph-like object
        import pandas as pd
        from abb.data.graph import ConnectomeGraph, build_graph
        n = adj_coo.shape[0]
        dummy_neurons = pd.DataFrame({
            "bodyId": list(range(n)),
            "type": [None] * n,
            "predictedNt": [None] * n,
        })
        dummy_edges = pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
        return build_graph(dummy_neurons, dummy_edges)

    if arch_id == "A1-BIO":
        cfg = AgentConfig(arch_id="A1-BIO", obs_dim=obs_dim, act_dim=act_dim,
                          n_layers=n_layers, seed=seed,
                          assumption_ids=("SA-001","SA-002","SA-006","SA-007","SA-008","SA-009"))
        return A1Bio(cfg, adj_coo, **kwargs)

    if arch_id == "A1-FROZEN":
        cfg = AgentConfig(arch_id="A1-FROZEN", obs_dim=obs_dim, act_dim=act_dim,
                          n_layers=n_layers, seed=seed,
                          assumption_ids=("SA-001","SA-002","SA-006","SA-007","SA-008","SA-009"))
        return A1Frozen(cfg, adj_coo, **kwargs)

    if arch_id == "A1-BIAS":
        cfg = AgentConfig(arch_id="A1-BIAS", obs_dim=obs_dim, act_dim=act_dim,
                          n_layers=n_layers, seed=seed,
                          assumption_ids=("SA-001","SA-002","SA-006","SA-007","SA-008","SA-009"))
        return A1Bias(cfg, adj_coo, **kwargs)

    if arch_id == "A1-EDGE":
        cfg = AgentConfig(arch_id="A1-EDGE", obs_dim=obs_dim, act_dim=act_dim,
                          n_layers=n_layers, seed=seed,
                          assumption_ids=("SA-001","SA-002","SA-006","SA-007","SA-008","SA-009"))
        return A1Edge(cfg, adj_coo, **kwargs)

    if arch_id == "A1-BOTH":
        cfg = AgentConfig(arch_id="A1-BOTH", obs_dim=obs_dim, act_dim=act_dim,
                          n_layers=n_layers, seed=seed,
                          assumption_ids=("SA-001","SA-002","SA-006","SA-007","SA-008","SA-009"))
        return A1Both(cfg, adj_coo, **kwargs)

    # ── Topology controls ──────────────────────────────────────────────
    g = _adj_wrap()

    if arch_id == "A3-ER":
        return A3ER.from_graph(g, **common, **kwargs)
    if arch_id == "A3-CONFIG":
        return A3Config.from_graph(g, **common, **kwargs)
    if arch_id == "A3-SBM":
        return A3SBM.from_graph(g, **common, **kwargs)
    if arch_id == "A3-DENSE":
        return A3Dense.from_graph(g, **common)

    if arch_id == "A5-WSHUFFLE":
        return A5WShuffle.from_graph(g, **common, **kwargs)
    if arch_id == "A5-WSIGN":
        return A5WSign.from_graph(g, **common, **kwargs)
    if arch_id == "A5-ETARGET":
        return A5ETarget.from_graph(g, **common, **kwargs)

    raise RuntimeError(f"Unhandled arch_id: {arch_id}")  # should never reach here
