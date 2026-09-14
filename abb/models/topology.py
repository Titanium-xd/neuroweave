"""
abb.models.topology — Topology control and edge weight control architectures.

These models provide the structural controls against which A1-BIO is compared.
They test WHAT in the MaleCNS connectivity provides any advantage:
the degree sequence, the block structure, the weight magnitudes, the signs,
or the specific wiring.

CRITICALLY: These controls are NOT biological models. They exist to isolate
the contribution of specific topological properties via ablation.

Architecture variants
---------------------
A3-ER       : Erdős–Rényi random graph.
              Same N and M as the reference subgraph; random rewiring.
              Baseline: does any sparse graph work as well as MaleCNS?

A3-CONFIG   : Configuration model.
              Preserves the exact in- and out-degree sequence of the
              reference subgraph, but rewires which neurons are connected.
              Tests whether degree distribution alone explains any advantage.

A3-SBM      : Stochastic block model.
              Uses cell-type labels as block IDs. Connection probability
              between blocks is estimated from the reference subgraph.
              Tests whether cell-type block structure alone matters.

A3-DENSE    : Fully connected (dense) linear layer.
              Same N as graph models. Parameter-count will be N² vs M
              for sparse models. Not parameter-matched; reported separately.

A5-WSHUFFLE : MaleCNS topology, edge weights randomly shuffled.
              Same weight magnitude distribution; different assignment.
              Tests whether specific synapse-count values matter beyond
              the topology.

A5-WSIGN    : MaleCNS topology, edge signs randomly reassigned (+1/-1/0).
              Tests whether the NT-derived sign pattern matters.
              [SA-002 ablation]

A5-ETARGET  : MaleCNS topology, targets rewired (sources kept, targets
              randomly permuted). Tests whether the specific post-synaptic
              targets matter beyond degree.

All topology controls use the SAME A1-BIO trainable parameter setting
(both weights and biases trainable) so differences are topology-only.

Assumptions active:
  SA-006 (subgraph), SA-007 (message passing), SA-008 (projection),
  SA-009 (readout). SA-001 and SA-002 are ablated/modified per variant.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from abb.models.base import AbstractAgent, AgentConfig
from abb.models.bio_variants import _A1Base


# ---------------------------------------------------------------------------
# Graph rewiring utilities
# ---------------------------------------------------------------------------


def _er_rewire(adj: sp.coo_matrix, seed: int = 42) -> sp.coo_matrix:
    """
    Erdős–Rényi rewire: preserve N and M; random target assignment.
    Self-loops excluded.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    m = adj.nnz

    # Generate M unique directed edges (no self-loops)
    edges = set()
    while len(edges) < m:
        batch = rng.integers(0, n, size=(m * 2, 2))
        for r, c in batch:
            if r != c:
                edges.add((int(r), int(c)))
            if len(edges) >= m:
                break

    rows, cols = zip(*edges)
    # Assign random weights drawn from original weight distribution
    orig_vals = np.abs(adj.data)
    new_vals = rng.choice(orig_vals, size=m, replace=True).astype(np.float32)
    # Random signs from {-1, +1, 0} matching original sign distribution
    orig_signs = np.sign(adj.data).astype(int)
    sign_choices = rng.choice(orig_signs, size=m, replace=True).astype(np.float32)
    data = new_vals * sign_choices

    return sp.coo_matrix((data, (list(rows), list(cols))), shape=(n, n))


def _config_rewire(adj: sp.coo_matrix, seed: int = 42) -> sp.coo_matrix:
    """
    Configuration model: preserve exact in/out degree sequence.
    Uses directed edge swaps (Markov chain) to guarantee exact degree preservation
    while randomizing the topology.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    adj = adj.copy()
    adj.sum_duplicates()
    coo = adj.tocoo()
    
    # We will swap edges in place
    rows = coo.row.copy()
    cols = coo.col.copy()
    m = len(rows)
    
    edge_set = set(zip(rows, cols))
    
    n_swaps = m * 10
    success = 0
    
    for _ in range(n_swaps * 5):  # allow failures
        if success >= n_swaps:
            break
            
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
            
        u, v = rows[i], cols[i]
        x, y = rows[j], cols[j]
        
        # New proposed edges: u->y and x->v
        # Check self-loops
        if u == y or x == v:
            continue
            
        # Check multi-edges (don't create duplicates)
        if (u, y) in edge_set or (x, v) in edge_set:
            continue
            
        # Perform swap
        edge_set.remove((u, v))
        edge_set.remove((x, y))
        edge_set.add((u, y))
        edge_set.add((x, v))
        
        rows[i], cols[i] = u, y
        rows[j], cols[j] = x, v
        success += 1

    orig_vals = np.abs(coo.data)
    orig_signs = np.sign(coo.data).astype(np.float32)
    new_vals = rng.choice(orig_vals, size=m, replace=True).astype(np.float32)
    new_signs = rng.choice(orig_signs, size=m, replace=True).astype(np.float32)
    data = new_vals * new_signs

    return sp.coo_matrix((data, (rows, cols)), shape=(n, n))


def _sbm_rewire(
    adj: sp.coo_matrix,
    block_labels: np.ndarray,
    seed: int = 42,
) -> sp.coo_matrix:
    """
    Stochastic block model: estimate block-to-block connection probabilities
    from reference adj, then sample new edges.

    Parameters
    ----------
    adj : sp.coo_matrix
        Reference adjacency.
    block_labels : np.ndarray of int, shape (N,)
        Block ID for each node (e.g. cell-type index).
    seed : int
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    blocks = np.unique(block_labels)
    n_blocks = len(blocks)
    block_idx = {b: i for i, b in enumerate(blocks)}
    node_to_block = np.array([block_idx[b] for b in block_labels])

    # Estimate P(edge | block_i → block_j) from reference
    P = np.zeros((n_blocks, n_blocks))
    N_pairs = np.zeros((n_blocks, n_blocks))

    coo = adj.tocoo()
    for r, c in zip(coo.row, coo.col):
        bi, bj = node_to_block[r], node_to_block[c]
        P[bi, bj] += 1

    for bi in range(n_blocks):
        nodes_i = np.where(node_to_block == bi)[0]
        for bj in range(n_blocks):
            nodes_j = np.where(node_to_block == bj)[0]
            N_pairs[bi, bj] = len(nodes_i) * len(nodes_j) - (
                len(nodes_i) if bi == bj else 0
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(N_pairs > 0, P / N_pairs, 0.0)

    # Sample new edges
    rows_new, cols_new = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = P[node_to_block[i], node_to_block[j]]
            if rng.random() < p:
                rows_new.append(i)
                cols_new.append(j)

    m = len(rows_new)
    if m == 0:
        return sp.coo_matrix(([], ([], [])), shape=(n, n))

    orig_vals = np.abs(adj.data)
    orig_signs = np.sign(adj.data).astype(np.float32)
    new_vals = rng.choice(orig_vals, size=m, replace=True).astype(np.float32)
    new_signs = rng.choice(orig_signs, size=m, replace=True).astype(np.float32)
    data = new_vals * new_signs

    return sp.coo_matrix((data, (rows_new, cols_new)), shape=(n, n))


def _shuffle_weights(adj: sp.coo_matrix, seed: int = 42) -> sp.coo_matrix:
    """A5-WSHUFFLE: same topology, shuffled magnitudes (signs preserved)."""
    rng = np.random.default_rng(seed)
    coo = adj.tocoo()
    magnitudes = np.abs(coo.data)
    signs = np.sign(coo.data).astype(np.float32)
    rng.shuffle(magnitudes)
    data = magnitudes * signs
    return sp.coo_matrix((data, (coo.row.copy(), coo.col.copy())), shape=adj.shape)


def _randomise_signs(adj: sp.coo_matrix, seed: int = 42) -> sp.coo_matrix:
    """A5-WSIGN: same topology and magnitudes, random signs from {-1, +1, 0}."""
    rng = np.random.default_rng(seed)
    coo = adj.tocoo()
    orig_signs = np.sign(coo.data).astype(int)
    new_signs = rng.choice(orig_signs, size=len(coo.data), replace=True).astype(np.float32)
    data = np.abs(coo.data).astype(np.float32) * new_signs
    return sp.coo_matrix((data, (coo.row.copy(), coo.col.copy())), shape=adj.shape)


def _rewire_targets(adj: sp.coo_matrix, seed: int = 42) -> sp.coo_matrix:
    """A5-ETARGET: same source nodes, targets randomly permuted per row."""
    rng = np.random.default_rng(seed)
    coo = adj.tocoo()
    n = adj.shape[0]
    cols_new = rng.permutation(coo.col)
    data = coo.data.copy()
    # Remove self-loops created by permutation
    mask = coo.row != cols_new
    return sp.coo_matrix(
        (data[mask], (coo.row[mask], cols_new[mask])), shape=(n, n)
    )


# ---------------------------------------------------------------------------
# Topology control models — all reuse _A1Base
# ---------------------------------------------------------------------------


class _TopoControl(_A1Base):
    """Shared base for topology controls: same training as A1-BIO."""

    _trainable_weights = True
    _trainable_bias = True

    @classmethod
    def _make(
        cls,
        rewired_adj: sp.coo_matrix,
        arch_id: str,
        obs_dim: int,
        act_dim: int,
        n_layers: int,
        seed: int,
        assumption_ids: tuple,
        **kwargs,
    ) -> "_TopoControl":
        cfg = AgentConfig(
            arch_id=arch_id,
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=assumption_ids,
        )
        return cls(cfg, rewired_adj, **kwargs)


class A3ER(_TopoControl):
    """A3-ER: Erdős–Rényi random topology control."""

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A3ER":
        from abb.models.operator import adj_from_graph

        ref_adj = adj_from_graph(graph)
        rewired = _er_rewire(ref_adj, seed=seed)
        return cls._make(
            rewired, "A3-ER", obs_dim, act_dim, n_layers, seed,
            ("SA-006", "SA-007", "SA-008", "SA-009"), **kwargs,
        )


class A3Config(_TopoControl):
    """A3-CONFIG: Configuration model (degree-matched) control."""

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A3Config":
        from abb.models.operator import adj_from_graph

        ref_adj = adj_from_graph(graph)
        rewired = _config_rewire(ref_adj, seed=seed)
        return cls._make(
            rewired, "A3-CONFIG", obs_dim, act_dim, n_layers, seed,
            ("SA-006", "SA-007", "SA-008", "SA-009"), **kwargs,
        )


class A3SBM(_TopoControl):
    """A3-SBM: Stochastic block model (cell-type blocks) control."""

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A3SBM":
        from abb.models.operator import adj_from_graph

        ref_adj = adj_from_graph(graph)
        # Use cell-type labels as block IDs (integer-encoded)
        node_df = graph.node_df
        type_col = "type" if "type" in node_df.columns else None
        if type_col is not None:
            types = node_df[type_col].fillna("unknown").astype("category")
            block_labels = types.cat.codes.to_numpy()
        else:
            # No type info: single block (degenerates to ER)
            block_labels = np.zeros(graph.n_nodes, dtype=int)

        rewired = _sbm_rewire(ref_adj, block_labels, seed=seed)
        return cls._make(
            rewired, "A3-SBM", obs_dim, act_dim, n_layers, seed,
            ("SA-006", "SA-007", "SA-008", "SA-009"), **kwargs,
        )


class A3Dense(AbstractAgent):
    """
    A3-DENSE: Fully connected (dense) baseline.

    Uses a standard nn.Linear for message passing (N → N).
    NOT parameter-matched to A1-BIO: has N² parameters vs M (sparse nnz).
    This difference is explicitly reported in benchmark tables.

    [SA-007-DENSE-EXCEPTION: No sparsity constraint for this control.]
    """

    def __init__(self, config: AgentConfig, n_nodes: int, **kwargs) -> None:
        super().__init__(config)
        self._n_nodes = n_nodes

        self.input_proj = torch.nn.Linear(config.obs_dim, n_nodes)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for _ in range(config.n_layers)]
        )
        self.act_fn = torch.nn.ReLU()
        self.readout = torch.nn.Linear(n_nodes, config.act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)
        for layer in self.layers:
            x = self.act_fn(layer(x))
        return self.readout(x)

    def reset_state(self, batch_size: int = 1, device=None) -> None:
        pass  # stateless

    def flops_estimate(self) -> int:
        n = self._n_nodes
        proj = 2 * self.config.obs_dim * n
        layers = self.config.n_layers * 2 * n * n
        readout = 2 * n * self.config.act_dim
        return proj + layers + readout

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
    ) -> "A3Dense":
        cfg = AgentConfig(
            arch_id="A3-DENSE",
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=("SA-007", "SA-008", "SA-009"),
        )
        return cls(cfg, n_nodes=graph.n_nodes)


# ---------------------------------------------------------------------------
# Edge weight / sign ablation controls (A5-*)
# ---------------------------------------------------------------------------


class A5WShuffle(_TopoControl):
    """A5-WSHUFFLE: MaleCNS topology, shuffled edge weight magnitudes."""

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A5WShuffle":
        from abb.models.operator import adj_from_graph

        ref_adj = adj_from_graph(graph)
        ablated = _shuffle_weights(ref_adj, seed=seed)
        return cls._make(
            ablated, "A5-WSHUFFLE", obs_dim, act_dim, n_layers, seed,
            ("SA-006", "SA-007", "SA-008", "SA-009"), **kwargs,
        )


class A5WSign(_TopoControl):
    """A5-WSIGN: MaleCNS topology, randomised edge signs. [SA-002 ablation]"""

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A5WSign":
        from abb.models.operator import adj_from_graph

        ref_adj = adj_from_graph(graph)
        ablated = _randomise_signs(ref_adj, seed=seed)
        return cls._make(
            ablated, "A5-WSIGN", obs_dim, act_dim, n_layers, seed,
            ("SA-006", "SA-007", "SA-008", "SA-009"), **kwargs,
        )


class A5ETarget(_TopoControl):
    """A5-ETARGET: MaleCNS source nodes kept, targets randomly permuted."""

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A5ETarget":
        from abb.models.operator import adj_from_graph

        ref_adj = adj_from_graph(graph)
        ablated = _rewire_targets(ref_adj, seed=seed)
        return cls._make(
            ablated, "A5-ETARGET", obs_dim, act_dim, n_layers, seed,
            ("SA-006", "SA-007", "SA-008", "SA-009"), **kwargs,
        )
