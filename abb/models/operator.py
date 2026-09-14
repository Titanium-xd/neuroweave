"""
abb.models.operator — Custom sparse signed message-passing operator.

Mathematical definition
-----------------------
Given a directed graph G = (V, E, w, s) where:
  - V : set of N nodes
  - E : set of M directed edges (pre → post)
  - w_ij ∈ R+ : synapse-count weight on edge (i → j)       [DATASET-FACT]
  - s_ij ∈ {-1, 0, +1} : NT-derived sign on edge (i → j)  [ASSUMPTION SA-002]

The signed adjacency value is:
    A_ij = s_ij · w_ij

The single-layer message-passing step is:
    h_j = Σ_{i: (i→j) ∈ E}  A_ij · x_i
        = (A^T x)_j

In matrix form (using the convention A[pre, post]):
    H = X · A     (transposed sparse matmul, shape: B × N → B × N)
    or equivalently:
    H^T = A^T · X^T

This is intentionally:
  - NO degree normalisation (unlike GCN)    [SA-007-NOTE-1]
  - NO attention                            [SA-007-NOTE-2]
  - NO learned topology selection           [SA-007-NOTE-3]
  - Explicit, fixed or trainable edge weights

SignedSparseLinear stores the adjacency as a torch.sparse COO tensor
(or COO on older PyTorch) with a separate optional trainable parameter
for per-edge weight multipliers.

A full GNN layer = SignedSparseLinear + optional bias + activation.
This is wrapped in ConnectomeLayer.

Engineering assumptions active: SA-007
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level operator: SignedSparseLinear
# ---------------------------------------------------------------------------


class SignedSparseLinear(nn.Module):
    """
    Sparse signed message-passing operator.

    Computes:  H = X @ A_eff      shape: (B, N) @ (N, N) → (B, N)

    where A_eff[i, j] = base_weight[i, j] * edge_scale[k]  (if trainable)
                       = base_weight[i, j]                  (if frozen)

    The operator stores edges in COO format (row=pre, col=post). The matmul
    is performed via torch.sparse.mm for CPU, or torch.mm on dense tensors
    for small graphs during testing.

    Parameters
    ----------
    adj_coo : scipy.sparse.coo_matrix
        The signed adjacency matrix in COO form.
        Shape (N, N), entry [i, j] = signed weight for edge i → j.
    trainable_weights : bool
        If True, a per-edge learnable multiplier (init=1.0) is added.
        If False, the adjacency is a fixed buffer (no gradient).
    dtype : torch.dtype
        Floating-point type. Default float32.

    [ASSUMPTION SA-007]
    """

    def __init__(
        self,
        adj_coo: sp.coo_matrix,
        trainable_weights: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        n = adj_coo.shape[0]
        assert adj_coo.shape == (n, n), "adjacency must be square"

        self.n_nodes: int = n

        # Normalise COO (sum duplicate entries)
        adj_coo = adj_coo.tocsr().tocoo()

        self.nnz: int = int(adj_coo.nnz)  # after dedup
        rows = torch.tensor(adj_coo.row.astype(np.int64), dtype=torch.long)
        cols = torch.tensor(adj_coo.col.astype(np.int64), dtype=torch.long)
        vals = torch.tensor(adj_coo.data.astype(np.float32), dtype=dtype)

        self.register_buffer("_edge_rows", rows)  # (nnz,)
        self.register_buffer("_edge_cols", cols)  # (nnz,)
        self.register_buffer("_base_vals", vals)  # (nnz,) — base signed weights

        if trainable_weights:
            # Per-edge learnable scale factor, initialised at 1.0
            self.edge_scale = nn.Parameter(torch.ones(self.nnz, dtype=dtype))
        else:
            self.edge_scale = None

    # ------------------------------------------------------------------
    # Build the effective sparse matrix
    # ------------------------------------------------------------------

    def _effective_sparse(self, device: torch.device) -> torch.Tensor:
        """
        Construct the sparse tensor A_eff for current parameters.

        Uses torch.sparse_coo_tensor (works on CPU and CUDA).
        """
        if self.edge_scale is not None:
            vals = self._base_vals * self.edge_scale
        else:
            vals = self._base_vals

        indices = torch.stack([self._edge_rows, self._edge_cols], dim=0)  # (2, nnz)
        # Suppress torch sparse invariant check warning — we guarantee valid COO.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse invariant checks")
            A = torch.sparse_coo_tensor(
                indices,
                vals,
                size=(self.n_nodes, self.n_nodes),
                device=device,
                dtype=vals.dtype,
            ).coalesce()
        return A

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply signed sparse message-passing.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, N) — one feature per node per batch item.

        Returns
        -------
        torch.Tensor
            Shape (B, N) — aggregated messages at each node.

        Notes
        -----
        The computation is H = X @ A (batch of row-vectors times adjacency).
        In terms of receiving messages: h_j = sum_{i→j} A[i,j] * x_i.
        This uses torch.sparse.mm which is sparse on the right side.
        We transpose: H^T = A^T @ X^T, then transpose back.
        """
        assert x.ndim == 2 and x.shape[1] == self.n_nodes, (
            f"Expected x shape (B, {self.n_nodes}), got {x.shape}"
        )
        A = self._effective_sparse(x.device)
        # H^T = A^T @ X^T   → (N, B) = (N, N) @ (N, B)
        # torch.sparse.mm requires (sparse, dense)
        x_t = x.t().contiguous()  # (N, B)
        h_t = torch.sparse.mm(A.t(), x_t)  # (N, B)
        return h_t.t()  # (B, N)

    def extra_repr(self) -> str:
        trainable = self.edge_scale is not None
        return (
            f"n_nodes={self.n_nodes}, nnz={self.nnz}, "
            f"trainable_weights={trainable}"
        )


# ---------------------------------------------------------------------------
# ConnectomeLayer — one full message-passing step
# ---------------------------------------------------------------------------


class ConnectomeLayer(nn.Module):
    """
    One connectome message-passing layer:

        h = activation( SignedSparseLinear(x) + bias )

    Parameters
    ----------
    adj_coo : scipy.sparse.coo_matrix
        Signed adjacency matrix. Shape (N, N).
    trainable_weights : bool
        Whether per-edge weights are learnable.
    trainable_bias : bool
        Whether a per-node additive bias is learnable.
    activation : str
        Non-linearity: ``"relu"``, ``"tanh"``, ``"none"``.
    dtype : torch.dtype

    [ASSUMPTION SA-007]
    """

    def __init__(
        self,
        adj_coo: sp.coo_matrix,
        trainable_weights: bool = True,
        trainable_bias: bool = True,
        activation: str = "relu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.n_nodes = adj_coo.shape[0]
        self.operator = SignedSparseLinear(adj_coo, trainable_weights=trainable_weights, dtype=dtype)

        if trainable_bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(self.n_nodes, dtype=dtype)
            )
        else:
            self.register_buffer("bias", torch.zeros(self.n_nodes, dtype=dtype))

        act_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
            "gelu": nn.GELU(),
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation '{activation}'. Choose: {list(act_map)}")
        self.activation = act_map[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N) → (B, N)"""
        h = self.operator(x)   # sparse message passing
        h = h + self.bias      # broadcast bias over batch
        return self.activation(h)

    @property
    def nnz(self) -> int:
        return self.operator.nnz

    def flops(self) -> int:
        """Multiply-accumulate count: 2 * nnz per batch element."""
        return 2 * self.nnz


# ---------------------------------------------------------------------------
# Helper: build signed COO from ConnectomeGraph
# ---------------------------------------------------------------------------


def adj_from_graph(graph: "ConnectomeGraph") -> sp.coo_matrix:
    """
    Extract the signed COO adjacency from a ``ConnectomeGraph``.

    Uses ``signed_weight`` column if present; falls back to ``weight``.

    Returns
    -------
    scipy.sparse.coo_matrix
        Shape (N, N), entries = signed synapse-count weights.
    """
    from abb.data.graph import ConnectomeGraph  # local import

    edge_df = graph.edge_df
    n = graph.n_nodes

    body_id_to_idx = {int(bid): idx for idx, bid in enumerate(graph.node_df["body_id"])}

    if "signed_weight" in edge_df.columns:
        data = edge_df["signed_weight"].astype(np.float32).to_numpy()
    elif "weight_norm" in edge_df.columns:
        data = edge_df["weight_norm"].astype(np.float32).to_numpy()
    else:
        data = edge_df["weight"].astype(np.float32).to_numpy()

    rows = edge_df["bodyId_pre"].map(body_id_to_idx).to_numpy(dtype=np.int32)
    cols = edge_df["bodyId_post"].map(body_id_to_idx).to_numpy(dtype=np.int32)

    return sp.coo_matrix((data, (rows, cols)), shape=(n, n))
