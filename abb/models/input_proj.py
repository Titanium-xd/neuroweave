"""
abb.models.input_proj — Input projection and readout head.

The task defines a fixed observation dimensionality D_obs and action
dimensionality D_act. These do NOT change when a different subgraph
is selected. A graph model must project:

    obs ∈ R^{B × D_obs}  →  node_features ∈ R^{B × N}  [SA-008]

and read out:

    node_features ∈ R^{B × N}  →  act ∈ R^{B × D_act}  [SA-009]

Projection strategies (SA-008):
  ``linear`` : obs @ W_in + b_in   (W_in ∈ R^{D_obs × N})
  ``padded`` : direct copy into first min(D_obs,N) nodes; rest = 0.

Readout strategies (SA-009):
  ``linear`` : node_features @ W_out + b_out  (W_out ∈ R^{N × D_act})
  ``mean``   : mean_pool(node_features) → R^{B × N} →linear→ R^{B × D_act}
               (mean over batch dimension of tokens; uses ALL nodes)
  ``select`` : selected_nodes @ W_out  (W_out ∈ R^{n_sel × D_act})

Note on "mean": the literature uses global mean pooling to average over
the node dimension, collapsing (B, N) → (B,) when features are 1-D.
Because our nodes carry scalar features, "mean" produces (B,) which
is then projected to act_dim via a 1→D_act linear. Use "linear" when
you want to preserve all per-node information.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Input projection  [SA-008]
# ---------------------------------------------------------------------------


class InputProjection(nn.Module):
    """
    Project observation vector into per-node initial features.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation (task-defined, fixed).
    n_nodes : int
        Number of graph nodes (subgraph-dependent).
    strategy : str
        ``"linear"`` (trainable) or ``"padded"`` (no parameters).
    bias : bool
        Include bias in linear mode.
    dtype : torch.dtype

    [ASSUMPTION SA-008]
    """

    STRATEGIES = ("linear", "padded")

    def __init__(
        self,
        obs_dim: int,
        n_nodes: int,
        strategy: str = "linear",
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.obs_dim = obs_dim
        self.n_nodes = n_nodes
        self.strategy = strategy

        if strategy == "linear":
            self.proj = nn.Linear(obs_dim, n_nodes, bias=bias, dtype=dtype)
        else:
            self.proj = None
            self._n_copy = min(obs_dim, n_nodes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (B, D_obs)

        Returns
        -------
        (B, N) — per-node initial features
        """
        if self.strategy == "linear":
            return self.proj(obs)
        # padded
        B = obs.shape[0]
        out = obs.new_zeros(B, self.n_nodes)
        out[:, : self._n_copy] = obs[:, : self._n_copy]
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops(self) -> int:
        if self.strategy == "linear":
            return 2 * self.obs_dim * self.n_nodes
        return self._n_copy


# ---------------------------------------------------------------------------
# Readout head  [SA-009]
# ---------------------------------------------------------------------------


class ReadoutHead(nn.Module):
    """
    Project graph node features to action space.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes.
    act_dim : int
        Dimensionality of the action space (task-defined, fixed).
    strategy : str
        ``"linear"``, ``"mean"``, or ``"select"``.
    output_node_idxs : list[int] | None
        Required when strategy is ``"select"``.
    bias : bool
    dtype : torch.dtype

    [ASSUMPTION SA-009]
    """

    STRATEGIES = ("linear", "mean", "select")

    def __init__(
        self,
        n_nodes: int,
        act_dim: int,
        strategy: str = "linear",
        output_node_idxs: Optional[list[int]] = None,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.n_nodes = n_nodes
        self.act_dim = act_dim
        self.strategy = strategy

        if strategy == "select":
            if output_node_idxs is None:
                raise ValueError("output_node_idxs required for strategy='select'")
            self.register_buffer(
                "_out_idxs",
                torch.tensor(output_node_idxs, dtype=torch.long),
            )
            in_features = len(output_node_idxs)
        elif strategy == "mean":
            # Global mean pool: (B, N) → (B, 1). Then 1→act_dim linear.
            in_features = 1
        else:  # "linear"
            in_features = n_nodes

        self.linear = nn.Linear(in_features, act_dim, bias=bias, dtype=dtype)
        self._in_features = in_features

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : (B, N)

        Returns
        -------
        (B, D_act)
        """
        if self.strategy == "mean":
            # Mean over node dimension → (B, 1)
            pooled = node_features.mean(dim=1, keepdim=True)
            return self.linear(pooled)
        elif self.strategy == "select":
            selected = node_features[:, self._out_idxs]  # (B, n_sel)
            return self.linear(selected)
        else:  # "linear"
            return self.linear(node_features)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops(self) -> int:
        return 2 * self._in_features * self.act_dim
