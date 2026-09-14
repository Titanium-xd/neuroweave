"""
abb.models.bio_variants — Biological connectome-topology-constrained architectures.

These models use the MaleCNS v1.0 sparse connectivity as their graph structure.
They are NOT simulations of a biological brain. They test whether a
connectome-derived connectivity prior conveys an advantage on benchmark tasks.

Architecture variants
---------------------
All A1-* models share the same MaleCNS graph topology. They differ only in
which parameters are frozen and which are trainable:

  A1-BIO      Weights (initialized from MaleCNS synapse counts + SA-002 signs),
              biases (initialized to 0) — BOTH trainable.
              Primary biological architecture under test.

  A1-FROZEN   All graph parameters frozen. Only the readout head is trained.
              Tests whether raw MaleCNS connectivity, untrained, transfers
              to the benchmark task.

  A1-BIAS     Edge weights frozen at MaleCNS values. Only biases are trainable.
              Isolates the effect of per-node offset learning.

  A1-EDGE     Only edge weights are trainable (initialized from MaleCNS).
              Biases frozen at 0. Isolates edge weight learning.

  A1-BOTH     Both edge weights and biases trainable (alias for A1-BIO with
              explicit naming to match the control matrix).

Common architecture:
  obs → InputProjection → [ConnectomeLayer × n_layers] → ReadoutHead → act

Assumptions active: SA-001, SA-002, SA-006, SA-007, SA-008, SA-009
"""

from __future__ import annotations

from typing import Optional

import scipy.sparse as sp
import torch
import torch.nn as nn

from abb.models.base import AbstractAgent, AgentConfig
from abb.models.input_proj import InputProjection, ReadoutHead
from abb.models.operator import ConnectomeLayer, adj_from_graph


# ---------------------------------------------------------------------------
# Shared base class for all A1-* architectures
# ---------------------------------------------------------------------------


class _A1Base(AbstractAgent):
    """
    Internal base for all biological connectome-topology-constrained models.

    Subclasses set ``_trainable_weights`` and ``_trainable_bias`` class
    attributes before calling super().__init__().
    """

    _trainable_weights: bool = True   # override in subclasses
    _trainable_bias: bool = True       # override in subclasses

    def __init__(
        self,
        config: AgentConfig,
        adj_coo: sp.coo_matrix,
        input_strategy: str = "linear",
        readout_strategy: str = "linear",
        activation: str = "relu",
        output_node_idxs: Optional[list[int]] = None,
    ) -> None:
        super().__init__(config)

        n_nodes = adj_coo.shape[0]
        self._n_nodes = n_nodes
        self._adj_nnz = int(adj_coo.nnz)

        # Input projection: D_obs → N  [SA-008]
        self.input_proj = InputProjection(
            obs_dim=config.obs_dim,
            n_nodes=n_nodes,
            strategy=input_strategy,
        )

        # Stack of ConnectomeLayers  [SA-007]
        layers = []
        for _ in range(config.n_layers):
            layers.append(
                ConnectomeLayer(
                    adj_coo=adj_coo,
                    trainable_weights=self._trainable_weights,
                    trainable_bias=self._trainable_bias,
                    activation=activation,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Readout head: N → D_act  [SA-009]
        self.readout = ReadoutHead(
            n_nodes=n_nodes,
            act_dim=config.act_dim,
            strategy=readout_strategy,
            output_node_idxs=output_node_idxs,
        )

        # After construction: freeze graph layers if needed
        if not self._trainable_weights and not self._trainable_bias:
            for layer in self.layers:
                for param in layer.parameters():
                    param.requires_grad_(False)
        elif not self._trainable_weights:
            for layer in self.layers:
                if layer.operator.edge_scale is not None:
                    layer.operator.edge_scale.requires_grad_(False)
        elif not self._trainable_bias:
            for layer in self.layers:
                if isinstance(layer.bias, nn.Parameter):
                    layer.bias.requires_grad_(False)

    # ------------------------------------------------------------------
    # AbstractAgent interface
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (B, D_obs)

        Returns
        -------
        (B, D_act)
        """
        x = self.input_proj(obs)           # (B, N)
        for layer in self.layers:
            x = layer(x)                   # (B, N)
        return self.readout(x)             # (B, D_act)

    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """No-op: A1-* models are stateless (no recurrence)."""
        pass

    def flops_estimate(self) -> int:
        """
        Estimate FLOPs for one forward pass (B=1).

        = input_proj + n_layers * sparse_matmul + readout
        """
        proj_flops = self.input_proj.flops()
        layer_flops = sum(layer.flops() for layer in self.layers)
        readout_flops = self.readout.flops()
        return proj_flops + layer_flops + readout_flops

    def config_dict(self) -> dict:
        d = super().config_dict()
        d["n_nodes"] = self._n_nodes
        d["adj_nnz"] = self._adj_nnz
        d["trainable_weights"] = self._trainable_weights
        d["trainable_bias"] = self._trainable_bias
        return d


# ---------------------------------------------------------------------------
# A1-BIO — full trainable biological model
# ---------------------------------------------------------------------------


class A1Bio(_A1Base):
    """
    A1-BIO: MaleCNS topology + NT-signed weights, fully trainable.

    Both edge weights (initialized from synapse-count × NT-sign) and
    per-node biases (initialized to 0) are learnable parameters.

    This is the primary biological architecture under test.
    [SA-001, SA-002, SA-007, SA-008, SA-009]
    """

    _trainable_weights = True
    _trainable_bias = True

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 128,
        seed: int = 42,
        **kwargs,
    ) -> "A1Bio":
        adj = adj_from_graph(graph)
        cfg = AgentConfig(
            arch_id="A1-BIO",
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            seed=seed,
            assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
        )
        return cls(cfg, adj, **kwargs)


# ---------------------------------------------------------------------------
# A1-FROZEN — fully frozen graph
# ---------------------------------------------------------------------------


class A1Frozen(_A1Base):
    """
    A1-FROZEN: MaleCNS topology, all graph parameters frozen.

    Only the readout head is trained. Tests whether raw MaleCNS
    connectivity, without any gradient update, conveys a useful prior.
    [SA-001, SA-002, SA-007, SA-008, SA-009]
    """

    _trainable_weights = False
    _trainable_bias = False

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A1Frozen":
        adj = adj_from_graph(graph)
        cfg = AgentConfig(
            arch_id="A1-FROZEN",
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
        )
        return cls(cfg, adj, **kwargs)


# ---------------------------------------------------------------------------
# A1-BIAS — frozen edges, trainable biases
# ---------------------------------------------------------------------------


class A1Bias(_A1Base):
    """
    A1-BIAS: MaleCNS edge weights frozen, per-node biases trainable.

    Isolates the contribution of per-neuron offset learning while
    keeping the connectivity fixed.
    [SA-001, SA-002, SA-007, SA-008, SA-009]
    """

    _trainable_weights = False
    _trainable_bias = True

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A1Bias":
        adj = adj_from_graph(graph)
        cfg = AgentConfig(
            arch_id="A1-BIAS",
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
        )
        return cls(cfg, adj, **kwargs)


# ---------------------------------------------------------------------------
# A1-EDGE — trainable edges, frozen bias
# ---------------------------------------------------------------------------


class A1Edge(_A1Base):
    """
    A1-EDGE: Only edge weights are trainable (initialized from MaleCNS).

    Biases are frozen at 0. Isolates the contribution of edge weight
    learning while keeping biases fixed.
    [SA-001, SA-002, SA-007, SA-008, SA-009]
    """

    _trainable_weights = True
    _trainable_bias = False

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A1Edge":
        adj = adj_from_graph(graph)
        cfg = AgentConfig(
            arch_id="A1-EDGE",
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
        )
        return cls(cfg, adj, **kwargs)


# ---------------------------------------------------------------------------
# A1-BOTH — explicit alias for both trainable (same as A1-BIO)
# ---------------------------------------------------------------------------


class A1Both(_A1Base):
    """
    A1-BOTH: Both edge weights and biases trainable.

    Explicit companion to A1-FROZEN, A1-BIAS, A1-EDGE in the
    ablation matrix. Functionally identical to A1-BIO but named
    for clarity in result tables.
    [SA-001, SA-002, SA-007, SA-008, SA-009]
    """

    _trainable_weights = True
    _trainable_bias = True

    @classmethod
    def from_graph(
        cls,
        graph: "ConnectomeGraph",
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        seed: int = 42,
        **kwargs,
    ) -> "A1Both":
        adj = adj_from_graph(graph)
        cfg = AgentConfig(
            arch_id="A1-BOTH",
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
        )
        return cls(cfg, adj, **kwargs)
