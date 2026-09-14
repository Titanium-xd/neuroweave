"""
abb.models.base — Abstract agent interface and configuration.

All ABB models implement AbstractAgent. This guarantees:
- A uniform forward/step interface for benchmark evaluation
- Exact parameter counting (no estimation)
- FLOPs estimation tied to actual sparse nnz counts
- Provenance export for reproducibility
- save/load checkpoints

Mathematical convention
-----------------------
Input:  obs ∈ R^{B × D_obs}   (batch_size × obs_dim)
Output: act ∈ R^{B × D_act}   (batch_size × act_dim)

The task defines D_obs and D_act. Subgraph size does NOT redefine
the task dimensionality. Models that use a graph internally must
project obs → graph-space and project graph-space → act explicitly.
[ASSUMPTION SA-008, SA-009]
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from abb import BENCHMARK_VERSION


# ---------------------------------------------------------------------------
# AgentConfig — serialisable configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """
    Serialisable configuration record for any ABB agent.

    Every field that affects model behaviour must appear here so that
    a saved checkpoint can be reconstructed identically.

    Parameters
    ----------
    arch_id : str
        Architecture identifier (e.g. ``"A1-BIO"``).
    obs_dim : int
        Dimensionality of the observation vector.
    act_dim : int
        Dimensionality of the action output.
    hidden_dim : int
        Internal representation dimension. For graph models, this is
        the node-feature dimension after input projection. [SA-008]
    n_layers : int
        Number of message-passing layers for graph models; number of
        hidden layers for MLPs; number of recurrent steps for RNNs.
    seed : int
        Initialization seed. Same seed → identical initial weights.
    assumption_ids : tuple[str, ...]
        SA-XXX identifiers active for this configuration.
    extra : dict
        Architecture-specific extra settings (architecture-specific).
    """

    arch_id: str
    obs_dim: int
    act_dim: int
    hidden_dim: int = 128
    n_layers: int = 2
    seed: int = 42
    assumption_ids: tuple[str, ...] = ("SA-007", "SA-008", "SA-009")
    extra: dict = field(default_factory=dict)
    benchmark_version: str = BENCHMARK_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = {
            "arch_id": self.arch_id,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "assumption_ids": list(self.assumption_ids),
            "extra": self.extra,
            "benchmark_version": self.benchmark_version,
        }
        return d

    def config_hash(self) -> str:
        """Deterministic 16-char hash of this config (for checkpoint naming)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# AbstractAgent
# ---------------------------------------------------------------------------


class AbstractAgent(nn.Module, ABC):
    """
    Common interface for all ABB model architectures.

    Subclasses implement:
    - forward()       : obs → action output (mandatory)
    - reset_state()   : clear recurrent/hidden state (mandatory)
    - param_count()   : exact trainable + total parameter count
    - flops_estimate(): estimated FLOPs for one forward pass
    - config_dict()   : full provenance dict for logging

    Subclasses may override save_checkpoint() / load_checkpoint()
    if they need custom serialisation.
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self.config = config
        # Set global seed for reproducible initialisation
        torch.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute action output from observation.

        Parameters
        ----------
        obs : torch.Tensor
            Shape (B, D_obs). Batch of observations.

        Returns
        -------
        torch.Tensor
            Shape (B, D_act). Action logits or values.
        """
        ...

    @abstractmethod
    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """
        Reset any internal recurrent or integrator state.

        Must be called at the start of each episode. Stateless models
        (MLP, GNN without temporal recurrence) implement this as a no-op.
        """
        ...

    # ------------------------------------------------------------------
    # Metrics (concrete with defaults — override for custom counting)
    # ------------------------------------------------------------------

    def param_count(self) -> dict[str, int]:
        """
        Return exact parameter counts.

        Returns
        -------
        dict with keys:
            ``trainable``  — parameters with requires_grad=True
            ``frozen``     — parameters with requires_grad=False
            ``total``      — all parameters
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total,
        }

    @abstractmethod
    def flops_estimate(self) -> int:
        """
        Estimate floating-point operations for one forward pass (batch_size=1).

        For sparse operators: 2 * nnz (multiply-accumulate).
        For dense layers: 2 * in_dim * out_dim.
        Returns a lower-bound estimate (does not count activations).
        """
        ...

    def config_dict(self) -> dict[str, Any]:
        """Return the full provenance record for this agent."""
        d = self.config.to_dict()
        d.update(self.param_count())
        d["flops_estimate"] = self.flops_estimate()
        return d

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path | str) -> None:
        """Save state dict and config to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
                "abb_checkpoint_version": "1",
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: Path | str, config: AgentConfig, map_location: str = "cpu"
    ) -> "AbstractAgent":
        """
        Restore an agent from a checkpoint.

        Parameters
        ----------
        path : Path | str
            Checkpoint file written by save_checkpoint().
        config : AgentConfig
            Config to reconstruct the model. Must match the saved config.
        map_location : str
            PyTorch map_location string.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        agent = cls(config)
        agent.load_state_dict(ckpt["state_dict"])
        return agent

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pc = self.param_count()
        return (
            f"{type(self).__name__}("
            f"arch={self.config.arch_id}, "
            f"obs={self.config.obs_dim}→act={self.config.act_dim}, "
            f"trainable_params={pc['trainable']:,}, "
            f"frozen_params={pc['frozen']:,})"
        )
