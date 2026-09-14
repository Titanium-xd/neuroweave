"""
abb.models.baselines — Conventional baseline architectures.

These models do NOT use graph connectivity. They serve as lower and
upper bounds for performance comparison.

A0  Random Policy   : outputs random actions. Lower bound baseline.
                      Demonstrates chance-level performance with zero learning.

A7  MLP             : Multi-layer perceptron. Standard dense feedforward.
                      Parameter count is independently reported and may differ
                      from graph models.

A8  RNN             : LSTM or GRU recurrent network. Tests whether temporal
                      memory helps on sequential tasks.
                      Only recurrent baseline; all other models are stateless.

Engineering assumptions: SA-008 (projection), SA-009 (readout).
No SA-007 (no graph operator in baselines).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from abb.models.base import AbstractAgent, AgentConfig


# ---------------------------------------------------------------------------
# A0 — Random Policy
# ---------------------------------------------------------------------------


class A0Random(AbstractAgent):
    """
    A0: Random policy baseline.

    Returns uniformly random actions regardless of input. Zero learnable
    parameters. Provides the chance-level performance floor.

    Not trainable. reset_state() is a no-op.
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        # No parameters
        self._act_dim = config.act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns uniform random actions sampled at each call."""
        B = obs.shape[0]
        return torch.rand(B, self._act_dim, device=obs.device, dtype=obs.dtype)

    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        pass  # stateless

    def flops_estimate(self) -> int:
        return self._act_dim  # one random draw per action dim

    @classmethod
    def build(cls, obs_dim: int, act_dim: int, seed: int = 42) -> "A0Random":
        cfg = AgentConfig(
            arch_id="A0",
            obs_dim=obs_dim,
            act_dim=act_dim,
            seed=seed,
            assumption_ids=(),
        )
        return cls(cfg)


# ---------------------------------------------------------------------------
# A7 — MLP
# ---------------------------------------------------------------------------


class A7MLP(AbstractAgent):
    """
    A7: Multi-layer perceptron.

    Architecture:
        obs → Linear(obs_dim, hidden_dim) → [ReLU → Linear(H, H)] × (n_layers - 1)
            → Linear(H, act_dim) → act

    All layers fully connected (dense). Parameter count depends on hidden_dim
    and n_layers, and is independent of any graph subgraph size.

    [SA-008 analogue: identity, obs passed directly to first layer]
    [SA-009 analogue: final linear to act_dim]
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        H = config.hidden_dim
        D_obs = config.obs_dim
        D_act = config.act_dim

        layers: list[nn.Module] = [nn.Linear(D_obs, H), nn.ReLU()]
        for _ in range(config.n_layers - 1):
            layers += [nn.Linear(H, H), nn.ReLU()]
        layers.append(nn.Linear(H, D_act))

        self.net = nn.Sequential(*layers)
        self._H = H

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def reset_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        pass  # stateless

    def flops_estimate(self) -> int:
        D_obs = self.config.obs_dim
        H = self._H
        D_act = self.config.act_dim
        n = self.config.n_layers
        # first layer + (n-1) hidden + output
        return 2 * D_obs * H + 2 * (n - 1) * H * H + 2 * H * D_act

    @classmethod
    def build(
        cls,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        seed: int = 42,
    ) -> "A7MLP":
        cfg = AgentConfig(
            arch_id="A7-MLP",
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=(),
        )
        return cls(cfg)


# ---------------------------------------------------------------------------
# A8 — RNN (LSTM or GRU)
# ---------------------------------------------------------------------------


class A8RNN(AbstractAgent):
    """
    A8: Recurrent network (LSTM or GRU) baseline.

    Architecture:
        obs → Linear(obs_dim, hidden_dim) → [LSTM/GRU] → Linear(H, act_dim)

    The hidden state is carried across time steps within an episode.
    reset_state() clears the hidden state at episode boundaries.

    Parameters
    ----------
    config : AgentConfig
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    """

    def __init__(
        self,
        config: AgentConfig,
        cell_type: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__(config)
        self.cell_type = cell_type
        H = config.hidden_dim
        D_obs = config.obs_dim
        D_act = config.act_dim

        self.input_proj = nn.Linear(D_obs, H)

        if cell_type == "lstm":
            self.rnn = nn.LSTM(H, H, num_layers=config.n_layers, batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(H, H, num_layers=config.n_layers, batch_first=True)
        else:
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {cell_type!r}")

        self.readout = nn.Linear(H, D_act)
        self._H = H
        self._hidden: Optional[tuple | torch.Tensor] = None

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Clear RNN hidden state. Call at the start of each episode."""
        device = device or next(self.parameters()).device
        H = self._H
        n_layers = self.config.n_layers
        zeros = torch.zeros(n_layers, batch_size, H, device=device)
        if self.cell_type == "lstm":
            self._hidden = (zeros, zeros.clone())
        else:
            self._hidden = zeros

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (B, D_obs)
            One timestep per call. Adds sequence dimension internally.

        Returns
        -------
        (B, D_act)
        """
        B = obs.shape[0]
        if self._hidden is None:
            self.reset_state(batch_size=B, device=obs.device)

        x = self.input_proj(obs)          # (B, H)
        x = x.unsqueeze(1)                # (B, 1, H) — single timestep

        if self.cell_type == "lstm":
            # Detach to prevent backprop through full episode history
            h, c = self._hidden
            h, c = h[:, :B, :].contiguous(), c[:, :B, :].contiguous()
            out, (h_new, c_new) = self.rnn(x, (h, c))
            self._hidden = (h_new.detach(), c_new.detach())
        else:
            h = self._hidden[:, :B, :].contiguous()
            out, h_new = self.rnn(x, h)
            self._hidden = h_new.detach()

        out = out.squeeze(1)              # (B, H)
        return self.readout(out)          # (B, D_act)

    def flops_estimate(self) -> int:
        H = self._H
        D_obs = self.config.obs_dim
        D_act = self.config.act_dim
        n = self.config.n_layers

        proj = 2 * D_obs * H
        # LSTM: 4 gates, each H×H + H×H; GRU: 3 gates
        gate_factor = 4 if self.cell_type == "lstm" else 3
        rnn_flops = n * gate_factor * (2 * H * H)
        readout = 2 * H * D_act
        return proj + rnn_flops + readout

    @classmethod
    def build(
        cls,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        cell_type: str = "lstm",
        seed: int = 42,
    ) -> "A8RNN":
        cfg = AgentConfig(
            arch_id=f"A8-{cell_type.upper()}",
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            seed=seed,
            assumption_ids=(),
        )
        return cls(cfg, cell_type=cell_type)
