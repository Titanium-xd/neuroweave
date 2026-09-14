"""
tests/models/test_baselines.py — Tests for A0, A7-MLP, A8-LSTM, A8-GRU.

Covers:
- A0 produces random output (different each call)
- A0 has zero parameters
- MLP output shape and parameter count
- RNN output shape, hidden state management
- LSTM vs GRU variant selection
- RNN determinism after reset
- flops_estimate() values
"""

from __future__ import annotations

import pytest
import torch

from abb.models.baselines import A0Random, A7MLP, A8RNN

OBS = 8
ACT = 4
H = 16


@pytest.fixture()
def a0() -> A0Random:
    return A0Random.build(obs_dim=OBS, act_dim=ACT, seed=42)


@pytest.fixture()
def mlp() -> A7MLP:
    return A7MLP.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=3, seed=42)


@pytest.fixture()
def lstm() -> A8RNN:
    return A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=2, cell_type="lstm")


@pytest.fixture()
def gru() -> A8RNN:
    return A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=2, cell_type="gru")


# ---------------------------------------------------------------------------
# A0 Random Policy
# ---------------------------------------------------------------------------


class TestA0Random:
    def test_output_shape(self, a0):
        obs = torch.randn(5, OBS)
        assert a0(obs).shape == (5, ACT)

    def test_zero_parameters(self, a0):
        pc = a0.param_count()
        assert pc["total"] == 0
        assert pc["trainable"] == 0

    def test_different_outputs_each_call(self, a0):
        """Random policy must differ between calls (with overwhelming probability)."""
        obs = torch.randn(1, OBS)
        out1 = a0(obs)
        out2 = a0(obs)
        assert not torch.allclose(out1, out2)

    def test_output_range(self, a0):
        """torch.rand → [0, 1)"""
        obs = torch.randn(100, OBS)
        out = a0(obs)
        assert (out >= 0).all() and (out < 1).all()

    def test_arch_id(self, a0):
        assert a0.config.arch_id == "A0"

    def test_flops_positive(self, a0):
        assert a0.flops_estimate() > 0


# ---------------------------------------------------------------------------
# A7 MLP
# ---------------------------------------------------------------------------


class TestA7MLP:
    def test_output_shape(self, mlp):
        obs = torch.randn(6, OBS)
        assert mlp(obs).shape == (6, ACT)

    def test_all_trainable(self, mlp):
        pc = mlp.param_count()
        assert pc["frozen"] == 0
        assert pc["trainable"] > 0

    def test_gradient_flows(self, mlp):
        obs = torch.randn(3, OBS)
        loss = mlp(obs).sum()
        loss.backward()
        for p in mlp.parameters():
            assert p.grad is not None

    def test_flops_estimate_mlp(self, mlp):
        f = mlp.flops_estimate()
        # At minimum: first layer 2*OBS*H + output 2*H*ACT
        assert f >= 2 * OBS * H + 2 * H * ACT

    def test_n_layers_scales_params(self):
        """More layers → more parameters."""
        m2 = A7MLP.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=2)
        m4 = A7MLP.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=4)
        assert m4.param_count()["total"] > m2.param_count()["total"]

    def test_deterministic_eval(self, mlp):
        mlp.eval()
        obs = torch.randn(2, OBS)
        assert torch.allclose(mlp(obs), mlp(obs))

    def test_arch_id(self, mlp):
        assert mlp.config.arch_id == "A7-MLP"


# ---------------------------------------------------------------------------
# A8 LSTM
# ---------------------------------------------------------------------------


class TestA8LSTM:
    def test_output_shape(self, lstm):
        lstm.reset_state(batch_size=4)
        obs = torch.randn(4, OBS)
        assert lstm(obs).shape == (4, ACT)

    def test_hidden_state_initialised_on_first_call(self, lstm):
        assert lstm._hidden is None
        obs = torch.randn(1, OBS)
        lstm(obs)
        assert lstm._hidden is not None

    def test_hidden_is_tuple_for_lstm(self, lstm):
        lstm.reset_state(batch_size=1)
        assert isinstance(lstm._hidden, tuple)
        assert len(lstm._hidden) == 2  # (h, c)

    def test_reset_zeroes_hidden(self, lstm):
        lstm.reset_state(batch_size=2)
        h, c = lstm._hidden
        assert h.abs().max().item() < 1e-6
        assert c.abs().max().item() < 1e-6

    def test_state_evolves_across_steps(self, lstm):
        lstm.eval()
        lstm.reset_state(batch_size=1)
        obs = torch.randn(1, OBS)
        out1 = lstm(obs)
        out2 = lstm(obs)  # same input, state has evolved
        assert not torch.allclose(out1, out2)

    def test_reproducible_after_reset(self, lstm):
        lstm.eval()
        obs = torch.randn(1, OBS)
        lstm.reset_state(batch_size=1)
        o1 = lstm(obs)
        lstm.reset_state(batch_size=1)
        o2 = lstm(obs)
        torch.testing.assert_close(o1, o2)

    def test_flops_lstm(self, lstm):
        f = lstm.flops_estimate()
        assert f > 0

    def test_arch_id(self, lstm):
        assert "LSTM" in lstm.config.arch_id


# ---------------------------------------------------------------------------
# A8 GRU
# ---------------------------------------------------------------------------


class TestA8GRU:
    def test_output_shape(self, gru):
        gru.reset_state(batch_size=3)
        obs = torch.randn(3, OBS)
        assert gru(obs).shape == (3, ACT)

    def test_hidden_is_tensor_for_gru(self, gru):
        gru.reset_state(batch_size=1)
        assert isinstance(gru._hidden, torch.Tensor)
        assert not isinstance(gru._hidden, tuple)

    def test_reset_zeroes_hidden_gru(self, gru):
        gru.reset_state(batch_size=1)
        assert gru._hidden.abs().max().item() < 1e-6

    def test_gru_has_fewer_params_than_lstm(self):
        """GRU has 3 gates vs LSTM 4 → fewer parameters for same H."""
        lstm = A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=1, cell_type="lstm")
        gru = A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=H, n_layers=1, cell_type="gru")
        assert gru.param_count()["total"] < lstm.param_count()["total"]

    def test_invalid_cell_type_raises(self):
        from abb.models.base import AgentConfig
        cfg = AgentConfig(arch_id="A8-TEST", obs_dim=OBS, act_dim=ACT)
        with pytest.raises(ValueError, match="cell_type"):
            A8RNN(cfg, cell_type="transformer")
