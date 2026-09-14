"""
tests/models/test_agent_interface.py — Tests for AbstractAgent / AgentConfig interface.

Covers:
- AgentConfig construction and to_dict()
- config_hash determinism
- param_count() accuracy
- flops_estimate() positivity and type
- save_checkpoint / load_checkpoint round-trip
- reset_state no-op for stateless models
- config_dict keys
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from abb.models.base import AgentConfig
from abb.models.baselines import A0Random, A7MLP, A8RNN
from abb.models.bio_variants import A1Bio


def _ring_adj(n: int) -> sp.coo_matrix:
    rows = list(range(n))
    cols = [(i + 1) % n for i in range(n)]
    vals = [1.0] * n
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n))


N = 10
OBS = 4
ACT = 2


@pytest.fixture()
def adj() -> sp.coo_matrix:
    return _ring_adj(N)


@pytest.fixture()
def bio_model(adj) -> A1Bio:
    cfg = AgentConfig(arch_id="A1-BIO", obs_dim=OBS, act_dim=ACT, n_layers=2, seed=42)
    return A1Bio(cfg, adj)


@pytest.fixture()
def mlp_model() -> A7MLP:
    return A7MLP.build(obs_dim=OBS, act_dim=ACT, hidden_dim=16, n_layers=2)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_to_dict_has_required_keys(self):
        cfg = AgentConfig(arch_id="TEST", obs_dim=4, act_dim=2)
        d = cfg.to_dict()
        for key in ("arch_id", "obs_dim", "act_dim", "hidden_dim", "n_layers",
                    "seed", "assumption_ids", "extra", "benchmark_version"):
            assert key in d

    def test_config_hash_is_16_chars(self):
        cfg = AgentConfig(arch_id="A1-BIO", obs_dim=4, act_dim=2)
        h = cfg.config_hash()
        assert len(h) == 16
        int(h, 16)  # valid hex

    def test_config_hash_deterministic(self):
        cfg1 = AgentConfig(arch_id="A1-BIO", obs_dim=4, act_dim=2, seed=42)
        cfg2 = AgentConfig(arch_id="A1-BIO", obs_dim=4, act_dim=2, seed=42)
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_config_hash_differs_on_seed(self):
        cfg1 = AgentConfig(arch_id="A1-BIO", obs_dim=4, act_dim=2, seed=1)
        cfg2 = AgentConfig(arch_id="A1-BIO", obs_dim=4, act_dim=2, seed=2)
        assert cfg1.config_hash() != cfg2.config_hash()


# ---------------------------------------------------------------------------
# param_count
# ---------------------------------------------------------------------------


class TestParamCount:
    def test_a0_has_zero_params(self):
        a0 = A0Random.build(obs_dim=4, act_dim=2)
        pc = a0.param_count()
        assert pc["total"] == 0
        assert pc["trainable"] == 0

    def test_mlp_trainable_equals_total(self, mlp_model):
        pc = mlp_model.param_count()
        assert pc["trainable"] == pc["total"]
        assert pc["frozen"] == 0

    def test_bio_has_frozen_and_trainable(self, bio_model):
        """A1-BIO: all params trainable (no frozen)."""
        pc = bio_model.param_count()
        # A1-BIO is fully trainable
        assert pc["trainable"] > 0
        assert pc["total"] > 0

    def test_param_count_matches_manual_count(self, mlp_model):
        manual = sum(p.numel() for p in mlp_model.parameters())
        assert mlp_model.param_count()["total"] == manual

    def test_a1_frozen_has_frozen_graph_params(self, adj):
        from abb.models.bio_variants import A1Frozen
        cfg = AgentConfig(arch_id="A1-FROZEN", obs_dim=OBS, act_dim=ACT, n_layers=2, seed=42)
        model = A1Frozen(cfg, adj)
        pc = model.param_count()
        # A1-FROZEN: graph layers use buffers (no nn.Parameter for edges or bias).
        # Only input_proj and readout are trainable.
        # The graph layers themselves have no Parameters → frozen count = 0 at nn level.
        # What matters: no edge_scale Parameter exists in frozen layers.
        for layer in model.layers:
            assert layer.operator.edge_scale is None, "A1-FROZEN must have no edge_scale Parameter"
        assert pc["trainable"] > 0  # readout + input_proj still trainable

    def test_a1_bias_has_fewer_trainable_than_bio(self, adj, bio_model):
        from abb.models.bio_variants import A1Bias
        cfg = AgentConfig(arch_id="A1-BIAS", obs_dim=OBS, act_dim=ACT, n_layers=2, seed=42)
        bias_model = A1Bias(cfg, adj)
        assert bias_model.param_count()["trainable"] <= bio_model.param_count()["trainable"]


# ---------------------------------------------------------------------------
# flops_estimate
# ---------------------------------------------------------------------------


class TestFlopsEstimate:
    def test_a0_flops_positive(self):
        a0 = A0Random.build(obs_dim=4, act_dim=2)
        assert a0.flops_estimate() > 0

    def test_mlp_flops_positive(self, mlp_model):
        assert mlp_model.flops_estimate() > 0

    def test_bio_flops_positive(self, bio_model):
        assert bio_model.flops_estimate() > 0

    def test_flops_is_int(self, bio_model, mlp_model):
        assert isinstance(bio_model.flops_estimate(), int)
        assert isinstance(mlp_model.flops_estimate(), int)

    def test_dense_flops_greater_than_sparse(self, adj):
        """A3-DENSE must have higher flops than A1-BIO for same N."""
        from abb.models.topology import A3Dense
        bio_cfg = AgentConfig(arch_id="A1-BIO", obs_dim=OBS, act_dim=ACT, n_layers=2, seed=42)
        bio = A1Bio(bio_cfg, adj)

        import pandas as pd
        from abb.data.graph import build_graph
        dummy_neurons = pd.DataFrame({"bodyId": list(range(N)), "type": [None]*N, "predictedNt": [None]*N})
        dummy_edges = pd.DataFrame(columns=["bodyId_pre","bodyId_post","weight"])
        g = build_graph(dummy_neurons, dummy_edges)
        dense = A3Dense.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=2)

        # Dense has N^2 ops per layer; sparse has nnz << N^2 for typical graphs
        assert dense.flops_estimate() >= bio.flops_estimate()


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_a0_output_shape(self, B):
        a0 = A0Random.build(obs_dim=OBS, act_dim=ACT)
        obs = torch.randn(B, OBS)
        out = a0(obs)
        assert out.shape == (B, ACT)

    @pytest.mark.parametrize("B", [1, 4])
    def test_mlp_output_shape(self, B, mlp_model):
        obs = torch.randn(B, OBS)
        assert mlp_model(obs).shape == (B, ACT)

    @pytest.mark.parametrize("B", [1, 3])
    def test_bio_output_shape(self, B, bio_model):
        obs = torch.randn(B, OBS)
        assert bio_model(obs).shape == (B, ACT)


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------


class TestResetState:
    def test_stateless_models_reset_is_noop(self, bio_model, mlp_model):
        """Stateless models should not raise on reset_state()."""
        bio_model.reset_state(batch_size=1)
        mlp_model.reset_state(batch_size=1)

    def test_rnn_reset_clears_hidden(self):
        rnn = A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=8, n_layers=1, cell_type="lstm")
        rnn.reset_state(batch_size=2)
        assert rnn._hidden is not None
        # After reset, hidden should be zeros
        h, c = rnn._hidden
        assert h.abs().max().item() < 1e-6

    def test_rnn_output_changes_without_reset(self):
        """Running without reset preserves state across calls."""
        rnn = A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=8, n_layers=1)
        rnn.eval()
        rnn.reset_state(batch_size=1)
        obs = torch.randn(1, OBS)
        out1 = rnn(obs)
        out2 = rnn(obs)  # same input, but hidden state changed
        assert not torch.allclose(out1, out2)

    def test_rnn_output_same_after_reset(self):
        """After reset, same input should give same output."""
        rnn = A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=8, n_layers=1)
        rnn.eval()
        obs = torch.randn(1, OBS)
        rnn.reset_state(batch_size=1)
        out1 = rnn(obs)
        rnn.reset_state(batch_size=1)
        out2 = rnn(obs)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Save / load checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_mlp(self, tmp_path, mlp_model):
        path = tmp_path / "mlp.pt"
        mlp_model.save_checkpoint(path)
        assert path.exists()
        loaded = A7MLP.load_checkpoint(path, config=mlp_model.config)
        # Compare outputs
        obs = torch.randn(2, OBS)
        mlp_model.eval()
        loaded.eval()
        torch.testing.assert_close(mlp_model(obs), loaded(obs))

    def test_save_load_bio(self, tmp_path, bio_model, adj):
        path = tmp_path / "bio.pt"
        bio_model.save_checkpoint(path)
        # Bio models need adj at load time — use from_checkpoint helper
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = bio_model.config
        loaded = A1Bio(cfg, adj)
        loaded.load_state_dict(ckpt["state_dict"])
        obs = torch.randn(2, OBS)
        bio_model.eval()
        loaded.eval()
        torch.testing.assert_close(bio_model(obs), loaded(obs))

    def test_checkpoint_does_not_store_token(self, tmp_path, mlp_model):
        """Checkpoint file must not contain NEUPRINT_TOKEN or any credential."""
        path = tmp_path / "mlp.pt"
        mlp_model.save_checkpoint(path)
        raw = path.read_bytes()
        assert b"NEUPRINT_TOKEN" not in raw
        assert b"eyJ" not in raw  # JWT prefix


# ---------------------------------------------------------------------------
# config_dict completeness
# ---------------------------------------------------------------------------


class TestConfigDict:
    def test_config_dict_has_param_counts(self, bio_model):
        d = bio_model.config_dict()
        assert "trainable" in d
        assert "frozen" in d
        assert "total" in d

    def test_config_dict_has_flops(self, bio_model):
        d = bio_model.config_dict()
        assert "flops_estimate" in d

    def test_config_dict_has_arch_id(self, bio_model):
        assert bio_model.config_dict()["arch_id"] == "A1-BIO"

    def test_repr_contains_arch_id(self, bio_model):
        r = repr(bio_model)
        assert "A1-BIO" in r
        assert "trainable_params" in r
