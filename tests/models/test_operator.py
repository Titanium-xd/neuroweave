"""
tests/models/test_operator.py — Unit tests for SignedSparseLinear and ConnectomeLayer.

Covers:
- Sparse matrix construction and forward pass
- Directed edges (asymmetric adjacency)
- Signed weights (positive, negative, zero)
- Frozen vs trainable weight variants
- Message passing correctness (manual validation)
- FLOPs estimate
- Device-agnostic behaviour
- COO duplicate summation
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from abb.models.operator import ConnectomeLayer, SignedSparseLinear


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_coo(n: int, edges: list[tuple[int, int, float]]) -> sp.coo_matrix:
    """Create a COO matrix from a list of (row, col, value) triples."""
    if not edges:
        return sp.coo_matrix((n, n))
    rows, cols, vals = zip(*edges)
    return sp.coo_matrix(
        (np.array(vals, dtype=np.float32), (list(rows), list(cols))),
        shape=(n, n),
    )


@pytest.fixture()
def simple_adj() -> sp.coo_matrix:
    """
    4-node graph:
        0 → 1 (weight=+2.0)
        1 → 2 (weight=-3.0)
        2 → 0 (weight=+1.0)
        0 → 3 (weight=+0.5)
    """
    return _make_coo(4, [(0, 1, 2.0), (1, 2, -3.0), (2, 0, 1.0), (0, 3, 0.5)])


@pytest.fixture()
def single_edge_adj() -> sp.coo_matrix:
    """2-node graph: 0 → 1 (weight=+5.0). Easy to verify by hand."""
    return _make_coo(2, [(0, 1, 5.0)])


# ---------------------------------------------------------------------------
# SignedSparseLinear
# ---------------------------------------------------------------------------


class TestSignedSparseLinear:
    def test_construction_frozen(self, simple_adj):
        op = SignedSparseLinear(simple_adj, trainable_weights=False)
        assert op.n_nodes == 4
        assert op.nnz == 4
        assert op.edge_scale is None

    def test_construction_trainable(self, simple_adj):
        op = SignedSparseLinear(simple_adj, trainable_weights=True)
        assert op.edge_scale is not None
        assert op.edge_scale.shape == (4,)
        assert op.edge_scale.requires_grad

    def test_no_grad_on_base_vals(self, simple_adj):
        op = SignedSparseLinear(simple_adj, trainable_weights=True)
        assert not op._base_vals.requires_grad

    def test_forward_single_edge_by_hand(self, single_edge_adj):
        """h_1 = 5.0 * x_0 (only edge: 0→1, weight=5)"""
        op = SignedSparseLinear(single_edge_adj, trainable_weights=False)
        x = torch.tensor([[2.0, 0.0]])  # (B=1, N=2)
        h = op(x)
        assert h.shape == (1, 2)
        # Node 0 receives nothing → h[0,0] = 0
        assert abs(h[0, 0].item()) < 1e-5
        # Node 1 receives from node 0: h[0,1] = 5.0 * 2.0 = 10.0
        assert abs(h[0, 1].item() - 10.0) < 1e-4

    def test_forward_negative_weight(self):
        """h_1 = -3.0 * x_0"""
        adj = _make_coo(2, [(0, 1, -3.0)])
        op = SignedSparseLinear(adj, trainable_weights=False)
        x = torch.tensor([[4.0, 0.0]])
        h = op(x)
        assert abs(h[0, 1].item() - (-12.0)) < 1e-4

    def test_forward_output_shape(self, simple_adj):
        op = SignedSparseLinear(simple_adj, trainable_weights=False)
        B = 5
        x = torch.randn(B, 4)
        h = op(x)
        assert h.shape == (B, 4)

    def test_forward_wrong_input_dim_raises(self, simple_adj):
        op = SignedSparseLinear(simple_adj, trainable_weights=False)
        with pytest.raises(AssertionError):
            op(torch.randn(2, 3))  # wrong N

    def test_directed_asymmetry(self):
        """0→1 edge should NOT create a 1→0 message."""
        adj = _make_coo(2, [(0, 1, 1.0)])
        op = SignedSparseLinear(adj, trainable_weights=False)
        x = torch.tensor([[0.0, 3.0]])  # node 1 active, not node 0
        h = op(x)
        # No edge from 1→0, so node 0 receives nothing
        assert abs(h[0, 0].item()) < 1e-5
        # No edge into node 1 from node 1, so node 1 receives nothing
        assert abs(h[0, 1].item()) < 1e-5

    def test_trainable_edge_scale_applied(self):
        """Doubling edge_scale should double the output."""
        adj = _make_coo(2, [(0, 1, 2.0)])
        op = SignedSparseLinear(adj, trainable_weights=True)
        with torch.no_grad():
            op.edge_scale.fill_(2.0)  # scale = 2 → effective weight = 4
        x = torch.tensor([[1.0, 0.0]])
        h = op(x)
        assert abs(h[0, 1].item() - 4.0) < 1e-4

    def test_gradient_flows_through_trainable_scale(self):
        adj = _make_coo(2, [(0, 1, 1.0)])
        op = SignedSparseLinear(adj, trainable_weights=True)
        x = torch.tensor([[1.0, 0.0]])
        h = op(x)
        loss = h.sum()
        loss.backward()
        assert op.edge_scale.grad is not None
        assert op.edge_scale.grad.shape == (1,)

    def test_no_gradient_when_frozen(self):
        adj = _make_coo(2, [(0, 1, 1.0)])
        op = SignedSparseLinear(adj, trainable_weights=False)
        x = torch.tensor([[1.0, 0.0]], requires_grad=True)
        h = op(x)
        loss = h.sum()
        loss.backward()
        # No learnable params — edge_scale is None, no gradients stored
        assert op.edge_scale is None

    def test_coo_duplicates_are_summed(self):
        """Two edges with the same (pre, post) should be summed."""
        n = 2
        rows = np.array([0, 0], dtype=np.int32)
        cols = np.array([1, 1], dtype=np.int32)
        vals = np.array([3.0, 2.0], dtype=np.float32)
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
        op = SignedSparseLinear(adj, trainable_weights=False)
        assert op.nnz == 1  # duplicates merged → single entry with value 5
        x = torch.tensor([[1.0, 0.0]])
        h = op(x)
        assert abs(h[0, 1].item() - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# ConnectomeLayer
# ---------------------------------------------------------------------------


class TestConnectomeLayer:
    def test_output_shape(self, simple_adj):
        layer = ConnectomeLayer(simple_adj, trainable_weights=True, trainable_bias=True)
        x = torch.randn(3, 4)
        out = layer(x)
        assert out.shape == (3, 4)

    def test_relu_activation_nonneg(self, simple_adj):
        layer = ConnectomeLayer(simple_adj, activation="relu")
        x = torch.randn(10, 4)
        out = layer(x)
        assert (out >= 0).all()

    def test_tanh_bounded(self, simple_adj):
        layer = ConnectomeLayer(simple_adj, activation="tanh")
        x = torch.randn(10, 4) * 100
        out = layer(x)
        assert (out >= -1).all() and (out <= 1).all()

    def test_invalid_activation_raises(self, simple_adj):
        with pytest.raises(ValueError, match="Unknown activation"):
            ConnectomeLayer(simple_adj, activation="sigmoid")

    def test_trainable_bias_param(self, simple_adj):
        layer = ConnectomeLayer(simple_adj, trainable_bias=True)
        assert isinstance(layer.bias, torch.nn.Parameter)
        assert layer.bias.requires_grad

    def test_frozen_bias_buffer(self, simple_adj):
        layer = ConnectomeLayer(simple_adj, trainable_bias=False)
        # bias is a buffer, not a Parameter
        assert not isinstance(layer.bias, torch.nn.Parameter)

    def test_flops_returns_positive(self, simple_adj):
        layer = ConnectomeLayer(simple_adj)
        assert layer.flops() == 2 * simple_adj.nnz

    def test_nnz_property(self, simple_adj):
        layer = ConnectomeLayer(simple_adj)
        assert layer.nnz == simple_adj.nnz

    def test_deterministic_given_same_input(self, simple_adj):
        """Same input should always produce same output (no dropout etc)."""
        layer = ConnectomeLayer(simple_adj)
        layer.eval()
        x = torch.randn(2, 4)
        out1 = layer(x)
        out2 = layer(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_consistency(self, simple_adj):
        """A batch of identical inputs should give identical outputs."""
        layer = ConnectomeLayer(simple_adj)
        layer.eval()
        x_single = torch.randn(1, 4)
        x_batch = x_single.expand(5, -1)
        out_single = layer(x_single)
        out_batch = layer(x_batch)
        torch.testing.assert_close(out_batch[0], out_single[0])
