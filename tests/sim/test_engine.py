"""
tests/sim/test_engine.py — Essential tests for ConnectomeSimulator.

Verifies:
1. Activity propagates through real (ring) sparse graph
2. Stimulation produces measurable downstream activity
3. Same seed → deterministic results
4. No dense matrix allocation (nnz check)
5. History shape correctness
6. Reset clears state
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from abb.sim.engine import ConnectomeSimulator, SimConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ring_graph(n: int, weight: float = 1.0):
    """Ring: 0→1→2→...→(n-1)→0. All positive weights."""
    rows = list(range(n))
    cols = [(i + 1) % n for i in range(n)]
    vals = [weight] * n
    adj = sp.coo_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, n))
    df = pd.DataFrame({
        "bodyId": range(n),
        "nt_sign": [1] * n,
        "predictedNt": ["acetylcholine"] * n,
    })
    return adj, df


def _signed_graph(n: int):
    """Same ring but alternating signs (+/-) to test sign propagation."""
    rows = list(range(n))
    cols = [(i + 1) % n for i in range(n)]
    vals = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    adj = sp.coo_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, n))
    df = pd.DataFrame({
        "bodyId": range(n),
        "nt_sign": vals,
        "predictedNt": ["acetylcholine" if i % 2 == 0 else "gaba" for i in range(n)],
    })
    return adj, df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicPropagation:
    def test_downstream_becomes_nonzero(self):
        """Stimulating node 0 in a ring should activate node 1 after one step."""
        adj, df = _ring_graph(10, weight=2.0)
        cfg = SimConfig(decay=0.0, leak=1.0, seed=42, init_noise=0.0)
        sim = ConnectomeSimulator(adj, df, cfg)
        sim.reset()
        sim.stimulate([0], values=np.array([1.0]))
        sim.step()
        # Node 1 receives from node 0 via the ring edge (weight 2.0)
        state = sim.state
        assert abs(state[1]) > 1e-4, f"Node 1 should be active, got {state[1]}"

    def test_silent_without_stimulation(self):
        """Without stimulation and zero init_noise, state stays near zero."""
        adj, df = _ring_graph(10)
        cfg = SimConfig(decay=0.5, leak=0.5, seed=42, init_noise=0.0)
        sim = ConnectomeSimulator(adj, df, cfg)
        sim.reset()
        for _ in range(5):
            sim.step()
        assert np.abs(sim.state).max() < 1e-5

    def test_activity_spreads_after_multiple_steps(self):
        """After k steps, activity should reach node k in a chain."""
        n = 10
        adj, df = _ring_graph(n, weight=2.0)
        cfg = SimConfig(decay=0.0, leak=1.0, seed=42, init_noise=0.0)
        sim = ConnectomeSimulator(adj, df, cfg)
        sim.reset()
        sim.stimulate([0], values=np.array([1.0]))
        for _ in range(3):
            sim.step()
        # After 3 steps, node 3 should have non-trivial activation
        assert abs(sim.state[3]) > 1e-4

    def test_inhibitory_edge_reduces_activation(self):
        """A negative-weight edge should reduce (not increase) downstream activation."""
        adj, df = _signed_graph(4)
        cfg = SimConfig(decay=0.0, leak=1.0, seed=42, init_noise=0.0)
        sim = ConnectomeSimulator(adj, df, cfg)
        sim.reset()
        # Stimulate node 1 (which has a negative outgoing edge to node 2)
        sim.stimulate([1], values=np.array([1.0]))
        sim.step()
        # Node 2 receives -1.0 * 1.0 from node 1 → should be negative
        # (edge 1→2 has weight -1.0 since index 1 is odd in alternating graph)
        assert sim.state[2] < 0, f"Node 2 should be inhibited, got {sim.state[2]}"


class TestDeterminism:
    def test_same_seed_same_result(self):
        adj, df = _ring_graph(20)
        cfg = SimConfig(seed=99, init_noise=0.05, decay=0.1, leak=0.5)

        sim1 = ConnectomeSimulator(adj, df, cfg)
        sim1.reset()
        sim1.stimulate([0, 5, 10])
        sim1.run(n_steps=10, stim_indices=[0, 5, 10])

        sim2 = ConnectomeSimulator(adj, df, cfg)
        sim2.reset()
        sim2.stimulate([0, 5, 10])
        sim2.run(n_steps=10, stim_indices=[0, 5, 10])

        np.testing.assert_array_equal(sim1.state, sim2.state)

    def test_different_seed_different_result(self):
        adj, df = _ring_graph(20)
        for seed, arr in [(1, None), (2, None)]:
            cfg = SimConfig(seed=seed, init_noise=0.1)
            sim = ConnectomeSimulator(adj, df, cfg)
            sim.reset()
            sim.stimulate([0])
            sim.run(n_steps=5)
            if arr is None:
                arr = sim.state.copy()
            else:
                # Different seeds → different init noise → different states
                assert not np.allclose(arr, sim.state)


class TestNoDesneMatrix:
    def test_nnz_equals_ring_edges(self):
        n = 30
        adj, df = _ring_graph(n)
        sim = ConnectomeSimulator(adj, df)
        # A ring graph has exactly n edges
        assert sim.nnz == n

    def test_n_nodes_correct(self):
        adj, df = _ring_graph(25)
        sim = ConnectomeSimulator(adj, df)
        assert sim.n_nodes == 25


class TestHistoryAndReset:
    def test_history_shape(self):
        adj, df = _ring_graph(10)
        cfg = SimConfig(n_steps=15)
        sim = ConnectomeSimulator(adj, df, cfg)
        sim.reset()
        sim.run(n_steps=15, stim_indices=[0])
        h = sim.history
        assert h.shape == (15, 10)

    def test_reset_clears_history(self):
        adj, df = _ring_graph(10)
        sim = ConnectomeSimulator(adj, df)
        sim.reset()
        sim.run(n_steps=5, stim_indices=[0])
        sim.reset()
        assert sim.history.shape[0] == 0
        assert np.abs(sim.state).max() < 0.5  # near-zero after reset

    def test_activity_stats_keys(self):
        adj, df = _ring_graph(10)
        sim = ConnectomeSimulator(adj, df)
        sim.reset()
        sim.stimulate([0])
        sim.step()
        stats = sim.activity_stats()
        for key in ("mean_abs", "max_abs", "n_active_exc", "n_active_inh", "n_silent", "step"):
            assert key in stats

    def test_run_returns_history_array(self):
        adj, df = _ring_graph(12)
        sim = ConnectomeSimulator(adj, df)
        sim.reset()
        h = sim.run(n_steps=8, stim_indices=[0, 6])
        assert isinstance(h, np.ndarray)
        assert h.shape == (8, 12)
