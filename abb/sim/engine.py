"""
abb.sim.engine — Minimal connectome activity propagation engine.

WHAT THIS IS
------------
A simple discrete-time rate model for propagating activity through a
MaleCNS-derived sparse directed graph. It is a computational engineering
tool, NOT a biological simulation of a fruit fly.

WHAT THIS IS NOT
----------------
- It is NOT a Leaky Integrate-and-Fire (LIF) simulator.
  LIF is Phase 3.
- It does NOT model fly behavior, navigation, or cognition.
- It does NOT claim to reproduce biological neural dynamics.

ACTIVITY MODEL  [ASSUMPTION SA-010]
------------------------------------
Discrete-time rate propagation:

    x_{t+1} = tanh( (1 - decay) * x_t  +  leak * A_signed @ x_t  +  stim_t )

Where:
    x_t      ∈ R^N   : neuron activation state at step t
    A_signed ∈ R^{N×N} (sparse COO): signed adjacency  (A[i,j] = s_ij * w_ij)
    decay    ∈ (0,1) : fraction of activation lost per step  [SA-010]
    leak     ∈ R+    : message strength scaling  [SA-010]
    stim_t   ∈ R^N  : external stimulation at step t

The tanh nonlinearity bounds output to [-1, +1], matching the excitatory/
inhibitory sign convention from SA-002.

This model is purely an engineering approximation. The parameter choices
(decay, leak) are NOT derived from MaleCNS electrophysiology data.

SA-010: Engineering assumption — rate model for activity propagation demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# SimConfig
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """
    Configuration for the ConnectomeSimulator.

    All parameters that affect dynamics must be here.
    None of these values are measured from MaleCNS electrophysiology.
    They are engineering choices for a demonstration.  [SA-010]
    """

    decay: float = 0.15
    """Fraction of activation retained per step (1 - decay). Range (0, 1)."""

    leak: float = 0.08
    """Scales the propagated message strength. Lower = slower spread."""

    stim_strength: float = 1.0
    """Amplitude of external stimulation applied to input neurons."""

    seed: int = 42
    """Random seed for any stochastic elements (initial noise, random stim)."""

    n_steps: int = 60
    """Number of simulation steps to run."""

    stim_period: int = 12
    """Re-stimulate input neurons every this many steps (0 = once only)."""

    init_noise: float = 0.01
    """Small random initial activation to break symmetry.  [SA-010]"""

    assumption_ids: tuple = ("SA-002", "SA-010")


# ---------------------------------------------------------------------------
# ConnectomeSimulator
# ---------------------------------------------------------------------------


class ConnectomeSimulator:
    """
    Minimal activity propagation engine for a MaleCNS-derived sparse graph.

    Parameters
    ----------
    adj_coo : sp.coo_matrix
        Signed adjacency matrix. Shape (N, N). A[i,j] = weight for i→j.
        Produced by adj_from_graph() from the data layer.
    node_df : pd.DataFrame
        Node metadata (body_id, predictedNt, nt_sign, type …).
    config : SimConfig

    Notes
    -----
    Internally the adjacency is stored as a scipy CSR matrix for fast
    matrix-vector multiply on CPU. No dense N×N matrix is ever allocated.
    """

    def __init__(
        self,
        adj_coo: sp.coo_matrix,
        node_df: "pd.DataFrame",
        config: Optional[SimConfig] = None,
    ) -> None:
        self.config = config or SimConfig()
        self.n_nodes: int = adj_coo.shape[0]
        self.node_df = node_df

        # Store as CSR for efficient row/col slicing
        # Shape (N, N); entry [i, j] = signed weight for edge i → j
        self._adj_csr: sp.csr_matrix = adj_coo.tocsr()
        self._nnz: int = self._adj_csr.nnz

        # State
        self._rng = np.random.default_rng(self.config.seed)
        self._state: np.ndarray = np.zeros(self.n_nodes, dtype=np.float32)
        self._history: list[np.ndarray] = []
        self._step_idx: int = 0

        # Cached: which nodes are excitatory/inhibitory/modulatory?
        if "nt_sign" in node_df.columns:
            self._nt_sign = node_df["nt_sign"].fillna(0).to_numpy(dtype=np.float32)
        else:
            self._nt_sign = np.ones(self.n_nodes, dtype=np.float32)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state to small random noise. Re-seeds from config."""
        self._rng = np.random.default_rng(self.config.seed)
        self._state = (
            self._rng.standard_normal(self.n_nodes).astype(np.float32)
            * self.config.init_noise
        )
        self._history = []
        self._step_idx = 0

    def stimulate(
        self,
        node_indices: list[int] | np.ndarray,
        values: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set activation for selected nodes.

        Parameters
        ----------
        node_indices : list[int] or array of int
            Indices (0-based) of nodes to stimulate.
        values : array | None
            Activation values. If None, uses config.stim_strength for all.
        """
        node_indices = np.asarray(node_indices, dtype=np.int32)
        if values is None:
            v = np.full(len(node_indices), self.config.stim_strength, dtype=np.float32)
        else:
            v = np.asarray(values, dtype=np.float32)
        self._state[node_indices] = v

    def step(self, stim_indices: Optional[list[int]] = None) -> np.ndarray:
        """
        Advance simulation by one step.

        Activity propagation:
            x_{t+1} = tanh( (1-decay)*x_t + leak*(A^T @ x_t) + stim )

        The sparse matmul (A^T @ x) is an N-vector: each node j collects
        signed weighted contributions from its presynaptic partners.
        This uses scipy CSR matvec — no dense matrix allocation.

        Parameters
        ----------
        stim_indices : list[int] | None
            Nodes to re-stimulate this step (set to stim_strength before update).

        Returns
        -------
        np.ndarray
            New state x_{t+1}, shape (N,).
        """
        cfg = self.config

        # Re-stimulate if requested
        if stim_indices is not None:
            self.stimulate(stim_indices)

        # Propagate: A^T @ x  (incoming messages to each node j)
        # A is (N,N) where A[i,j] = weight i→j
        # A^T @ x  gives for each j: sum_{i} A[i,j] * x[i]  = sum of incoming
        incoming = self._adj_csr.T.dot(self._state)  # shape (N,)

        # Update rule  [SA-010]
        new_state = np.tanh(
            (1.0 - cfg.decay) * self._state + cfg.leak * incoming
        )

        self._state = new_state.astype(np.float32)
        self._history.append(self._state.copy())
        self._step_idx += 1
        return self._state

    def run(
        self,
        n_steps: Optional[int] = None,
        stim_indices: Optional[list[int]] = None,
        stim_period: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run the simulation for n_steps.

        Parameters
        ----------
        n_steps : int | None
            Steps to run. Defaults to config.n_steps.
        stim_indices : list[int] | None
            Nodes to stimulate. Applied at step 0 and every stim_period steps.
        stim_period : int | None
            Re-stimulate period. 0 = once only. Defaults to config.stim_period.

        Returns
        -------
        np.ndarray
            Activity history, shape (n_steps, N).
        """
        n = n_steps or self.config.n_steps
        period = stim_period if stim_period is not None else self.config.stim_period

        # Initial stimulation
        if stim_indices is not None:
            self.stimulate(stim_indices)

        for t in range(n):
            re_stim = None
            if stim_indices is not None and period > 0 and t > 0 and t % period == 0:
                re_stim = stim_indices
            self.step(stim_indices=re_stim)

        return self.history

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Current activation state, shape (N,)."""
        return self._state.copy()

    @property
    def history(self) -> np.ndarray:
        """Recorded history of states, shape (n_recorded, N)."""
        if not self._history:
            return np.zeros((0, self.n_nodes), dtype=np.float32)
        return np.stack(self._history, axis=0)

    @property
    def nnz(self) -> int:
        """Number of nonzero edges (no dense matrix ever allocated)."""
        return self._nnz

    def activity_stats(self) -> dict:
        """Summary statistics for current state."""
        s = self._state
        return {
            "mean_abs": float(np.abs(s).mean()),
            "max_abs": float(np.abs(s).max()),
            "n_active_exc": int((s > 0.05).sum()),   # excitatory-range
            "n_active_inh": int((s < -0.05).sum()),  # inhibitory-range
            "n_silent": int((np.abs(s) <= 0.05).sum()),
            "step": self._step_idx,
        }
