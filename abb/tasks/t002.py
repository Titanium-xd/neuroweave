"""
abb.tasks.t002 — T-002: Temporal Sequence Memory

A delay-recall task. The agent observes a signal at the beginning of an episode,
then receives noise/silence for several steps, and must output the original signal's
class at the final step.

Tests temporal integration and memory capacity. Stateless models should fail (chance accuracy).

Format:
  obs is (B, T, D_obs)
  labels is (B,) - representing the target class

Since runner.py expects to pass obs to model, we will update runner.py to handle
3D tensors by looping over the sequence dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import TensorDataset

@dataclass
class T002Config:
    task_id: str = "T-002"
    description: str = "Temporal Sequence Memory (Delay-Recall)"
    d_obs: int = 16
    d_act: int = 2
    seq_len: int = 10
    n_train: int = 800
    n_dev: int = 100
    n_test: int = 100
    seed: int = 42

class T002Dataset:
    """
    Generates T-002 sequence dataset.
    Sequence length T.
    t=0: One-hot encoded class signal in the first two dimensions of D_obs.
    t=1..T-1: Zero vectors.
    """
    def __init__(self, config: T002Config) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        
        # Generate datasets
        self._X_train, self._y_train = self._generate(config.n_train)
        self._X_dev, self._y_dev = self._generate(config.n_dev)
        self._X_test, self._y_test = self._generate(config.n_test)
        
        # PyTorch TensorDatasets
        self.train = self._to_dataset(self._X_train, self._y_train)
        self.dev = self._to_dataset(self._X_dev, self._y_dev)
        self.test = self._to_dataset(self._X_test, self._y_test)
        
    def _generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        D = self.config.d_obs
        T = self.config.seq_len
        
        # Random binary classes
        y = self._rng.integers(0, 2, size=n_samples)
        
        # Initialize sequences with zeros
        X = np.zeros((n_samples, T, D), dtype=np.float32)
        
        # Inject the signal at t=0
        for i in range(n_samples):
            # One-hot encoding of the class at the start of the sequence
            X[i, 0, y[i]] = 1.0
            
            # Optional: Add small background noise to all steps
            X[i] += self._rng.normal(0, 0.05, size=(T, D)).astype(np.float32)
            
        return X, y

    def _to_dataset(self, X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y).long()
        )
