"""
abb.tasks.t001 — T-001: Binary Pattern Discrimination

TASK DESCRIPTION
----------------
A 64-dimensional binary observation is presented to the model.
The model must predict the binary class (0 or 1).

STIMULUS DESIGN  [ASSUMPTION SA-T001]
--------------------------------------
Class 0 : 64 bits drawn i.i.d. from Bernoulli(p_low = 0.25)
           → sparse activation pattern, predominantly silent
Class 1 : 64 bits drawn i.i.d. from Bernoulli(p_high = 0.75)
           → dense activation pattern, predominantly active

This creates two statistically distinguishable activation regimes:
"sparse input" vs "dense input". The key features are spread across
ALL 64 dimensions — a model cannot solve this by monitoring a single
dimension. It must integrate across its input.

This is an input-integration task, not a memory or sequence task.
It tests whether the network can discriminate two global activation
states, which maps naturally to the connectome's role in integrating
distributed sensory input.

SPECIFICATION  [T-001-v0.1]
----------------------------
- D_obs      = 64
- D_act      = 2  (logits for class 0 / class 1)
- n_train    = 800
- n_dev      = 100
- n_test     = 100
- Balance    = 50/50 per class in every split
- Seed       = fixed per split (train=1, dev=2, test=3)
- Input type = float32 in {0.0, 1.0}

Trivial baselines:
- A0-CHANCE     : random uniform prediction → ~50% accuracy
- A0-MAJORITY   : always predict class 1 → 50% (balanced dataset)

[SA-T001] Engineering assumption: Bernoulli p_low=0.25, p_high=0.75 chosen
to make the task learnable but not trivially solved by a single threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class T001Config:
    """Frozen configuration for T-001. Changing any field changes the task."""

    task_id: str = "T-001"
    description: str = "Binary Pattern Discrimination"
    d_obs: int = 64
    d_act: int = 2
    n_train: int = 800
    n_dev: int = 100
    n_test: int = 100
    p_low: float = 0.25    # Bernoulli p for class 0
    p_high: float = 0.75   # Bernoulli p for class 1
    seed_train: int = 1
    seed_dev: int = 2
    seed_test: int = 3
    assumption_ids: tuple = ("SA-T001",)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "d_obs": self.d_obs,
            "d_act": self.d_act,
            "n_train": self.n_train,
            "n_dev": self.n_dev,
            "n_test": self.n_test,
            "p_low": self.p_low,
            "p_high": self.p_high,
            "seed_train": self.seed_train,
            "seed_dev": self.seed_dev,
            "seed_test": self.seed_test,
            "assumption_ids": list(self.assumption_ids),
        }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _generate_split(
    n: int,
    d: int,
    p_low: float,
    p_high: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n samples with balanced classes.

    Returns
    -------
    X : float32 array (n, d)  — binary observations
    y : int64 array  (n,)     — class labels {0, 1}
    """
    rng = np.random.default_rng(seed)
    n_per_class = n // 2

    x0 = rng.binomial(1, p_low,  size=(n_per_class, d)).astype(np.float32)
    x1 = rng.binomial(1, p_high, size=(n_per_class, d)).astype(np.float32)

    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([
        np.zeros(n_per_class, dtype=np.int64),
        np.ones(n_per_class, dtype=np.int64),
    ])

    # Shuffle jointly (same seed offset)
    idx = rng.permutation(n)
    return X[idx], y[idx]


class T001Dataset:
    """
    T-001 dataset: three fixed splits.

    Attributes
    ----------
    train, dev, test : TensorDataset
        Each yields (obs: float32 (64,), label: int64 scalar).
    config : T001Config
    """

    def __init__(self, config: Optional[T001Config] = None) -> None:
        self.config = config or T001Config()
        cfg = self.config

        X_tr, y_tr = _generate_split(cfg.n_train, cfg.d_obs, cfg.p_low, cfg.p_high, cfg.seed_train)
        X_dv, y_dv = _generate_split(cfg.n_dev,   cfg.d_obs, cfg.p_low, cfg.p_high, cfg.seed_dev)
        X_te, y_te = _generate_split(cfg.n_test,  cfg.d_obs, cfg.p_low, cfg.p_high, cfg.seed_test)

        self.train = TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        )
        self.dev   = TensorDataset(
            torch.from_numpy(X_dv), torch.from_numpy(y_dv)
        )
        self.test  = TensorDataset(
            torch.from_numpy(X_te), torch.from_numpy(y_te)
        )

        # Store raw arrays for inspection
        self._X_train, self._y_train = X_tr, y_tr
        self._X_dev,   self._y_dev   = X_dv, y_dv
        self._X_test,  self._y_test  = X_te, y_te

    # ------------------------------------------------------------------

    def sample(self, split: str = "train", idx: int = 0) -> dict:
        """Return a single labelled example with metadata."""
        X, y = {
            "train": (self._X_train, self._y_train),
            "dev":   (self._X_dev,   self._y_dev),
            "test":  (self._X_test,  self._y_test),
        }[split]
        obs = X[idx]
        label = int(y[idx])
        return {
            "split": split,
            "idx": idx,
            "obs": obs,
            "label": label,
            "label_name": "dense" if label == 1 else "sparse",
            "obs_sum": float(obs.sum()),
            "obs_mean": float(obs.mean()),
        }

    def majority_accuracy(self) -> float:
        """Accuracy of a majority-class predictor (50% for balanced dataset)."""
        majority = int((self._y_train == 1).mean() >= 0.5)
        return float((self._y_test == majority).mean())

    def chance_accuracy(self) -> float:
        """Expected accuracy of a random uniform predictor = 50%."""
        return 0.50

    def class_balance(self, split: str = "train") -> dict:
        y = {"train": self._y_train, "dev": self._y_dev, "test": self._y_test}[split]
        return {"class_0": int((y == 0).sum()), "class_1": int((y == 1).sum())}
