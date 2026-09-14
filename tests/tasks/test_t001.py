"""
tests/tasks/test_t001.py — Essential tests for T-001 task.

Verifies:
1. Deterministic dataset generation (same seed → same data)
2. Correct D_obs = 64
3. Train/dev/test are disjoint (different seeds → different samples)
4. Class balance is 50/50
5. Model receives correct input format
6. End-to-end task execution works with A7-MLP
7. Majority baseline == 50% (balanced)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import scipy.sparse as sp
import pandas as pd

from abb.tasks.t001 import T001Config, T001Dataset, _generate_split
from abb.tasks.runner import evaluate, run_task
from abb.models.baselines import A0Random, A7MLP
from abb.data.graph import build_graph


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestT001Dataset:
    def test_d_obs_is_64(self):
        ds = T001Dataset()
        obs, _ = ds.train[0]
        assert obs.shape == (64,)

    def test_n_train_correct(self):
        ds = T001Dataset()
        assert len(ds.train) == 800

    def test_n_dev_correct(self):
        ds = T001Dataset()
        assert len(ds.dev) == 100

    def test_n_test_correct(self):
        ds = T001Dataset()
        assert len(ds.test) == 100

    def test_observations_binary(self):
        ds = T001Dataset()
        X, y = ds._X_train, ds._y_train
        assert set(np.unique(X)).issubset({0.0, 1.0})

    def test_class_balance_train(self):
        ds = T001Dataset()
        bal = ds.class_balance("train")
        assert bal["class_0"] == 400
        assert bal["class_1"] == 400

    def test_class_balance_dev(self):
        ds = T001Dataset()
        bal = ds.class_balance("dev")
        assert bal["class_0"] == 50
        assert bal["class_1"] == 50

    def test_deterministic_same_seed(self):
        ds1 = T001Dataset()
        ds2 = T001Dataset()
        np.testing.assert_array_equal(ds1._X_train, ds2._X_train)
        np.testing.assert_array_equal(ds1._y_train, ds2._y_train)

    def test_splits_differ(self):
        """Train and test should have different samples (different seeds)."""
        ds = T001Dataset()
        # Different seeds → different first rows with overwhelming probability
        assert not np.allclose(ds._X_train[0], ds._X_test[0])

    def test_custom_seed_changes_data(self):
        cfg1 = T001Config(seed_train=1)
        cfg2 = T001Config(seed_train=999)
        ds1 = T001Dataset(cfg1)
        ds2 = T001Dataset(cfg2)
        assert not np.allclose(ds1._X_train, ds2._X_train)

    def test_majority_accuracy_fifty_percent(self):
        ds = T001Dataset()
        assert abs(ds.majority_accuracy() - 0.50) < 1e-6

    def test_chance_accuracy_fifty_percent(self):
        ds = T001Dataset()
        assert abs(ds.chance_accuracy() - 0.50) < 1e-6

    def test_class0_is_sparse(self):
        """Class 0 (p=0.25) should have mean ~0.25."""
        ds = T001Dataset()
        y = ds._y_train
        x0 = ds._X_train[y == 0]
        assert 0.15 < x0.mean() < 0.35, f"Class 0 mean: {x0.mean()}"

    def test_class1_is_dense(self):
        """Class 1 (p=0.75) should have mean ~0.75."""
        ds = T001Dataset()
        y = ds._y_train
        x1 = ds._X_train[y == 1]
        assert 0.65 < x1.mean() < 0.85, f"Class 1 mean: {x1.mean()}"

    def test_sample_returns_correct_keys(self):
        ds = T001Dataset()
        s = ds.sample("train", idx=0)
        for key in ("split", "idx", "obs", "label", "label_name", "obs_sum", "obs_mean"):
            assert key in s

    def test_d_act_is_2(self):
        cfg = T001Config()
        assert cfg.d_act == 2


# ---------------------------------------------------------------------------
# Model interface tests
# ---------------------------------------------------------------------------


class TestModelInterface:
    def test_a0_runs_on_t001_input(self):
        """A0 should accept T-001 obs shape."""
        ds = T001Dataset()
        obs, label = ds.train[0]
        obs = obs.unsqueeze(0)  # (1, 64)
        model = A0Random.build(obs_dim=64, act_dim=2)
        out = model(obs)
        assert out.shape == (1, 2)

    def test_mlp_runs_on_t001_input(self):
        ds = T001Dataset()
        obs, label = ds.train[0]
        obs = obs.unsqueeze(0)
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=32, n_layers=2)
        out = model(obs)
        assert out.shape == (1, 2)

    def test_mlp_produces_different_outputs_for_different_classes(self):
        """Model should respond differently to class-0 and class-1 patterns."""
        ds = T001Dataset()
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=32)
        model.eval()
        x0 = torch.from_numpy(ds._X_train[ds._y_train == 0][:4])
        x1 = torch.from_numpy(ds._X_train[ds._y_train == 1][:4])
        o0 = model(x0)
        o1 = model(x1)
        assert not torch.allclose(o0, o1)


# ---------------------------------------------------------------------------
# End-to-end task execution
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_evaluate_returns_correct_keys(self):
        ds = T001Dataset()
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=16, n_layers=2)
        result = evaluate(model, ds.dev)
        for key in ("accuracy", "loss", "n_samples", "n_correct"):
            assert key in result

    def test_evaluate_accuracy_in_range(self):
        ds = T001Dataset()
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=16)
        result = evaluate(model, ds.dev)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_run_task_mlp_improves_over_chance(self):
        """After 20 epochs, MLP should beat 50% on dev."""
        ds = T001Dataset()
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=64, n_layers=3, seed=42)
        result = run_task(
            model, ds.train, ds.dev, ds.test,
            n_epochs=20, lr=1e-3, batch_size=32, verbose=False,
        )
        assert result["test_acc"] > 0.55, f"Expected >55% accuracy, got {result['test_acc']:.3f}"

    def test_run_task_returns_required_keys(self):
        ds = T001Dataset()
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=16, n_layers=1)
        result = run_task(
            model, ds.train, ds.dev, ds.test,
            n_epochs=2, verbose=False,
        )
        for key in ("history", "best_dev_acc", "test_acc", "test_loss",
                    "n_params", "wall_time_s", "arch_id"):
            assert key in result

    def test_history_has_n_entries(self):
        ds = T001Dataset()
        model = A7MLP.build(obs_dim=64, act_dim=2, hidden_dim=16)
        result = run_task(model, ds.train, ds.dev, ds.test, n_epochs=5, verbose=False)
        assert len(result["history"]) == 5
