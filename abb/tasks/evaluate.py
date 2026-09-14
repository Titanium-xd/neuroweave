"""
abb.tasks.evaluate — Multi-seed evaluation harness for ABB tasks.

Provides standard evaluation loop over multiple seeds, aggregate reporting,
and reproducible JSON schema output for all benchmark runs.
"""

from __future__ import annotations

import time
import numpy as np
import scipy.stats as st
from typing import Callable, Any, Optional

import torch
from torch.utils.data import TensorDataset

from abb.models.base import AbstractAgent
from abb.tasks.runner import run_task


def compute_ci95(data: list[float]) -> float:
    """Compute the 95% confidence interval half-width for a list of values."""
    if len(data) < 2:
        return 0.0
    sem = st.sem(data)
    return float(sem * st.t.ppf((1 + 0.95) / 2., len(data) - 1))


def run_benchmark(
    arch_id: str,
    task_id: str,
    model_builder: Callable[[int], AbstractAgent],
    train_set: TensorDataset,
    dev_set: TensorDataset,
    test_set: TensorDataset,
    seeds: list[int] = (42, 43, 44, 45, 46),
    n_epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    frozen_params: int = 0,
) -> dict:
    """
    Run a multi-seed benchmark experiment.

    Parameters
    ----------
    arch_id : str
    task_id : str
    model_builder : Callable[[int], AbstractAgent]
        Function that takes a seed and returns an un-trained model instance.
    train_set, dev_set, test_set : TensorDataset
    seeds : list of int
    n_epochs : int
    lr : float
    batch_size : int
    device : torch.device
    frozen_params : int
        Number of frozen parameters (e.g. MaleCNS synapse priors).

    Returns
    -------
    dict
        A standard benchmark result dictionary.
    """
    device = device or torch.device("cpu")
    
    results = []
    
    print(f"\nEvaluating {arch_id} on {task_id} across {len(seeds)} seeds...")
    
    # Run loop
    for seed in seeds:
        print(f"  Running seed {seed} ...", end="", flush=True)
        model = model_builder(seed)
        n_params = model.param_count()["trainable"]
        
        t0 = time.monotonic()
        res = run_task(
            model, train_set, dev_set, test_set,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size,
            device=device, verbose=False
        )
        total_time = time.monotonic() - t0
        
        # Calculate AULC (Area Under Learning Curve) for dev accuracy
        # history is a list of dicts: {"epoch": ..., "dev_acc": ...}
        history = res.get("history", [])
        dev_accs = [row["dev_acc"] for row in history]
        aulc = float(np.trapezoid(dev_accs)) if len(dev_accs) > 1 else 0.0
        
        seed_result = {
            "seed": seed,
            "test_acc": res["test_acc"],
            "best_dev_acc": res["best_dev_acc"],
            "aulc": aulc,
            "wall_time_s": total_time,
            "history": history
        }
        results.append(seed_result)
        print(f" test_acc={res['test_acc']*100:.1f}%, time={total_time:.1f}s")
    
    # Aggregation
    test_accs = [r["test_acc"] for r in results]
    times = [r["wall_time_s"] for r in results]
    
    aggregate = {
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_acc_ci95": compute_ci95(test_accs),
        "wall_time_s_mean": float(np.mean(times)),
    }
    
    sample_budget = len(train_set) * n_epochs
    
    return {
        "architecture": arch_id,
        "task": task_id,
        "seeds": list(seeds),
        "aggregate": aggregate,
        "results_per_seed": results,
        "metadata": {
            "trainable_parameters": n_params,
            "frozen_parameters": frozen_params,
            "sample_budget": sample_budget,
            "compute_budget_epochs": n_epochs,
            "total_wall_clock_s": sum(times),
            "device": str(device),
        }
    }

