"""
scripts/tasks/run_confirmation_t002.py — Final T-002 confirmation experiment.

Uses the candidate SA-010 configuration identified from the exploratory
sensitivity sweep (decay=0.05, leak=0.02).

Compares:
  - A1-BIO (SA-010 best: 0.05/0.02)
  - A1-FROZEN (SA-010 best: 0.05/0.02)
  - A1-BIO (SA-010 default: 0.15/0.08)
  - A8-LSTM
  - A0 Random

Reports both DEV and TEST performance separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abb.tasks.t002 import T002Config, T002Dataset
from abb.tasks.evaluate import run_benchmark, compute_ci95
from abb.models.base import AgentConfig
from abb.models.baselines import A0Random, A8RNN
from abb.models.bio_variants import A1Bio, A1Frozen


def _load_malecns_adj(max_nodes: int = 150):
    n_paths = list(Path("data/raw/malecns_e2e_validation").glob("neurons_*.parquet"))
    e_paths = list(Path("data/raw/malecns_e2e_validation").glob("edges_*.parquet"))

    if n_paths and e_paths:
        neurons = pd.read_parquet(n_paths[0]).head(max_nodes)
        all_edges = pd.read_parquet(e_paths[0])
        body_ids = set(neurons["bodyId"])
        edges = all_edges[
            all_edges["bodyId_pre"].isin(body_ids) & all_edges["bodyId_post"].isin(body_ids)
        ]
        bid_to_idx = {int(b): i for i, b in enumerate(neurons["bodyId"])}
        rows = edges["bodyId_pre"].map(bid_to_idx).to_numpy(dtype=np.int32)
        cols = edges["bodyId_post"].map(bid_to_idx).to_numpy(dtype=np.int32)
        col = "signed_weight" if "signed_weight" in edges.columns else "weight"
        vals = edges[col].to_numpy(dtype=np.float32)
        n = len(neurons)
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
        adj.sum_duplicates()
        return adj, n
    else:
        raise FileNotFoundError("MaleCNS cache not found.")


def main():
    out_dir = Path("artifacts/confirmation_t002")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  T-002 FINAL CONFIRMATION EXPERIMENT")
    print("  Candidate config: decay=0.05, leak=0.02 (from sensitivity sweep)")
    print("=" * 68)

    cfg = T002Config()
    ds = T002Dataset(cfg)
    seeds = [42, 43, 44, 45, 46]

    print("Loading MaleCNS reference graph...")
    ref_adj, n_nodes = _load_malecns_adj(max_nodes=150)

    # --- Architecture definitions ---
    architectures = [
        ("A0-Random",             False),
        ("A1-BIO (0.05/0.02)",    False),
        ("A1-FROZEN (0.05/0.02)", True),
        ("A1-BIO (0.15/0.08)",    False),
        ("A8-LSTM",               False),
    ]

    def get_builder(name: str, frozen: bool):
        if name == "A0-Random":
            return lambda seed: A0Random.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, seed=seed), 0
        elif name == "A1-BIO (0.05/0.02)":
            def build(seed):
                c = AgentConfig(arch_id="A1-BIO-SA010-best", obs_dim=cfg.d_obs, act_dim=cfg.d_act,
                                n_layers=2, seed=seed, assumption_ids=("SA-010",),
                                extra={"use_sa010": True, "decay": 0.05, "leak": 0.02})
                return A1Bio(c, ref_adj)
            return build, 0
        elif name == "A1-FROZEN (0.05/0.02)":
            def build(seed):
                c = AgentConfig(arch_id="A1-FROZEN-SA010-best", obs_dim=cfg.d_obs, act_dim=cfg.d_act,
                                n_layers=2, seed=seed, assumption_ids=("SA-010",),
                                extra={"use_sa010": True, "decay": 0.05, "leak": 0.02})
                return A1Frozen(c, ref_adj)
            return build, int(ref_adj.nnz) * 2 + n_nodes * 2
        elif name == "A1-BIO (0.15/0.08)":
            def build(seed):
                c = AgentConfig(arch_id="A1-BIO-SA010-default", obs_dim=cfg.d_obs, act_dim=cfg.d_act,
                                n_layers=2, seed=seed, assumption_ids=("SA-010",),
                                extra={"use_sa010": True, "decay": 0.15, "leak": 0.08})
                return A1Bio(c, ref_adj)
            return build, 0
        elif name == "A8-LSTM":
            return lambda seed: A8RNN.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=2, cell_type="lstm", seed=seed), 0
        else:
            raise ValueError(name)

    all_results = {}

    for name, frozen in architectures:
        builder, frozen_p = get_builder(name, frozen)
        res = run_benchmark(
            arch_id=name, task_id="T-002",
            model_builder=builder,
            train_set=ds.train, dev_set=ds.dev, test_set=ds.test,
            seeds=seeds, n_epochs=30, frozen_params=frozen_p,
        )
        if "A1" in name:
            res["metadata"]["graph_nodes"] = n_nodes
            res["metadata"]["graph_edges"] = int(ref_adj.nnz)

        all_results[name] = res
        safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        with open(out_dir / f"{safe}.json", "w") as f:
            json.dump(res, f, indent=2)

    # --- Print table ---
    print("\n" + "=" * 90)
    print("  T-002 FINAL CONFIRMATION RESULTS  (5 seeds, 95% CI, 30 epochs)")
    print("=" * 90)
    print(f"  {'Architecture':<26}  {'DEV Acc':>16}  {'TEST Acc':>16}  {'Train-P':>8}  {'Frozen-P':>9}  {'Time':>7}")
    print("  " + "─" * 84)

    for name, _ in architectures:
        r = all_results[name]
        agg = r["aggregate"]
        meta = r["metadata"]

        dev_accs = [sr["best_dev_acc"] for sr in r["results_per_seed"]]
        dev_mean = np.mean(dev_accs) * 100
        dev_ci   = compute_ci95(dev_accs) * 100

        test_mean = agg["test_acc_mean"] * 100
        test_ci   = agg["test_acc_ci95"] * 100

        p_t  = meta["trainable_parameters"]
        p_f  = meta["frozen_parameters"]
        t    = agg["wall_time_s_mean"]

        dev_str  = f"{dev_mean:.1f}% ± {dev_ci:.1f}%"
        test_str = f"{test_mean:.1f}% ± {test_ci:.1f}%"

        print(f"  {name:<26}  {dev_str:>16}  {test_str:>16}  {p_t:>8,}  {p_f:>9,}  {t:>5.1f}s")

    print("=" * 90)
    print()
    print("  NOTE: Candidate config (0.05/0.02) was selected from exploratory")
    print("  sensitivity sweep. This table is the held-out test evaluation.")


if __name__ == "__main__":
    main()
