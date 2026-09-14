"""
scripts/tasks/run_benchmark.py — The single command to run an ABB benchmark experiment.

Runs a selected architecture over multiple seeds on a specified task, using the
standard evaluation harness to record all required metrics and schema fields.

USAGE
-----
    python scripts/tasks/run_benchmark.py --arch A1-BIO --task T-001 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abb.tasks.t001 import T001Config, T001Dataset
from abb.tasks.evaluate import run_benchmark
from abb.models.base import AgentConfig
from abb.models.baselines import A0Random, A7MLP, A8RNN
from abb.models.bio_variants import A1Bio


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
        raise FileNotFoundError("MaleCNS cache not found. Run smoke test first.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, help="Architecture ID (e.g. A7-MLP, A1-BIO)")
    parser.add_argument("--task", type=str, default="T-001", help="Task ID (default: T-001)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save", type=str, default="artifacts/benchmark_result.json")
    args = parser.parse_args()

    print("=" * 64)
    print(f"  ABB BENCHMARK RUNNER")
    print("=" * 64)

    if args.task != "T-001":
        raise NotImplementedError(f"Task {args.task} not implemented yet.")

    cfg = T001Config()
    ds = T001Dataset(cfg)
    
    seeds = [42 + i for i in range(args.seeds)]
    frozen_params = 0
    
    adj = None
    if "BIO" in args.arch:
        print("\nLoading MaleCNS reference graph...")
        adj, n_nodes = _load_malecns_adj(max_nodes=150)
        frozen_params = int(adj.nnz) * 2  # 2 layers

    def model_builder(seed: int):
        if args.arch == "A0":
            return A0Random.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, seed=seed)
        elif args.arch == "A7-MLP":
            return A7MLP.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=3, seed=seed)
        elif args.arch == "A8-LSTM":
            return A8RNN.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=2, cell_type="lstm", seed=seed)
        elif args.arch == "A8-GRU":
            return A8RNN.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=2, cell_type="gru", seed=seed)
        elif args.arch == "A1-BIO":
            bio_cfg = AgentConfig("A1-BIO", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=seed, assumption_ids=())
            return A1Bio(bio_cfg, adj)
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

    result = run_benchmark(
        arch_id=args.arch,
        task_id=args.task,
        model_builder=model_builder,
        train_set=ds.train,
        dev_set=ds.dev,
        test_set=ds.test,
        seeds=seeds,
        n_epochs=args.epochs,
        frozen_params=frozen_params
    )

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    # Print summary table
    agg = result["aggregate"]
    meta = result["metadata"]
    
    print("\n" + "=" * 64)
    print("  MULTI-SEED EVALUATION SUMMARY")
    print("=" * 64)
    print(f"  Architecture : {result['architecture']}")
    print(f"  Task         : {result['task']}")
    print(f"  Seeds        : {len(seeds)} ({seeds})")
    print("  " + "─" * 62)
    print(f"  Test Acc     : {agg['test_acc_mean']*100:.1f}% ± {agg['test_acc_std']*100:.1f}%")
    print(f"  95% CI       : ±{agg['test_acc_ci95']*100:.2f}%")
    print("  " + "─" * 62)
    print(f"  Params       : {meta['trainable_parameters']:,} trainable")
    if meta['frozen_parameters'] > 0:
        print(f"               : {meta['frozen_parameters']:,} frozen")
    print(f"  Mean Time    : {agg['wall_time_s_mean']:.1f}s / seed")
    print("=" * 64)
    print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
