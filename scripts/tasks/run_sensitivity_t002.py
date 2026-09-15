"""
scripts/tasks/run_sensitivity_t002.py — SA-010 Parameter Sensitivity Analysis on T-002
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

from abb.tasks.t002 import T002Config, T002Dataset
from abb.tasks.evaluate import run_benchmark
from abb.models.base import AgentConfig
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
        raise FileNotFoundError("MaleCNS cache not found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="artifacts/sensitivity_t002")
    args = parser.parse_args()
    
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  SA-010 SENSITIVITY ANALYSIS (A1-BIO on T-002)")
    print("=" * 64)

    cfg = T002Config()
    ds = T002Dataset(cfg)
    seeds = [42 + i for i in range(args.seeds)]
    
    print("Loading MaleCNS reference graph...")
    ref_adj, n_nodes = _load_malecns_adj(max_nodes=150)
    
    params = [
        (0.05, 0.02),
        (0.10, 0.05),
        (0.15, 0.08),  # default
        (0.20, 0.10),
        (0.30, 0.15),
    ]
    
    all_results = {}
    
    for decay, leak in params:
        name = f"A1-BIO (d={decay}, l={leak})"
        print(f"\nEvaluating: {name}")
        
        def builder(seed: int):
            c = AgentConfig(
                arch_id=f"A1-BIO_d{decay}_l{leak}", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=seed, 
                assumption_ids=("SA-010",), extra={"use_sa010": True, "decay": decay, "leak": leak}
            )
            return A1Bio(c, ref_adj)

        res = run_benchmark(
            arch_id=name,
            task_id="T-002",
            model_builder=builder,
            train_set=ds.train,
            dev_set=ds.dev,
            test_set=ds.test,
            seeds=seeds,
            n_epochs=args.epochs,
            frozen_params=0
        )
        
        all_results[name] = res
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(",", "")
        with open(out_dir / f"{safe_name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Print summary table
    print("\n" + "=" * 64)
    print("  SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 64)
    print(f"  {'Decay / Leak':<20}  {'Test Acc':>15}  {'Time (s)':>10}")
    print("  " + "─" * 56)
    for decay, leak in params:
        name = f"A1-BIO (d={decay}, l={leak})"
        r = all_results[name]
        agg = r["aggregate"]
        
        acc = f"{agg['test_acc_mean']*100:.1f}% ± {agg['test_acc_std']*100:.1f}%"
        t = f"{agg['wall_time_s_mean']:.1f}"
        
        print(f"  {f'{decay:.2f} / {leak:.2f}':<20}  {acc:>15}  {t:>10}")
    print("=" * 64)
    
if __name__ == "__main__":
    main()
