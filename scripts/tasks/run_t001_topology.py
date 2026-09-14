"""
scripts/tasks/run_t001_topology.py — T-001 Topology Controls Comparison.

Compares A1-BIO against topology ablations:
- A3-ER (Erdős–Rényi: same N, M, random wiring)
- A3-CONFIG (Configuration model: same N, M, exact in/out degree sequence)
- A3-DENSE (Fully connected: same N, N^2 edges)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abb.tasks.t001 import T001Config, T001Dataset
from abb.tasks.runner import run_task
from abb.models.base import AgentConfig
from abb.models.bio_variants import A1Bio
from abb.models.topology import _er_rewire, _config_rewire, A3ER, A3Config, A3Dense


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
        return adj, n
    else:
        raise FileNotFoundError("MaleCNS cache not found. Run smoke test first.")


def verify_degree_sequence(orig_adj: sp.coo_matrix, new_adj: sp.coo_matrix, name: str) -> None:
    orig_csr = orig_adj.tocsr()
    new_csr = new_adj.tocsr()
    orig_out = orig_csr.getnnz(axis=1)
    new_out = new_csr.getnnz(axis=1)
    orig_in = orig_csr.getnnz(axis=0)
    new_in = new_csr.getnnz(axis=0)
    
    out_match = np.array_equal(orig_out, new_out)
    in_match = np.array_equal(orig_in, new_in)
    
    print(f"  [{name}] Graph statistics:")
    print(f"    - Nodes: {new_adj.shape[0]}")
    print(f"    - Edges: {new_adj.nnz:,} (Orig: {orig_adj.nnz:,})")
    print(f"    - Out-degree sequence preserved: {out_match}")
    print(f"    - In-degree sequence preserved: {in_match}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save", type=str, default="artifacts/t001_topology_results.json")
    args = parser.parse_args()

    print("=" * 64)
    print("  ABB — Topology Controls (T-001)")
    print("=" * 64)

    cfg = T001Config()
    ds = T001Dataset(cfg)
    
    print("\nLoading reference MaleCNS subgraph...")
    ref_adj, n_nodes = _load_malecns_adj(max_nodes=150)
    
    # Generate graphs
    print("\nGenerating topology controls...")
    er_adj = _er_rewire(ref_adj, seed=42)
    config_adj = _config_rewire(ref_adj, seed=42)
    
    verify_degree_sequence(ref_adj, ref_adj, "A1-BIO (Reference)")
    verify_degree_sequence(ref_adj, er_adj, "A3-ER")
    verify_degree_sequence(ref_adj, config_adj, "A3-CONFIG")
    
    results = {}
    
    # Helper to run a model
    def run_model(model, arch_id, adj=None):
        print("\n" + "─" * 64)
        print(f"  [{arch_id}] Training for {args.epochs} epochs …")
        print(f"  Trainable Params: {model.param_count()['trainable']:,}")
        
        res = run_task(model, ds.train, ds.dev, ds.test,
                       n_epochs=args.epochs, lr=1e-3, batch_size=32,
                       verbose=True, log_every=10)
        
        edges = adj.nnz if adj is not None else n_nodes * n_nodes
        results[arch_id] = {
            "test_acc": res["test_acc"],
            "best_dev_acc": res["best_dev_acc"],
            "n_params": res["n_params"],
            "nodes": n_nodes,
            "edges": edges,
            "wall_time_s": res["wall_time_s"],
        }
        print(f"  Test accuracy : {res['test_acc']*100:.1f}%")

    # A1-BIO
    bio_cfg = AgentConfig("A1-BIO", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=42, assumption_ids=())
    bio_model = A1Bio(bio_cfg, ref_adj)
    run_model(bio_model, "A1-BIO", ref_adj)

    # A3-ER
    er_cfg = AgentConfig("A3-ER", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=42, assumption_ids=())
    er_model = A3ER(er_cfg, er_adj)
    run_model(er_model, "A3-ER", er_adj)
    
    # A3-CONFIG
    config_cfg = AgentConfig("A3-CONFIG", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=42, assumption_ids=())
    config_model = A3Config(config_cfg, config_adj)
    run_model(config_model, "A3-CONFIG", config_adj)
    
    # A3-DENSE
    dense_cfg = AgentConfig("A3-DENSE", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=42, assumption_ids=())
    dense_model = A3Dense(dense_cfg, n_nodes=n_nodes)
    run_model(dense_model, "A3-DENSE", None)
    
    # Results Table
    print("\n" + "=" * 70)
    print("  TOPOLOGY COMPARISON RESULTS (T-001)")
    print("=" * 70)
    print(f"  {'Architecture':<15}  {'Test Acc':>10}  {'Params':>10}  {'Edges':>10}  {'Time(s)':>8}")
    print("  " + "─" * 62)
    for arch_id in ["A1-BIO", "A3-ER", "A3-CONFIG", "A3-DENSE"]:
        r = results[arch_id]
        acc = f"{r['test_acc']*100:.1f}%"
        p = f"{r['n_params']:,}"
        e = f"{r['edges']:,}"
        t = f"{r['wall_time_s']:.1f}"
        print(f"  {arch_id:<15}  {acc:>10}  {p:>10}  {e:>10}  {t:>8}")
    print("=" * 70)
    
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
