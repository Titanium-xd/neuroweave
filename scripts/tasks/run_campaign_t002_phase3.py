"""
scripts/tasks/run_campaign_t002_phase3.py — Phase 3 ABB experiment campaign.
Runs 5 architectures on T-002 (Temporal Sequence Memory) for 5 seeds each,
using the SA-010 biological temporal dynamics for A1 models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abb.tasks.t002 import T002Config, T002Dataset
from abb.tasks.evaluate import run_benchmark
from abb.models.base import AgentConfig
from abb.models.baselines import A0Random, A7MLP, A8RNN
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="artifacts/campaign_3")
    args = parser.parse_args()
    
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  ABB PHASE 3 CAMPAIGN (T-002 with Biological Dynamics)")
    print("=" * 64)

    cfg = T002Config()
    ds = T002Dataset(cfg)
    seeds = [42 + i for i in range(args.seeds)]
    
    print("Loading MaleCNS reference graph...")
    ref_adj, n_nodes = _load_malecns_adj(max_nodes=150)
    
    architectures = [
        "A0 Random", "A1-BIO (SA-010)", "A1-FROZEN (SA-010)", "A7-MLP", "A8-LSTM"
    ]
    
    all_results = {}
    
    def get_builder(arch: str):
        frozen_p = 0
        adj_used = None
        
        def builder(seed: int):
            if arch == "A0 Random":
                return A0Random.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, seed=seed)
            elif arch == "A7-MLP":
                return A7MLP.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=3, seed=seed)
            elif arch == "A8-LSTM":
                return A8RNN.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=2, cell_type="lstm", seed=seed)
            elif arch == "A1-BIO (SA-010)":
                c = AgentConfig(
                    arch_id="A1-BIO-SA010", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=seed, 
                    assumption_ids=("SA-010",), extra={"use_sa010": True, "decay": 0.15, "leak": 0.08}
                )
                return A1Bio(c, ref_adj)
            elif arch == "A1-FROZEN (SA-010)":
                c = AgentConfig(
                    arch_id="A1-FROZEN-SA010", obs_dim=cfg.d_obs, act_dim=cfg.d_act, n_layers=2, seed=seed, 
                    assumption_ids=("SA-010",), extra={"use_sa010": True, "decay": 0.15, "leak": 0.08}
                )
                return A1Frozen(c, ref_adj)
            else:
                raise ValueError(f"Unknown arch {arch}")
                
        if "A1" in arch:
            adj_used = ref_adj
            if "FROZEN" in arch:
                frozen_p = int(adj_used.nnz) * 2 + 150 * 2  # weights + biases (2 layers)
            else:
                frozen_p = 0
        
        return builder, frozen_p, adj_used

    for arch in architectures:
        builder, frozen_p, adj_used = get_builder(arch)
        res = run_benchmark(
            arch_id=arch,
            task_id="T-002",
            model_builder=builder,
            train_set=ds.train,
            dev_set=ds.dev,
            test_set=ds.test,
            seeds=seeds,
            n_epochs=args.epochs,
            frozen_params=frozen_p
        )
        if adj_used is not None:
            res["metadata"]["graph_nodes"] = adj_used.shape[0]
            res["metadata"]["graph_edges"] = adj_used.nnz
        
        all_results[arch] = res
        
        with open(out_dir / f"{arch.replace(' ', '_').replace('_(SA-010)', '')}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("  T-002 PHASE 3 EXPERIMENT CAMPAIGN SUMMARY")
    print("=" * 80)
    print(f"  {'Architecture':<22}  {'Test Acc':>15}  {'Params (T/F)':>15}  {'Edges':>10}  {'Time':>8}")
    print("  " + "─" * 72)
    for arch in architectures:
        r = all_results[arch]
        agg = r["aggregate"]
        meta = r["metadata"]
        
        acc = f"{agg['test_acc_mean']*100:.1f}% ± {agg['test_acc_ci95']*100:.1f}%"
        p_t = meta["trainable_parameters"]
        p_f = meta["frozen_parameters"]
        p_str = f"{p_t} / {p_f}"
        edges = meta.get("graph_edges", "-")
        t = f"{agg['wall_time_s_mean']:.1f}s"
        
        print(f"  {arch:<22}  {acc:>15}  {p_str:>15}  {edges:>10}  {t:>8}")
    print("=" * 80)
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        means = [all_results[a]["aggregate"]["test_acc_mean"] * 100 for a in architectures]
        cis = [all_results[a]["aggregate"]["test_acc_ci95"] * 100 for a in architectures]
        
        x = np.arange(len(architectures))
        
        colors = []
        for a in architectures:
            if "A1" in a: colors.append("#2ca02c")
            elif "A0" in a: colors.append("#7f7f7f")
            else: colors.append("#1f77b4")
            
        bars = ax.bar(x, means, yerr=cis, capsize=5, color=colors, alpha=0.8)
        
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("T-002 Phase 3: Bio Temporal Dynamics (5 Seeds, 95% CI)")
        ax.set_xticks(x)
        ax.set_xticklabels(architectures, rotation=30, ha="right")
        ax.set_ylim(40, 105)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Chance (50%)')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.8, label='Bio-Constrained (SA-010)'),
            Patch(facecolor='#1f77b4', alpha=0.8, label='Dense ML Baselines'),
            Patch(facecolor='#7f7f7f', alpha=0.8, label='Random (A0)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plot_path = out_dir / "campaign_3_results.png"
        plt.savefig(plot_path, dpi=300)
        print(f"\nPlot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()
