"""
scripts/tasks/run_t001.py — T-001: Binary Pattern Discrimination end-to-end demo.

Runs T-001 on three architectures:
  A0        : random policy (lower bound)
  A7-MLP    : standard dense feedforward
  A1-BIO    : MaleCNS connectome-topology-constrained architecture

Prints one worked example (input → output → prediction) and a results table.

USAGE
-----
    python scripts/tasks/run_t001.py
    python scripts/tasks/run_t001.py --epochs 60 --no-bio
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abb.tasks.t001 import T001Config, T001Dataset
from abb.tasks.runner import evaluate, run_task
from abb.models.base import AgentConfig
from abb.models.baselines import A0Random, A7MLP
from abb.models.bio_variants import A1Bio


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------


def _load_malecns_adj(max_nodes: int = 150):
    """Load MaleCNS signed adjacency from Parquet cache or synthetic fallback."""
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
        return adj, n, f"MaleCNS v1.0 ({n} neurons, {adj.nnz:,} edges)"
    else:
        # Synthetic fallback
        rng = np.random.default_rng(42)
        n = 80
        m = int(n * n * 0.06)
        edges_set = set()
        while len(edges_set) < m:
            r, c = rng.integers(0, n, size=2)
            if r != c:
                edges_set.add((int(r), int(c)))
        rows_s, cols_s = zip(*edges_set)
        vals_s = rng.choice([-1.0, 1.0], size=m).astype(np.float32) * rng.exponential(2, m).astype(np.float32)
        adj = sp.coo_matrix((vals_s, (list(rows_s), list(cols_s))), shape=(n, n))
        return adj, n, f"Synthetic ER ({n} neurons, {adj.nnz:,} edges)"


# ---------------------------------------------------------------------------
# Worked example printer
# ---------------------------------------------------------------------------


def _print_example(ds: T001Dataset, model: torch.nn.Module, label: str) -> None:
    """Show one worked example: input → model → prediction."""
    print(f"\n  {'─'*60}")
    print(f"  Worked Example — {label}")
    print(f"  {'─'*60}")

    for class_idx in [0, 1]:
        y = ds._y_train
        idx = int(np.where(y == class_idx)[0][0])
        obs = torch.from_numpy(ds._X_train[idx]).unsqueeze(0)  # (1, 64)
        model.eval()
        with torch.no_grad():
            logits = model(obs)               # (1, 2)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(dim=1).item()

        class_name = "SPARSE (class 0)" if class_idx == 0 else "DENSE  (class 1)"
        pred_name  = "SPARSE (class 0)" if pred == 0 else "DENSE  (class 1)"
        correct    = "✓" if pred == class_idx else "✗"

        # Visualise observation as a mini bar
        bits = ds._X_train[idx]
        bar = "".join("█" if b > 0.5 else "░" for b in bits[:32]) + "…"

        print(f"\n  Input : {bar}")
        print(f"  Pattern: {class_name}  (mean={bits.mean():.2f})")
        print(f"  Logits : [{logits[0,0]:.3f}, {logits[0,1]:.3f}]")
        print(f"  Prob   : [p0={probs[0,0]:.3f}, p1={probs[0,1]:.3f}]")
        print(f"  Pred   : {pred_name}  {correct}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40,
                        help="Training epochs (default 40)")
    parser.add_argument("--no-bio", action="store_true",
                        help="Skip A1-BIO (faster, no graph needed)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save JSON results to this path")
    args = parser.parse_args()

    print()
    print("=" * 64)
    print("  ABB — T-001: Binary Pattern Discrimination")
    print("=" * 64)

    # Task
    cfg = T001Config()
    ds  = T001Dataset(cfg)
    print(f"\n  Task            : {cfg.task_id} — {cfg.description}")
    print(f"  D_obs           : {cfg.d_obs}")
    print(f"  Train/Dev/Test  : {cfg.n_train}/{cfg.n_dev}/{cfg.n_test}")
    print(f"  Class balance   : 50/50 (Bernoulli p_low={cfg.p_low}, p_high={cfg.p_high})")
    print(f"  Majority acc    : {ds.majority_accuracy()*100:.1f}%  (trivial baseline)")
    print(f"  Chance acc      : {ds.chance_accuracy()*100:.1f}%")

    results_all = {}
    device = torch.device("cpu")

    # ── A0 Random (baseline, no training) ─────────────────────────────
    print("\n" + "─" * 64)
    print("  [A0] Random policy baseline (no training)")
    a0 = A0Random.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act)
    a0_test = evaluate(a0, ds.test)
    results_all["A0"] = {
        "test_acc": a0_test["accuracy"],
        "n_params": 0,
        "epochs": 0,
        "wall_time_s": 0.0,
    }
    print(f"  Test accuracy : {a0_test['accuracy']*100:.1f}%  (expect ~50%)")

    # ── A7-MLP ────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print(f"  [A7-MLP] Training for {args.epochs} epochs …")
    mlp = A7MLP.build(obs_dim=cfg.d_obs, act_dim=cfg.d_act, hidden_dim=128, n_layers=3, seed=42)
    print(f"  Parameters: {mlp.param_count()['trainable']:,}")
    mlp_result = run_task(
        mlp, ds.train, ds.dev, ds.test,
        n_epochs=args.epochs, lr=1e-3, batch_size=32,
        verbose=True, log_every=10,
    )
    results_all["A7-MLP"] = {
        "test_acc": mlp_result["test_acc"],
        "best_dev_acc": mlp_result["best_dev_acc"],
        "n_params": mlp_result["n_params"],
        "epochs": args.epochs,
        "wall_time_s": mlp_result["wall_time_s"],
    }
    print(f"  Test accuracy : {mlp_result['test_acc']*100:.1f}%")
    _print_example(ds, mlp, "A7-MLP")

    # ── A1-BIO ────────────────────────────────────────────────────────
    if not args.no_bio:
        print("\n" + "─" * 64)
        print("  [A1-BIO] Loading MaleCNS graph …")
        adj, n_nodes, graph_label = _load_malecns_adj(max_nodes=150)
        print(f"  Graph: {graph_label}")

        bio_cfg = AgentConfig(
            arch_id="A1-BIO",
            obs_dim=cfg.d_obs,
            act_dim=cfg.d_act,
            n_layers=2,
            seed=42,
            assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
        )
        bio = A1Bio(bio_cfg, adj)
        print(f"  Parameters: {bio.param_count()['trainable']:,}  "
              f"(N={n_nodes}, nnz={adj.nnz:,})")

        print(f"  Training for {args.epochs} epochs …")
        bio_result = run_task(
            bio, ds.train, ds.dev, ds.test,
            n_epochs=args.epochs, lr=1e-3, batch_size=32,
            verbose=True, log_every=10,
        )
        results_all["A1-BIO"] = {
            "test_acc": bio_result["test_acc"],
            "best_dev_acc": bio_result["best_dev_acc"],
            "n_params": bio_result["n_params"],
            "n_nodes": n_nodes,
            "adj_nnz": int(adj.nnz),
            "epochs": args.epochs,
            "wall_time_s": bio_result["wall_time_s"],
            "graph_label": graph_label,
        }
        print(f"  Test accuracy : {bio_result['test_acc']*100:.1f}%")
        _print_example(ds, bio, "A1-BIO")

    # ── Results table ─────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  T-001 RESULTS SUMMARY")
    print("=" * 64)
    print(f"  {'Architecture':<20}  {'Test Acc':>10}  {'Params':>10}  {'Time(s)':>9}")
    print("  " + "─" * 55)
    print(f"  {'Majority/Chance':<20}  {'50.0%':>10}  {'0':>10}  {'—':>9}")
    for arch_id, r in results_all.items():
        acc_pct = f"{r['test_acc']*100:.1f}%"
        params  = f"{r['n_params']:,}" if r['n_params'] else "0"
        t       = f"{r['wall_time_s']:.1f}" if r['wall_time_s'] else "0.0"
        print(f"  {arch_id:<20}  {acc_pct:>10}  {params:>10}  {t:>9}")
    print("=" * 64)

    print("\n  T-001 INTERPRETATION:")
    print("  - Majority/Chance baseline: 50.0% (balanced classes)")
    print("  - A0 random: should be ~50% (no learning)")
    print("  - A7-MLP: should converge well above 90% (has full D_obs access)")
    print("  - A1-BIO: tests whether MaleCNS topology helps/hurts on this task")
    print()
    print("  NOTE: This task is designed to be solvable from the raw activations.")
    print("  A7-MLP is expected to outperform A1-BIO because it has direct access")
    print("  to all 64 input dimensions without going through sparse graph routing.")
    print("  The advantage of A1-BIO (if any) will be more visible in tasks that")
    print("  reward network integration structure — not implemented yet (Prompt 5).")

    if args.save:
        out = {
            "task": cfg.to_dict(),
            "results": results_all,
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {args.save}")


if __name__ == "__main__":
    main()
