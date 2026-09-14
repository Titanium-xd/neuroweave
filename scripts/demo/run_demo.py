"""
scripts/demo/run_demo.py — MaleCNS Connectome Activity Demonstration

WHAT THIS SHOWS
---------------
A real MaleCNS v1.0 connectome subgraph (150 neurons, fetched live from
neuPrint or loaded from local Parquet cache) propagating activity through
its sparse directed signed connections.

This is NOT a biological simulation of a fruit fly.
It demonstrates how the connectome-topology-constrained architecture
propagates signals through the real MaleCNS connectivity structure.

HOW TO RUN
----------
    python scripts/demo/run_demo.py --demo

Options:
    --demo          Run with live/cached MaleCNS data and show the figure
    --save PATH     Save figure to PNG instead of showing interactively
    --steps N       Number of simulation steps (default 60)
    --stim N        Number of input neurons to stimulate (default 8)
    --seed N        Random seed (default 42)
    --synthetic     Use a synthetic ring graph (no neuPrint needed)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abb.sim.engine import ConnectomeSimulator, SimConfig

# Assumption labels displayed in the figure
_DISCLAIMER = (
    "COMPUTATIONAL DEMONSTRATION — Not a biological simulation.\n"
    "Graph topology derived from MaleCNS v1.0. Dynamics = rate model [SA-010]."
)

# ── NT color mapping ────────────────────────────────────────────────────────
_NT_COLORS = {
    "acetylcholine": "#4FC3F7",   # cyan-blue (excitatory)
    "gaba":          "#EF9A9A",   # soft red (inhibitory)
    "glutamate":     "#CE93D8",   # purple (inhibitory context)
    "dopamine":      "#FFCC02",   # yellow (modulatory)
    "serotonin":     "#A5D6A7",   # green (modulatory)
    "octopamine":    "#FFB74D",   # orange (modulatory)
    None:            "#B0BEC5",   # grey (unknown)
}


def _nt_base_color(nt: str | None) -> str:
    if nt is None:
        return _NT_COLORS[None]
    return _NT_COLORS.get(str(nt).lower(), _NT_COLORS[None])


# ── Graph loading ───────────────────────────────────────────────────────────

def _load_malecns_subgraph():
    """
    Load MaleCNS subgraph.
    Tries Parquet cache first, then fetches live from neuPrint.
    Returns (adj_coo, node_df, edge_df, source_label).
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Try cache
    cache_dir = Path("data/raw/malecns_e2e_validation")
    cache_key = "b7a787b4a89798bb"   # key written by e2e_validation.py
    n_parquet = cache_dir / f"neurons_{cache_key}.parquet"
    e_parquet = cache_dir / f"edges_{cache_key}.parquet"

    if n_parquet.exists() and e_parquet.exists():
        print("  [cache] Loading MaleCNS subgraph from local Parquet cache …")
        neurons = pd.read_parquet(n_parquet)
        edges   = pd.read_parquet(e_parquet)
        label   = "MaleCNS v1.0 (Parquet cache)"
    else:
        print("  [live]  Fetching from neuPrint (this may take ~60 s) …")
        from abb.config.schema import NeuPrintConfig
        from abb.data.neuprint_client import NeuPrintClient
        from abb.data.annotations import AnnotationEnricher
        from abb.config.schema import NTSignScheme

        client = NeuPrintClient(config=NeuPrintConfig())
        neurons = client.fetch_neuron_sample(n=300, status="Traced")
        body_ids = neurons["bodyId"].tolist()
        edges = client.fetch_adjacencies_for_ids(body_ids, min_weight=1)
        enricher = AnnotationEnricher(nt_scheme=NTSignScheme.NT_RULE, random_seed=42)
        neurons, edges, _ = enricher.enrich(neurons, edges, normalization="raw")
        label = "MaleCNS v1.0 (live neuPrint)"

    return neurons, edges, label


def _build_subgraph_adj(neurons: pd.DataFrame, edges: pd.DataFrame, max_nodes: int = 150):
    """
    Build signed COO adjacency for a subgraph of up to max_nodes neurons.
    Returns (adj_coo, sub_neurons, sub_edges).
    """
    import networkx as nx

    # Take the first max_nodes by bodyId (deterministic)
    sub = neurons.head(max_nodes).copy()
    body_ids = set(sub["bodyId"].tolist())

    sub_edges = edges[
        edges["bodyId_pre"].isin(body_ids) & edges["bodyId_post"].isin(body_ids)
    ].copy()

    bid_to_idx = {int(bid): i for i, bid in enumerate(sub["bodyId"])}
    n = len(sub)

    rows = sub_edges["bodyId_pre"].map(bid_to_idx).to_numpy(dtype=np.int32)
    cols = sub_edges["bodyId_post"].map(bid_to_idx).to_numpy(dtype=np.int32)

    if "signed_weight" in sub_edges.columns:
        vals = sub_edges["signed_weight"].to_numpy(dtype=np.float32)
    elif "weight" in sub_edges.columns:
        vals = sub_edges["weight"].to_numpy(dtype=np.float32)
    else:
        vals = np.ones(len(sub_edges), dtype=np.float32)

    adj = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    sub["_idx"] = sub["bodyId"].map(bid_to_idx)
    return adj, sub.reset_index(drop=True), sub_edges


def _synthetic_subgraph(n: int = 80, seed: int = 42):
    """
    Fallback: Erdős–Rényi synthetic graph with NT sign labeling.
    Used when --synthetic flag is given or data is unavailable.
    """
    rng = np.random.default_rng(seed)
    nt_types = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin"]
    nt_signs = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "dopamine": 0, "serotonin": 0}

    neurons = pd.DataFrame({
        "bodyId": list(range(n)),
        "predictedNt": rng.choice(nt_types, size=n),
        "predictedNtConfidence": rng.uniform(0.7, 1.0, size=n),
        "type": [f"type_{i%5}" for i in range(n)],
    })
    neurons["nt_sign"] = neurons["predictedNt"].map(nt_signs).fillna(0).astype(int)

    # ER graph ~5% density
    m = int(n * n * 0.05)
    pairs = set()
    while len(pairs) < m:
        r, c = rng.integers(0, n, size=2)
        if r != c:
            pairs.add((int(r), int(c)))
    rows, cols = zip(*pairs)
    signs = neurons["nt_sign"].to_numpy()[list(rows)]
    vals = (rng.exponential(2, size=len(rows)) * signs).astype(np.float32)
    adj = sp.coo_matrix((vals, (list(rows), list(cols))), shape=(n, n))
    return adj, neurons, "Synthetic ER graph (no neuPrint)"


# ── Visualization ───────────────────────────────────────────────────────────

def _spring_layout(adj: sp.coo_matrix, seed: int = 42) -> np.ndarray:
    """Compute spring layout positions using networkx."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(adj.shape[0]))
    coo = adj.tocoo()
    for r, c in zip(coo.row, coo.col):
        G.add_edge(int(r), int(c))
    pos = nx.spring_layout(G, seed=seed, iterations=80, k=1.2 / (adj.shape[0] ** 0.5))
    return np.array([pos[i] for i in range(adj.shape[0])])


def run_demo(
    args: argparse.Namespace,
    adj: sp.coo_matrix,
    node_df: pd.DataFrame,
    source_label: str,
) -> None:
    """Build simulator, run it, and display/save the figure."""
    n = adj.shape[0]
    seed = args.seed
    n_steps = args.steps
    n_stim = min(args.stim, n)

    # Pick stimulated neurons — prefer high-confidence excitatory ones
    rng = np.random.default_rng(seed)
    if "nt_sign" in node_df.columns:
        exc_mask = (node_df["nt_sign"] == 1).to_numpy()
        exc_idxs = np.where(exc_mask)[0]
        if len(exc_idxs) >= n_stim:
            stim_idxs = rng.choice(exc_idxs, size=n_stim, replace=False).tolist()
        else:
            stim_idxs = rng.choice(n, size=n_stim, replace=False).tolist()
    else:
        stim_idxs = rng.choice(n, size=n_stim, replace=False).tolist()

    # NT colors per node
    nt_col_map = {
        "acetylcholine": "#4FC3F7", "gaba": "#EF9A9A", "glutamate": "#CE93D8",
        "dopamine": "#FFCC02", "serotonin": "#A5D6A7", "octopamine": "#FFB74D",
    }
    if "predictedNt" in node_df.columns:
        base_colors = [
            nt_col_map.get(str(nt).lower() if pd.notna(nt) else "", "#B0BEC5")
            for nt in node_df["predictedNt"]
        ]
    else:
        base_colors = ["#B0BEC5"] * n

    # Simulator
    cfg = SimConfig(
        decay=0.45, leak=0.04,
        stim_strength=1.0, seed=seed,
        n_steps=n_steps, stim_period=15,
        init_noise=0.002,
    )
    sim = ConnectomeSimulator(adj, node_df, cfg)
    sim.reset()
    sim.run(n_steps=n_steps, stim_indices=stim_idxs, stim_period=15)
    history = sim.history          # (n_steps, N)
    final_state = sim.state

    print(f"\n  Simulation complete — {n_steps} steps, {n} neurons, {adj.nnz} edges")
    stats = sim.activity_stats()
    print(f"  Active excitatory : {stats['n_active_exc']}")
    print(f"  Active inhibitory : {stats['n_active_inh']}")
    print(f"  Silent            : {stats['n_silent']}")
    print(f"  Mean |activation| : {stats['mean_abs']:.4f}")

    # ── Layout ─────────────────────────────────────────────────────────
    print("\n  Computing spring layout …")
    t0 = time.monotonic()
    positions = _spring_layout(adj, seed=seed)
    print(f"  Layout done in {time.monotonic()-t0:.1f}s")

    # ── Figure ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10), facecolor="#0D0D1A")
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.05, right=0.97,
        top=0.90, bottom=0.08,
        wspace=0.35, hspace=0.40,
    )

    # ── Title ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.96,
        "ABB  |  MaleCNS Connectome -- Activity Propagation Demo",
        ha="center", va="top",
        fontsize=16, fontweight="bold", color="#E0E0FF",
    )
    fig.text(
        0.5, 0.925,
        _DISCLAIMER,
        ha="center", va="top",
        fontsize=8, color="#888899", style="italic",
    )

    # ── Panel 1 (top-left × 2): Graph with final activation ─────────────
    ax_graph = fig.add_subplot(gs[0, :2])
    ax_graph.set_facecolor("#0A0A18")
    ax_graph.set_title(
        f"Connectome Graph — Final Activation  ({source_label})",
        color="#CCCCDD", fontsize=10, pad=6,
    )
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    # Draw edges (sample max 1500 for speed)
    coo = adj.tocoo()
    edge_sample = min(len(coo.row), 1500)
    sample_idx = rng.choice(len(coo.row), size=edge_sample, replace=False)
    for idx in sample_idx:
        r, c = int(coo.row[idx]), int(coo.col[idx])
        xvals = [positions[r, 0], positions[c, 0]]
        yvals = [positions[r, 1], positions[c, 1]]
        color = "#3344AA" if coo.data[idx] > 0 else "#AA3333"
        ax_graph.plot(xvals, yvals, color=color, alpha=0.12, linewidth=0.3)

    # Draw nodes
    activation = final_state  # (N,)
    act_norm = (activation + 1) / 2  # map [-1,1] → [0,1]
    node_sizes = 18 + 80 * np.abs(activation)

    for i in range(n):
        a = float(act_norm[i])
        base = base_colors[i]
        # Blend base color with activation brightness
        sc = ax_graph.scatter(
            positions[i, 0], positions[i, 1],
            s=float(node_sizes[i]),
            c=[[a, a * 0.6, 1 - a]] if activation[i] >= 0 else [[1 - a, 0.2, a]],
            alpha=0.75, linewidths=0,
            zorder=3,
        )

    # Mark stimulated nodes
    stim_x = positions[stim_idxs, 0]
    stim_y = positions[stim_idxs, 1]
    ax_graph.scatter(
        stim_x, stim_y, s=120, c="#FFEE44",
        marker="*", zorder=5, label=f"Stimulated ({n_stim})",
        edgecolors="#FFFFFF", linewidths=0.5,
    )
    ax_graph.legend(loc="upper right", fontsize=7, facecolor="#1A1A2E", edgecolor="#444455")

    # ── Panel 2 (top-right): Activity heatmap ─────────────────────────
    ax_heat = fig.add_subplot(gs[0, 2])
    ax_heat.set_facecolor("#0A0A18")
    ax_heat.set_title("Activity Heatmap (neurons × steps)", color="#CCCCDD", fontsize=10, pad=6)

    # Sort neurons by NT sign for visual grouping
    if "nt_sign" in node_df.columns:
        sort_order = node_df["nt_sign"].fillna(0).argsort()[::-1].to_numpy()
    else:
        sort_order = np.arange(n)

    im = ax_heat.imshow(
        history[:, sort_order].T,
        aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest", origin="upper",
    )
    ax_heat.set_xlabel("Simulation step", color="#AAAACC", fontsize=8)
    ax_heat.set_ylabel("Neuron (sorted by NT sign)", color="#AAAACC", fontsize=8)
    ax_heat.tick_params(colors="#AAAACC", labelsize=7)
    plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02,
                 label="Activation", format="%.1f")

    # ── Panel 3 (bottom-left): Mean activity over time ─────────────────
    ax_mean = fig.add_subplot(gs[1, 0])
    ax_mean.set_facecolor("#0A0A18")
    ax_mean.set_title("Mean |Activation| Over Time", color="#CCCCDD", fontsize=10, pad=6)

    mean_act = np.abs(history).mean(axis=1)
    ax_mean.fill_between(range(n_steps), mean_act, alpha=0.3, color="#4FC3F7")
    ax_mean.plot(mean_act, color="#4FC3F7", linewidth=1.5)
    ax_mean.set_xlabel("Step", color="#AAAACC", fontsize=8)
    ax_mean.set_ylabel("Mean |x|", color="#AAAACC", fontsize=8)
    ax_mean.tick_params(colors="#AAAACC", labelsize=7)
    ax_mean.set_xlim(0, n_steps - 1)
    ax_mean.axvline(x=0, color="#FFEE44", linestyle="--", alpha=0.6, linewidth=0.8,
                    label="Stim start")
    for sp_t in range(15, n_steps, 15):
        ax_mean.axvline(x=sp_t, color="#FFEE44", linestyle="--", alpha=0.3, linewidth=0.5)
    ax_mean.legend(fontsize=7, facecolor="#1A1A2E", edgecolor="#444455")

    # ── Panel 4 (bottom-mid): Excitatory vs Inhibitory active count ────
    ax_counts = fig.add_subplot(gs[1, 1])
    ax_counts.set_facecolor("#0A0A18")
    ax_counts.set_title("Active Neurons Per Step", color="#CCCCDD", fontsize=10, pad=6)

    exc_counts = (history > 0.05).sum(axis=1)
    inh_counts = (history < -0.05).sum(axis=1)
    ax_counts.fill_between(range(n_steps), exc_counts, alpha=0.35, color="#4FC3F7", label="Excitatory")
    ax_counts.fill_between(range(n_steps), -inh_counts, alpha=0.35, color="#EF9A9A", label="Inhibitory")
    ax_counts.plot(exc_counts, color="#4FC3F7", linewidth=1.2)
    ax_counts.plot(-inh_counts, color="#EF9A9A", linewidth=1.2)
    ax_counts.axhline(y=0, color="#555566", linewidth=0.5)
    ax_counts.set_xlabel("Step", color="#AAAACC", fontsize=8)
    ax_counts.set_ylabel("# Active neurons", color="#AAAACC", fontsize=8)
    ax_counts.tick_params(colors="#AAAACC", labelsize=7)
    ax_counts.set_xlim(0, n_steps - 1)
    ax_counts.legend(fontsize=7, facecolor="#1A1A2E", edgecolor="#444455")

    # ── Panel 5 (bottom-right): Final activation bar chart ─────────────
    ax_bar = fig.add_subplot(gs[1, 2])
    ax_bar.set_facecolor("#0A0A18")
    ax_bar.set_title("Final Activation Distribution", color="#CCCCDD", fontsize=10, pad=6)

    ax_bar.hist(
        final_state[final_state > 0.01], bins=30,
        color="#4FC3F7", alpha=0.7, label="Excitatory", density=True,
    )
    ax_bar.hist(
        final_state[final_state < -0.01], bins=30,
        color="#EF9A9A", alpha=0.7, label="Inhibitory", density=True,
    )
    ax_bar.set_xlabel("Activation value", color="#AAAACC", fontsize=8)
    ax_bar.set_ylabel("Density", color="#AAAACC", fontsize=8)
    ax_bar.tick_params(colors="#AAAACC", labelsize=7)
    ax_bar.legend(fontsize=7, facecolor="#1A1A2E", edgecolor="#444455")

    # ── Stats box ──────────────────────────────────────────────────────
    stats_txt = (
        f"Neurons: {n}   Synaptic edges: {adj.nnz:,}\n"
        f"Stimulated: {n_stim}   Steps: {n_steps}\n"
        f"Active (exc/inh): {stats['n_active_exc']} / {stats['n_active_inh']}\n"
        f"Mean |act|: {stats['mean_abs']:.3f}"
    )
    fig.text(
        0.5, 0.01, stats_txt,
        ha="center", va="bottom",
        fontsize=8, color="#888899",
        bbox=dict(facecolor="#13131F", edgecolor="#333344", alpha=0.8, pad=4),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n  Figure saved → {out.resolve()}")
    else:
        print("\n  Displaying figure (close the window to exit) …")
        plt.show()

    return fig


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MaleCNS Connectome Activity Demonstration"
    )
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run the demonstration (default: on)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this PNG path instead of displaying")
    parser.add_argument("--steps", type=int, default=60,
                        help="Number of simulation steps (default 60)")
    parser.add_argument("--stim", type=int, default=8,
                        help="Number of neurons to stimulate (default 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic graph (no neuPrint data needed)")
    args = parser.parse_args()

    print()
    print("=" * 64)
    print("  ABB — MaleCNS Connectome Activity Demonstration")
    print("=" * 64)
    print(f"  Steps   : {args.steps}")
    print(f"  Stimulate: {args.stim} neurons")
    print(f"  Seed    : {args.seed}")
    print(f"  Mode    : {'synthetic' if args.synthetic else 'MaleCNS data'}")
    print()

    if args.synthetic:
        adj, node_df, source_label = _synthetic_subgraph(n=80, seed=args.seed)
        edge_df = None
    else:
        try:
            neurons, edges, source_label = _load_malecns_subgraph()
            adj, node_df, edge_df = _build_subgraph_adj(neurons, edges, max_nodes=150)
            print(f"  Source       : {source_label}")
            print(f"  Neurons      : {len(node_df)}")
            print(f"  Edges (sub)  : {adj.nnz:,}")
        except Exception as exc:
            print(f"\n  [warn] Could not load MaleCNS data: {exc}")
            print("  Falling back to synthetic graph (use --synthetic to suppress this message)")
            adj, node_df, source_label = _synthetic_subgraph(n=80, seed=args.seed)

    print()
    run_demo(args, adj, node_df, source_label)


if __name__ == "__main__":
    main()
