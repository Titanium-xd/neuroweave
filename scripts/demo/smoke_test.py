"""Smoke test for simulator with real MaleCNS data."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

from abb.sim.engine import ConnectomeSimulator, SimConfig

n_paths = list(Path("data/raw/malecns_e2e_validation").glob("neurons_*.parquet"))
e_paths = list(Path("data/raw/malecns_e2e_validation").glob("edges_*.parquet"))

if not n_paths or not e_paths:
    print("No cache found — skipping smoke test")
    sys.exit(0)

neurons = pd.read_parquet(n_paths[0])
edges   = pd.read_parquet(e_paths[0])
print(f"Loaded {len(neurons)} neurons, {len(edges)} edges from cache")

sub = neurons.head(150)
body_ids = set(sub["bodyId"])
sub_edges = edges[
    edges["bodyId_pre"].isin(body_ids) & edges["bodyId_post"].isin(body_ids)
]
bid_to_idx = {int(b): i for i, b in enumerate(sub["bodyId"])}
rows = sub_edges["bodyId_pre"].map(bid_to_idx).to_numpy(dtype=np.int32)
cols = sub_edges["bodyId_post"].map(bid_to_idx).to_numpy(dtype=np.int32)
col = "signed_weight" if "signed_weight" in sub_edges.columns else "weight"
vals = sub_edges[col].to_numpy(dtype=np.float32)
adj = sp.coo_matrix((vals, (rows, cols)), shape=(150, 150))

cfg = SimConfig(decay=0.45, leak=0.04, seed=42, n_steps=30, init_noise=0.002)
sim = ConnectomeSimulator(adj, sub.reset_index(drop=True), cfg)
sim.reset()
sim.run(n_steps=30, stim_indices=list(range(8)), stim_period=15)

stats = sim.activity_stats()
h = sim.history

print(f"Simulation OK   | stats={stats}")
print(f"History shape   : {h.shape}  (should be (30, 150))")
print(f"No dense alloc  : nnz={sim.nnz}  (N^2=22500)")
assert h.shape == (30, 150), "Wrong history shape"
assert sim.nnz == adj.tocsr().nnz, "nnz mismatch"
assert stats["n_active_exc"] + stats["n_active_inh"] + stats["n_silent"] == 150
print("ALL SMOKE TESTS PASS ✓")
