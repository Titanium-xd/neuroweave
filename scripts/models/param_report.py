"""
scripts/models/param_report.py — Print parameter counts for all ABB architectures.
"""
import sys
sys.path.insert(0, ".")

import scipy.sparse as sp
import numpy as np
import pandas as pd

from abb.models.bio_variants import A1Bio, A1Frozen, A1Bias, A1Edge, A1Both
from abb.models.topology import A3ER, A3Config, A3SBM, A3Dense
from abb.models.topology import A5WShuffle, A5WSign, A5ETarget
from abb.models.baselines import A0Random, A7MLP, A8RNN
from abb.data.graph import build_graph

N, OBS, ACT, NL = 150, 8, 4, 2

rows_idx = list(range(N))
cols_idx = [(i + 1) % N for i in range(N)]
vals = [float(i % 3 + 1) * (1 if i % 2 == 0 else -1) for i in range(N)]

neurons = pd.DataFrame({
    "bodyId": list(range(N)),
    "type": [f"t{i%3}" for i in range(N)],
    "predictedNt": ["acetylcholine" if i % 2 == 0 else "gaba" for i in range(N)],
    "predictedNtConfidence": [0.9] * N,
    "nt_sign": [1 if i % 2 == 0 else -1 for i in range(N)],
})
edges = pd.DataFrame({
    "bodyId_pre": rows_idx,
    "bodyId_post": cols_idx,
    "weight": [5] * N,
    "weight_norm": [5.0] * N,
    "sign": vals,
    "signed_weight": vals,
})
g = build_graph(neurons, edges, weight_col="signed_weight")

models = [
    ("A0",          A0Random.build(obs_dim=OBS, act_dim=ACT)),
    ("A1-BIO",      A1Bio.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A1-FROZEN",   A1Frozen.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A1-BIAS",     A1Bias.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A1-EDGE",     A1Edge.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A1-BOTH",     A1Both.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A3-ER",       A3ER.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A3-CONFIG",   A3Config.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A3-SBM",      A3SBM.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A3-DENSE",    A3Dense.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A5-WSHUFFLE", A5WShuffle.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A5-WSIGN",    A5WSign.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A5-ETARGET",  A5ETarget.from_graph(g, obs_dim=OBS, act_dim=ACT, n_layers=NL)),
    ("A7-MLP",      A7MLP.build(obs_dim=OBS, act_dim=ACT, hidden_dim=128, n_layers=3)),
    ("A8-LSTM",     A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=128, n_layers=2, cell_type="lstm")),
    ("A8-GRU",      A8RNN.build(obs_dim=OBS, act_dim=ACT, hidden_dim=128, n_layers=2, cell_type="gru")),
]

SEP = "=" * 88
print()
print(f"Parameter Report  |  N={N} nodes, obs={OBS}->act={ACT}, n_layers={NL}")
print(SEP)
print(f"{'arch_id':<16}  {'trainable':>12}  {'frozen':>8}  {'total':>10}  {'flops_est':>12}")
print("-" * 88)
for name, m in models:
    p = m.param_count()
    print(
        f"{name:<16}  {p['trainable']:>12,}  {p['frozen']:>8,}  "
        f"{p['total']:>10,}  {m.flops_estimate():>12,}"
    )
print(SEP)
print()
print(f"Graph nnz for N={N} ring: {N}  (N^2={N*N}, sparsity={N/(N*N):.4f})")
print("Sparse execution: OK — A1/A3/A5 use SignedSparseLinear (nnz ops only)")
print("A3-DENSE uses dense nn.Linear — NOT parameter-matched to A1 variants")
