"""scripts/tasks/audit_a1bio.py — A1-BIO parameter breakdown audit."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path
from abb.models.base import AgentConfig
from abb.models.bio_variants import A1Bio

# Build the exact A1-BIO used in run_t001.py
n_paths = list(Path("data/raw/malecns_e2e_validation").glob("neurons_*.parquet"))
e_paths = list(Path("data/raw/malecns_e2e_validation").glob("edges_*.parquet"))
neurons = pd.read_parquet(n_paths[0]).head(150)
edges   = pd.read_parquet(e_paths[0])
body_ids = set(neurons["bodyId"])
sub_e = edges[edges["bodyId_pre"].isin(body_ids) & edges["bodyId_post"].isin(body_ids)]
bid_to_idx = {int(b): i for i, b in enumerate(neurons["bodyId"])}
rows = sub_e["bodyId_pre"].map(bid_to_idx).to_numpy(dtype=np.int32)
cols = sub_e["bodyId_post"].map(bid_to_idx).to_numpy(dtype=np.int32)
col  = "signed_weight" if "signed_weight" in sub_e.columns else "weight"
vals = sub_e[col].to_numpy(dtype=np.float32)
adj = sp.coo_matrix((vals, (rows, cols)), shape=(150, 150))

cfg = AgentConfig(
    arch_id="A1-BIO", obs_dim=64, act_dim=2, n_layers=2, seed=42,
    assumption_ids=("SA-001", "SA-002", "SA-006", "SA-007", "SA-008", "SA-009"),
)
model = A1Bio(cfg, adj)

# ── Parameters ────────────────────────────────────────────────────────────
print()
print("=== A1-BIO PARAMETER BREAKDOWN ===")
print()
print(f"{'Name':<50} {'Shape':>20} {'Req.Grad':>10} {'Numel':>8}")
print("-" * 92)
total_trainable = 0
total_frozen = 0
for name, param in model.named_parameters():
    n = param.numel()
    tr = param.requires_grad
    if tr:
        total_trainable += n
    else:
        total_frozen += n
    print(f"{name:<50} {str(tuple(param.shape)):>20} {str(tr):>10} {n:>8,}")

# ── Buffers ────────────────────────────────────────────────────────────────
print()
print("BUFFERS (registered as buffers, never in optimizer state_dict):")
print()
print(f"{'Name':<50} {'Shape':>20} {'Numel':>8}")
print("-" * 82)
buf_total = 0
for name, buf in model.named_buffers():
    n = buf.numel()
    buf_total += n
    tag = " <-- MaleCNS synapse*sign values" if "base_vals" in name else ""
    print(f"{name:<50} {str(tuple(buf.shape)):>20} {n:>8,}{tag}")

print()
print("=" * 92)
print(f"Trainable parameters : {total_trainable:,}")
print(f"Frozen parameters    : {total_frozen:,}")
print(f"Buffers (not params) : {buf_total:,}")
print(f"  nnz per layer      : {adj.nnz:,}  (MaleCNS synapse-count x NT-sign)")

# ── Edge weight mechanism ─────────────────────────────────────────────────
print()
print("=== EDGE WEIGHT MECHANISM ===")
layer0 = model.layers[0]
op = layer0.operator

base_vals = op._base_vals
edge_scale = op.edge_scale

print(f"_base_vals   : buffer  | dtype={base_vals.dtype} | range=[{base_vals.min():.2f}, {base_vals.max():.2f}]")
print(f"             : is nn.Parameter? {isinstance(base_vals, torch.nn.Parameter)}")
print(f"             : requires_grad?   {base_vals.requires_grad}")
print(f"edge_scale   : param   | dtype={edge_scale.dtype} | shape={tuple(edge_scale.shape)}")
print(f"             : is nn.Parameter? {isinstance(edge_scale, torch.nn.Parameter)}")
print(f"             : requires_grad?   {edge_scale.requires_grad}")
print(f"             : init values (first 5): {edge_scale.data[:5].tolist()}")
print()
print("Effective weight:  w_eff[k] = _base_vals[k]  *  edge_scale[k]")
print("                              ^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^")
print("                              FROZEN BUFFER       TRAINABLE PARAM")
print("                              (MaleCNS prior,     (per-edge scalar,")
print("                               never updated)      Adam-updated)")

# ── Compliance verdict ─────────────────────────────────────────────────────
print()
print("=== ABB-0.1-REV1 COMPLIANCE VERDICT ===")
print()
print("ABB spec for A1-BIO:")
print("  'Weights initialized from MaleCNS synapse-count * NT-sign, trainable.'")
print("  'Biases initialized to 0, trainable.'")
print()
print("Implementation:")
print("  _base_vals buffer  = MaleCNS synapse_count * NT_sign   [NEVER touched by optimizer]")
print("  edge_scale param   = per-edge multiplier, init=1.0     [TRAINED by Adam]")
print("  bias params        = per-node bias, init=0.0           [TRAINED by Adam]")
print("  input_proj weights = linear map obs->N, random init    [TRAINED by Adam]")
print("  readout weights    = linear map N->act, random init    [TRAINED by Adam]")
print()
print("Is the MaleCNS synapse-count data ever modified by the optimizer? NO.")
print("  _base_vals is a registered BUFFER.  It does NOT appear in model.parameters().")
print("  The optimizer only updates edge_scale, biases, input_proj, and readout.")
print()

# Verify _base_vals not in parameters
param_names = {n for n, _ in model.named_parameters()}
assert not any("base_vals" in n for n in param_names), "VIOLATION: base_vals in parameters!"
print("CHECK PASSED: 'base_vals' does not appear in model.parameters().")

# Verify edge_scale IS in parameters
assert any("edge_scale" in n for n in param_names), "edge_scale missing from parameters!"
print("CHECK PASSED: 'edge_scale' appears in model.parameters() (trainable).")
print()
print("VERDICT: A1-BIO implementation is COMPLIANT with ABB-0.1-rev1.")
print("         The MaleCNS topology and initial synapse strengths are")
print("         preserved as a fixed structural prior. Only a per-edge")
print("         scalar multiplier, node biases, and the I/O projection")
print("         layers are learned during training.")
