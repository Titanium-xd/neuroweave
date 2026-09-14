"""
scripts/models/sparse_benchmark.py — Sparse operator performance benchmark.

Benchmarks SignedSparseLinear forward pass on synthetic graphs of increasing size
to validate that sparse execution is faster than dense for sparse graphs (nnz << N^2).

Reports wall-clock time, effective FLOP/s, and speedup over dense baseline.
This benchmark uses CPU only (matching the reference hardware phase).

IMPORTANT: This is a performance measurement, NOT a production workload.
Do not use the results here to make strong claims without GPU validation.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _er_adj(n: int, nnz: int, seed: int = 42) -> sp.coo_matrix:
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < nnz:
        r = rng.integers(0, n)
        c = rng.integers(0, n)
        if r != c:
            edges.add((int(r), int(c)))
    rows, cols = zip(*edges)
    vals = rng.choice([-1.0, 1.0], size=nnz).astype(np.float32)
    return sp.coo_matrix((vals, (list(rows), list(cols))), shape=(n, n))


class BenchResult(NamedTuple):
    n_nodes: int
    nnz: int
    sparsity: float
    sparse_ms: float
    dense_ms: float
    speedup: float
    sparse_gflops: float
    dense_gflops: float


def _time_fn(fn, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times)) * 1000  # ms


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> list[BenchResult]:
    from abb.models.operator import SignedSparseLinear

    configs = [
        # (n_nodes, target_nnz_fraction)
        (50,    0.20),   # dense-ish small
        (50,    0.05),   # sparse small
        (200,   0.05),   # medium sparse
        (500,   0.02),   # realistic MaleCNS-like
        (500,   0.005),  # very sparse
        (1000,  0.002),  # large sparse
    ]

    results: list[BenchResult] = []

    for n, frac in configs:
        nnz = max(n, int(n * n * frac))
        nnz = min(nnz, n * (n - 1))  # cap at maximum possible

        adj = _er_adj(n, nnz, seed=42)
        op = SignedSparseLinear(adj, trainable_weights=False)
        dense_W = torch.randn(n, n)

        x = torch.randn(1, n)

        sparse_ms = _time_fn(lambda: op(x))

        def _dense_fwd():
            return x @ dense_W

        dense_ms = _time_fn(_dense_fwd)

        speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")
        flops = 2 * nnz
        sparsity = nnz / (n * n)

        # GFLOP/s
        sparse_gflops = (flops / 1e9) / (sparse_ms / 1000)
        dense_gflops  = (2 * n * n / 1e9) / (dense_ms / 1000)

        results.append(BenchResult(
            n_nodes=n, nnz=nnz, sparsity=sparsity,
            sparse_ms=sparse_ms, dense_ms=dense_ms, speedup=speedup,
            sparse_gflops=sparse_gflops, dense_gflops=dense_gflops,
        ))

    return results


def print_results(results: list[BenchResult]) -> None:
    hdr = (
        f"{'N':>6}  {'nnz':>8}  {'sparsity':>10}  "
        f"{'sparse(ms)':>12}  {'dense(ms)':>12}  "
        f"{'speedup':>9}  {'sparse GF/s':>12}  {'dense GF/s':>12}"
    )
    print()
    print("=" * len(hdr))
    print("  SignedSparseLinear CPU Benchmark  (median of 20 runs)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.n_nodes:>6}  {r.nnz:>8}  {r.sparsity:>10.4f}  "
            f"{r.sparse_ms:>12.4f}  {r.dense_ms:>12.4f}  "
            f"{r.speedup:>9.2f}×  {r.sparse_gflops:>12.4f}  {r.dense_gflops:>12.4f}"
        )
    print("=" * len(hdr))
    print()
    # Summary
    avg_speedup = np.mean([r.speedup for r in results])
    print(f"  Average speedup (sparse vs dense): {avg_speedup:.2f}×")
    print(f"  Note: PyTorch sparse COO is most efficient for sparsity < 5%.")
    print(f"        For very small N (<50), dense may outperform due to overhead.")
    print()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    results = run_benchmark()
    print_results(results)
