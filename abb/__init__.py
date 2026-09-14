"""
abb — Animal Brain Benchmark
============================
Phase 2 Part 1: Data Layer (complete)
Phase 2 Part 2: GNN Architecture Layer (complete)

Public constants
----------------
BENCHMARK_VERSION : str
    Frozen benchmark version string (matches all documentation headers).
DATASET_ID : str
    Canonical MaleCNS v1.0 neuPrint dataset identifier.
"""

BENCHMARK_VERSION: str = "ABB-0.1-rev1"
DATASET_ID: str = "male-cns:v1.0"
NEUPRINT_SERVER: str = "https://neuprint.janelia.org"

__version__: str = "0.1.0.dev1"
__all__ = ["BENCHMARK_VERSION", "DATASET_ID", "NEUPRINT_SERVER"]
