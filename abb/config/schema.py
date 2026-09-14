"""
abb.config.schema — Pydantic v2 schemas for all ABB configuration types.

All experimental parameters live here. No magic numbers are embedded in
data-access or model code; everything is driven from these validated schemas.

Configuration tier: DATA LAYER (Phase 2)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SubgraphStrategy(str, Enum):
    """Supported subgraph sampling strategies (BENCHMARK_DESIGN §4)."""

    KHOP = "khop"
    RANDOM_WALK = "random_walk"
    PATHWAY = "pathway"


class NormalizationScheme(str, Enum):
    """Synapse-count normalization schemes — SA-001 ablation axis."""

    RAW = "raw"            # Raw synapse count [SA-001 default]
    LOG1P = "log1p"        # log(1 + count)
    OUT_DEGREE = "out_degree"  # count / total_out_degree(presynaptic_neuron)
    BINARY = "binary"      # 1 if count > 0 else 0


class NTSignScheme(str, Enum):
    """NT-to-sign mapping scheme — SA-002 ablation axis."""

    NT_RULE = "nt_rule"    # Standard SA-002 rule table [default]
    ALL_LEARNABLE = "all_learnable"   # SA-002-ALT: ignore predictions; all signs learnable
    ALL_UNSIGNED = "all_unsigned"     # SA-002-NULL: no sign applied
    ALL_RANDOM = "all_random"         # SA-002-RAND: signs randomly reassigned


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class NeuPrintConfig(BaseModel):
    """neuPrint connection parameters. Token is NOT stored here — env var only."""

    server_url: str = Field(
        default="https://neuprint.janelia.org",
        description="neuPrint server URL. Do NOT embed tokens here.",
    )
    dataset_id: str = Field(
        default="male-cns:v1.0",
        description="Exact neuPrint dataset identifier (version-pinned).",
    )
    max_neurons_per_fetch: int = Field(
        default=5000,
        ge=1,
        le=50000,
        description="Maximum neurons per single fetch_neurons() call.",
    )
    connection_timeout_s: int = Field(
        default=60,
        ge=5,
        description="HTTP connection timeout in seconds.",
    )

    model_config = {"frozen": True}


class CacheConfig(BaseModel):
    """Local Parquet cache configuration."""

    cache_dir: Path = Field(
        default=Path("data/raw/malecns_v1_0"),
        description="Directory for cached Parquet files and provenance sidecars.",
    )
    overwrite_existing: bool = Field(
        default=False,
        description="If True, re-download and overwrite existing cached files.",
    )
    verify_checksums: bool = Field(
        default=True,
        description="Verify SHA-256 checksums on cache load.",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_path(cls, v: object) -> Path:
        return Path(v)

    model_config = {"frozen": True}


class ConnectivityRequirements(BaseModel):
    """
    Connectivity requirements for subgraph acceptance.
    Source: BENCHMARK_DESIGN.md §4.1
    """

    min_weakly_connected_frac: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of nodes that must belong to the largest "
            "weakly-connected component. Subgraphs below this are rejected."
        ),
    )
    min_edge_density: float = Field(
        default=1e-3,
        ge=0.0,
        description="Minimum edge density (edges / possible_edges). Rejects disconnected samples.",
    )
    min_nt_coverage: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of nodes with a non-null predictedNt value.",
    )
    max_rejection_attempts: int = Field(
        default=20,
        ge=1,
        description="Maximum number of re-sampling attempts before raising an error.",
    )

    model_config = {"frozen": True}


class SubgraphConfig(BaseModel):
    """Subgraph sampling configuration."""

    strategy: SubgraphStrategy = Field(
        default=SubgraphStrategy.KHOP,
        description="Sampling strategy.",
    )
    target_size: int = Field(
        default=5000,
        ge=2,
        le=50000,
        description="Target number of neurons in the extracted subgraph.",
    )
    khop_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of hops for k-hop neighbourhood strategy.",
    )
    random_walk_steps: int = Field(
        default=100000,
        ge=1000,
        description="Total steps for random-walk strategy.",
    )
    random_walk_walkers: int = Field(
        default=100,
        ge=1,
        description="Number of parallel walkers for random-walk strategy.",
    )
    seed: int = Field(
        default=42,
        description="RNG seed. Same seed guarantees deterministic subgraph selection.",
    )
    connectivity: ConnectivityRequirements = Field(
        default_factory=ConnectivityRequirements,
    )

    model_config = {"frozen": True}


class WeightConfig(BaseModel):
    """Edge weight normalization and sign assignment configuration."""

    normalization: NormalizationScheme = Field(
        default=NormalizationScheme.RAW,
        description="Synapse-count normalization scheme [SA-001].",
    )
    nt_sign_scheme: NTSignScheme = Field(
        default=NTSignScheme.NT_RULE,
        description="NT-to-sign mapping scheme [SA-002].",
    )
    nt_sign_random_seed: Optional[int] = Field(
        default=None,
        description="Seed for SA-002-RAND random sign reassignment. None = use SubgraphConfig.seed.",
    )
    weight_clip_max: Optional[float] = Field(
        default=None,
        description="If set, clip synapse count weights at this value before normalization.",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """
    Complete data-layer configuration for one ABB experiment.

    Every downloaded/cached dataset is tied to this config's neuprint and cache
    settings. Changing any field changes the cache key (via hash).
    """

    neuprint: NeuPrintConfig = Field(default_factory=NeuPrintConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    subgraph: SubgraphConfig = Field(default_factory=SubgraphConfig)
    weight: WeightConfig = Field(default_factory=WeightConfig)
    benchmark_version: str = Field(
        default="ABB-0.1-rev1",
        description="Benchmark version string. Must match documentation version.",
    )
    assumption_ids_applied: list[str] = Field(
        default_factory=lambda: ["SA-001", "SA-002", "SA-006"],
        description="Assumption IDs active in this config. Recorded in provenance.",
    )

    @model_validator(mode="after")
    def _check_version(self) -> "DataConfig":
        from abb import BENCHMARK_VERSION
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(
                f"benchmark_version mismatch: config has '{self.benchmark_version}', "
                f"package expects '{BENCHMARK_VERSION}'. Update config or package."
            )
        return self

    model_config = {"frozen": True}
