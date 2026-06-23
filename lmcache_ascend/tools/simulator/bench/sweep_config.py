"""Sweep configuration: SimConfig and SweepConfig."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from simulator.model.layout import engine_ids_for_pd
from simulator.model.capacity import (
    TierSpec,
    default_tiers,
    format_tier_capacity,
    gib_for_blocks,
    kv_bytes_per_token,
)
from simulator.model.layout import SimResources
from .mooncake_trace import MOONCAKE_TOKENS_PER_BLOCK
from .presets import DEFAULT_PRESET_NAMES, EngineBuildConfig

@dataclass(frozen=True)
class SimConfig:
    """Resource and scheduler knobs shared across policy presets."""

    hbm_gib: float = 32.0
    dram_gib: float = 64.0
    ssd_gib: float = 256.0
    kv_model: str = "llama3-8b"
    kv_bytes_per_token: float | None = None
    tokens_per_block: int = MOONCAKE_TOKENS_PER_BLOCK
    tiers: tuple[TierSpec, ...] | None = None
    dram_chunk_blocks: int = 4
    ssd_chunk_blocks: int = 4
    ssd_write_speed: float = 8.0
    max_num_seqs: int = 12
    max_num_batched_tokens: int = 24
    compute_speed: float = 64.0
    link_speed: float = 32.0
    interconnect_speed: float | None = None
    link_latency: float = 0.01
    work_per_block: float = 1.0
    work_per_transfer: float = 1.0
    max_steps: int = 5_000_000
    max_steps_per_request: int = 500
    wall_timeout_s: float | None = None
    show_progress: bool = False
    medium_models: dict[str, str] | None = None
    medium_model_params: dict[str, dict] = field(default_factory=dict)

    def resolved_kv_bytes_per_token(self) -> float:
        if self.kv_bytes_per_token is not None:
            return self.kv_bytes_per_token
        return kv_bytes_per_token(self.kv_model)

    def resources(
        self,
        *,
        tokens_per_block: int | None = None,
        prefill_ids: tuple[str, ...] | None = None,
        decode_ids: tuple[str, ...] | None = None,
    ) -> SimResources:
        tpb = tokens_per_block if tokens_per_block is not None else self.tokens_per_block
        pids, dids = engine_ids_for_pd(num_prefill=1, num_decode=1)
        if prefill_ids is not None:
            pids = prefill_ids
        if decode_ids is not None:
            dids = decode_ids
        return SimResources.from_tiers(
            self.resolved_tiers(prefill_ids=pids, decode_ids=dids),
            tokens_per_block=tpb,
            kv_bytes_per_token=self.resolved_kv_bytes_per_token(),
        )

    def resolved_tiers(
        self,
        *,
        prefill_ids: tuple[str, ...] | None = None,
        decode_ids: tuple[str, ...] | None = None,
    ) -> tuple[TierSpec, ...]:
        if self.tiers is not None:
            return self.tiers
        pids, dids = engine_ids_for_pd(num_prefill=1, num_decode=1)
        if prefill_ids is not None:
            pids = prefill_ids
        if decode_ids is not None:
            dids = decode_ids
        return default_tiers(
            hbm_gib=self.hbm_gib,
            dram_gib=self.dram_gib,
            ssd_gib=self.ssd_gib,
            dram_chunk_blocks=self.dram_chunk_blocks,
            ssd_chunk_blocks=self.ssd_chunk_blocks,
            prefill_ids=pids,
            decode_ids=dids,
        )

    @classmethod
    def with_block_slots(
        cls,
        *,
        hbm: int,
        dram: int = 80,
        ssd: int = 160,
        tokens_per_block: int = MOONCAKE_TOKENS_PER_BLOCK,
        kv_bytes_per_token: float = 256.0,
        dram_chunk_blocks: int = 4,
        ssd_chunk_blocks: int = 4,
        num_prefill: int = 1,
        num_decode: int = 1,
        **kwargs: object,
    ) -> SimConfig:
        """Build config from explicit block slot counts (tests)."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in kwargs.items() if k in fields}
        prefill_ids, decode_ids = engine_ids_for_pd(
            num_prefill=num_prefill, num_decode=num_decode
        )
        tiers_list: list[TierSpec] = []
        for eid in prefill_ids + decode_ids:
            tiers_list.append(
                TierSpec(
                    f"{eid}:hbm",
                    gib_for_blocks(
                        hbm,
                        tokens_per_block=tokens_per_block,
                        kv_bytes_per_token=kv_bytes_per_token,
                    ),
                    1,
                )
            )
        primary = prefill_ids[0]
        tiers_list.extend(
            [
                TierSpec(
                    f"{primary}:dram",
                    gib_for_blocks(
                        dram,
                        tokens_per_block=tokens_per_block,
                        kv_bytes_per_token=kv_bytes_per_token,
                        chunk_blocks=dram_chunk_blocks,
                    ),
                    dram_chunk_blocks,
                ),
                TierSpec(
                    f"{primary}:ssd",
                    gib_for_blocks(
                        ssd,
                        tokens_per_block=tokens_per_block,
                        kv_bytes_per_token=kv_bytes_per_token,
                        chunk_blocks=ssd_chunk_blocks,
                    ),
                    ssd_chunk_blocks,
                ),
            ]
        )
        tiers = tuple(tiers_list)
        return cls(
            tiers=tiers,
            kv_bytes_per_token=kv_bytes_per_token,
            tokens_per_block=tokens_per_block,
            **extra,
        )

    def engine_build(self, *, tokens_per_block: int | None = None) -> EngineBuildConfig:
        return EngineBuildConfig(
            compute_speed=self.compute_speed,
            link_speed=self.link_speed,
            link_latency=self.link_latency,
            ssd_write_speed=self.ssd_write_speed,
            work_per_block=self.work_per_block,
            work_per_transfer=self.work_per_transfer,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            resources=self.resources(tokens_per_block=tokens_per_block),
            medium_models=self.medium_models,
            medium_model_params=dict(self.medium_model_params),
        )



@dataclass
class SweepConfig:
    presets: tuple[str, ...] = DEFAULT_PRESET_NAMES
    num_requests: int = 64
    seeds: int = 3
    base_seed: int = 1000
    sim: SimConfig = field(default_factory=SimConfig)
    csv_path: str | None = None
    raw_csv_path: str | None = None
    trace_path: str | None = None
    trace_offset: int = 0
    trace_time_scale: float = 0.001
    tokens_per_block: int = MOONCAKE_TOKENS_PER_BLOCK
    drop_oversized: bool = False
    read_path: str | None = None
    pull_threshold: float = 1.0
    experiment_spec: str | None = None
    num_prefill: int = 1
    num_decode: int = 1
    routing: str = "bijection"

