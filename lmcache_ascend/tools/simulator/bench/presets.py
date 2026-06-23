"""Named policy presets and PD engine construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from simulator.runtime.engine import Engine
from simulator.policy.config import EngineConfig, LifecycleSpec, PlacementSpec, build_placement_spec
from simulator.core.memory import Memory
from simulator.policy.policies import EnginePolicies
from simulator.core.request import Request
from simulator.core.resource import BandwidthResource, ComputeResource
from simulator.runtime.tasks import TaskPool
from simulator.model.layout import SimResources, Topology, TOPOLOGY_TIERS, build_eviction_map, build_topology


@dataclass(frozen=True)
class PresetSpec:
    name: str
    description: str
    topology: Literal["hbm_only", "hbm_dram", "hbm_dram_ssd"]
    decode_pull: tuple[str, ...] = ("npu-0:hbm",)
    mirror_tiers: tuple[str, ...] = ()
    async_write_tiers: frozenset[str] = frozenset()
    retention: str = "unbounded"
    retention_params: dict = field(default_factory=dict)
    eviction: str = "lru"
    local_eviction: str | None = None
    downstream_eviction: str | None = None
    hold_kv_on_complete: bool = True
    prefill_retain_prefix_cache: bool = False
    decode_retain_prefix_cache: bool = False
    store_on_complete: tuple[str, ...] = ()
    mirror_on_forward: bool = True
    decode_read_path: str | None = None
    decode_read_path_params: dict = field(default_factory=dict)


def _placement_spec(spec: PresetSpec) -> PlacementSpec:
    return build_placement_spec(
        mirror_tiers=spec.mirror_tiers,
        async_write_tiers=spec.async_write_tiers,
        mirror_on_forward=spec.mirror_on_forward,
        retention_name=spec.retention,
        retention_params=spec.retention_params,
    )


def prefill_config(spec: PresetSpec, topo: Topology) -> EngineConfig:
    return EngineConfig(
        engine_id=topo.prefill_engine_id,
        local_tier=topo.prefill_local_tier,
        pull_mode="compute_only",
        placement=_placement_spec(spec),
        lifecycle=LifecycleSpec(
            hold_kv_on_complete=spec.hold_kv_on_complete,
            retain_prefix_cache=spec.prefill_retain_prefix_cache,
            store_on_complete=spec.store_on_complete,
        ),
    )


def decode_config(spec: PresetSpec, topo: Topology) -> EngineConfig:
    return EngineConfig(
        engine_id=topo.decode_engine_id,
        local_tier=topo.decode_local_tier,
        pull_sources=topo.decode_pull_sources(spec.decode_pull),
        pull_mode="ordered_pull",
        read_path_name=spec.decode_read_path,
        read_path_params=spec.decode_read_path_params,
        placement=_placement_spec(spec),
        lifecycle=LifecycleSpec(
            retain_prefix_cache=spec.decode_retain_prefix_cache,
        ),
    )


PRESETS: dict[str, PresetSpec] = {
    "baseline": PresetSpec(
        name="baseline",
        description="P compute-only; D ordered pull from P HBM",
        topology="hbm_only",
    ),
    "ordered_pull": PresetSpec(
        name="ordered_pull",
        description="Same as baseline (explicit ordered-pull decode)",
        topology="hbm_only",
    ),
    "min_cost_pull": PresetSpec(
        name="min_cost_pull",
        description="baseline topology; decode uses min-cost pull vs recompute",
        topology="hbm_only",
        decode_read_path="min_cost",
    ),
    "dram_tier": PresetSpec(
        name="dram_tier",
        description="P sync DRAM mirror; D pull DRAM then HBM",
        topology="hbm_dram",
        decode_pull=("npu-0:dram", "npu-0:hbm"),
        mirror_tiers=("npu-0:dram",),
    ),
    "consume_on_pull": PresetSpec(
        name="consume_on_pull",
        description="baseline + consume source on pull",
        topology="hbm_only",
        retention="consume_on_pull",
    ),
    "single_copy": PresetSpec(
        name="single_copy",
        description="baseline + one copy per tier",
        topology="hbm_only",
        retention="single_copy",
    ),
    "ssd_tier": PresetSpec(
        name="ssd_tier",
        description="P sync DRAM + async SSD; D pull SSD/DRAM/HBM",
        topology="hbm_dram_ssd",
        decode_pull=("npu-0:ssd", "npu-0:dram", "npu-0:hbm"),
        mirror_tiers=("npu-0:dram",),
        async_write_tiers=frozenset({"npu-0:ssd"}),
    ),
    "global_cap_2": PresetSpec(
        name="global_cap_2",
        description="baseline + global cap 2 across P/D HBM",
        topology="hbm_only",
        retention="global_cap",
        retention_params={
            "max_total": 2,
            "tier_keys": ("npu-0:hbm", "npu-1:hbm"),
            "per_tier_cap": None,
        },
    ),
    "evict_lru": PresetSpec(
        name="evict_lru",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; LRU on all tiers"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_prefix_cache=True,
        eviction="lru",
    ),
    "evict_fifo": PresetSpec(
        name="evict_fifo",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; FIFO on all tiers"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_prefix_cache=True,
        eviction="fifo",
    ),
    "evict_random": PresetSpec(
        name="evict_random",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; random on all tiers"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_prefix_cache=True,
        eviction="random",
    ),
    "evict_lfu": PresetSpec(
        name="evict_lfu",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; LFU on all tiers"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_prefix_cache=True,
        eviction="lfu",
    ),
}

EVICTION_PRESET_NAMES: tuple[str, ...] = (
    "evict_lru",
    "evict_fifo",
    "evict_random",
    "evict_lfu",
)

DEFAULT_PRESET_NAMES: tuple[str, ...] = (
    "baseline",
    "ordered_pull",
    "dram_tier",
    "consume_on_pull",
)


@dataclass(frozen=True)
class EngineBuildConfig:
    compute_speed: float = 64.0
    link_speed: float = 32.0
    link_latency: float = 0.01
    ssd_write_speed: float = 8.0
    work_per_block: float = 1.0
    work_per_transfer: float = 1.0
    max_num_seqs: int = 12
    max_num_batched_tokens: int = 24
    resources: SimResources = field(default_factory=SimResources.default)


def _transfer_links(
    memories: dict[str, Memory],
    pull_sources: list[str],
    link: BandwidthResource,
) -> dict[str, BandwidthResource]:
    return {src: link for src in pull_sources if src in memories}


def _write_links(
    memories: dict[str, Memory],
    cfg: EngineBuildConfig,
) -> dict[str, BandwidthResource]:
    links: dict[str, BandwidthResource] = {}
    if "npu-0:ssd" in memories:
        links["npu-0:ssd"] = BandwidthResource(
            base_speed=cfg.ssd_write_speed,
            latency=cfg.link_latency,
        )
    return links


def build_topology_for_preset(
    spec: PresetSpec,
    resources: SimResources | None = None,
    *,
    rng_seed: int = 0,
) -> Topology:
    tier_keys = TOPOLOGY_TIERS[spec.topology]
    eviction_map = build_eviction_map(
        tier_keys,
        local=spec.local_eviction or spec.eviction,
        downstream=spec.downstream_eviction or spec.eviction,
        rng_seed=rng_seed,
    )
    return build_topology(
        spec.topology,
        resources=resources or SimResources.default(),
        eviction_map=eviction_map,
    )


def build_pd_engines(
    requests: list[Request],
    pool: TaskPool,
    preset: PresetSpec | str,
    *,
    cfg: EngineBuildConfig | None = None,
    resources: SimResources | None = None,
    rng_seed: int = 0,
) -> tuple[Engine, Engine, Topology]:
    spec = PRESETS[preset] if isinstance(preset, str) else preset
    build = cfg or EngineBuildConfig()
    res = resources or build.resources
    topo = build_topology_for_preset(spec, res, rng_seed=rng_seed)
    compute = ComputeResource(base_speed=build.compute_speed)
    link = BandwidthResource(base_speed=build.link_speed, latency=build.link_latency)
    prefill_cfg = prefill_config(spec, topo)
    decode_cfg = decode_config(spec, topo)
    prefill_policies = EnginePolicies.from_config(prefill_cfg, topo.graph)
    decode_policies = EnginePolicies.from_config(decode_cfg, topo.graph)
    prefill = Engine(
        engine_id=topo.prefill_engine_id,
        requests=requests,
        pool=pool,
        memories=topo.memories,
        policies=prefill_policies,
        compute_res=compute,
        work_per_block=build.work_per_block,
        max_num_seqs=build.max_num_seqs,
        max_num_batched_tokens=build.max_num_batched_tokens,
        enable_chunked_prefill=True,
        write_links=_write_links(topo.memories, build),
        work_per_store=build.work_per_transfer,
    )
    decode = Engine(
        engine_id=topo.decode_engine_id,
        requests=[],
        pool=pool,
        memories=topo.memories,
        policies=decode_policies,
        compute_res=compute,
        bandwidth_res=link,
        transfer_links=_transfer_links(
            topo.memories, list(decode_policies.schedule.config.pull_sources), link
        ),
        work_per_block=build.work_per_block,
        work_per_transfer=build.work_per_transfer,
        max_num_seqs=build.max_num_seqs,
        max_num_batched_tokens=build.max_num_batched_tokens,
        enable_chunked_prefill=True,
        remote_kv_wait=True,
    )
    return prefill, decode, topo
