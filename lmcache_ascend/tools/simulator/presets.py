"""Named policy presets and PD engine construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .engine import Engine
from .memory import Memory
from .eviction import EvictionPolicy, HbmEvictionKind, make_hbm_eviction
from .policies import EnginePolicies
from .request import Request
from .resource import BandwidthResource, ComputeResource
from .tasks import TaskPool
from .topology import SimResources, Topology, build_topology


@dataclass(frozen=True)
class PresetSpec:
    name: str
    description: str
    topology: Literal["hbm_only", "hbm_dram", "hbm_dram_ssd"]
    decode_pull: tuple[str, ...] = ("npu-0:hbm",)
    mirror_tiers: tuple[str, ...] = ()
    async_write_tiers: frozenset[str] = frozenset()
    retention: Literal[
        "unbounded", "single_copy", "consume_on_pull", "global_cap"
    ] = "unbounded"
    global_cap_max: int = 2
    global_cap_tiers: tuple[str, ...] = ("npu-0:hbm", "npu-1:hbm")
    global_cap_per_tier: int | None = None
    hbm_eviction: HbmEvictionKind = "lru"
    tier_eviction: HbmEvictionKind = "lru"
    hold_kv_on_complete: bool = True
    prefill_retain_hbm_prefix_cache: bool = False
    decode_retain_hbm_prefix_cache: bool = False
    store_on_complete: tuple[str, ...] = ()
    mirror_on_forward: bool = True


def _hbm_eviction(spec: PresetSpec, *, rng_seed: int) -> EvictionPolicy:
    return make_hbm_eviction(spec.hbm_eviction, seed=rng_seed)


def _tier_eviction_policies(
    spec: PresetSpec, *, rng_seed: int
) -> dict[str, EvictionPolicy]:
    tier_keys = set(spec.mirror_tiers) | set(spec.store_on_complete)
    if not tier_keys:
        return {}
    policy = make_hbm_eviction(spec.tier_eviction, seed=rng_seed)
    return {tier_key: policy for tier_key in tier_keys}


def _effect_kwargs(spec: PresetSpec, *, rng_seed: int) -> dict:
    return dict(
        mirror_tiers=spec.mirror_tiers,
        async_write_tiers=spec.async_write_tiers,
        mirror_on_forward=spec.mirror_on_forward,
        tier_eviction=_tier_eviction_policies(spec, rng_seed=rng_seed),
        retention=spec.retention,
        global_cap_max=spec.global_cap_max,
        global_cap_tiers=spec.global_cap_tiers,
        global_cap_per_tier=spec.global_cap_per_tier,
    )


def _prefill_memory_kwargs(spec: PresetSpec) -> dict:
    return dict(
        retain_hbm_prefix_cache=spec.prefill_retain_hbm_prefix_cache,
        store_tiers_on_complete=spec.store_on_complete,
    )


def _decode_memory_kwargs(spec: PresetSpec) -> dict:
    return dict(
        retain_hbm_prefix_cache=spec.decode_retain_hbm_prefix_cache,
        store_tiers_on_complete=(),
    )


def _prefill_policies(
    spec: PresetSpec, topo: Topology, *, rng_seed: int = 0
) -> EnginePolicies:
    return EnginePolicies.compute_only(
        topo.prefill_hbm,
        hbm_eviction=_hbm_eviction(spec, rng_seed=rng_seed),
        **_effect_kwargs(spec, rng_seed=rng_seed),
    )


def _decode_policies(
    spec: PresetSpec, topo: Topology, *, rng_seed: int = 0
) -> EnginePolicies:
    sources = topo.decode_pull_sources(spec.decode_pull)
    return EnginePolicies.ordered_pull(
        topo.decode_hbm,
        sources,
        hbm_eviction=_hbm_eviction(spec, rng_seed=rng_seed),
        **_effect_kwargs(spec, rng_seed=rng_seed),
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
        global_cap_max=2,
        global_cap_tiers=("npu-0:hbm", "npu-1:hbm"),
        global_cap_per_tier=None,
    ),
    "evict_lru": PresetSpec(
        name="evict_lru",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; LRU on HBM and DRAM"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_hbm_prefix_cache=True,
        hbm_eviction="lru",
        tier_eviction="lru",
    ),
    "evict_fifo": PresetSpec(
        name="evict_fifo",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; FIFO on HBM and DRAM"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_hbm_prefix_cache=True,
        hbm_eviction="fifo",
        tier_eviction="fifo",
    ),
    "evict_random": PresetSpec(
        name="evict_random",
        description=(
            "LMCache PD: P offloads prefix to DRAM on complete and frees P HBM; "
            "D pulls prefix from DRAM only; D keeps APC on D HBM; random on HBM and DRAM"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_hbm_prefix_cache=True,
        hbm_eviction="random",
        tier_eviction="random",
    ),
}

EVICTION_PRESET_NAMES: tuple[str, ...] = (
    "evict_lru",
    "evict_fifo",
    "evict_random",
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
) -> Topology:
    return build_topology(spec.topology, resources=resources or SimResources.default())


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
    topo = build_topology_for_preset(spec, res)
    compute = ComputeResource(base_speed=build.compute_speed)
    link = BandwidthResource(base_speed=build.link_speed, latency=build.link_latency)
    decode_schedule = _decode_policies(spec, topo, rng_seed=rng_seed).schedule
    prefill = Engine(
        engine_id=topo.prefill_engine_id,
        requests=requests,
        pool=pool,
        memories=topo.memories,
        policies=_prefill_policies(spec, topo, rng_seed=rng_seed),
        compute_res=compute,
        work_per_block=build.work_per_block,
        max_num_seqs=build.max_num_seqs,
        max_num_batched_tokens=build.max_num_batched_tokens,
        enable_chunked_prefill=True,
        write_links=_write_links(topo.memories, build),
        work_per_store=build.work_per_transfer,
        **_prefill_memory_kwargs(spec),
    )
    decode = Engine(
        engine_id=topo.decode_engine_id,
        requests=[],
        pool=pool,
        memories=topo.memories,
        policies=_decode_policies(spec, topo, rng_seed=rng_seed),
        compute_res=compute,
        bandwidth_res=link,
        transfer_links=_transfer_links(
            topo.memories, decode_schedule.pull_sources, link
        ),
        work_per_block=build.work_per_block,
        work_per_transfer=build.work_per_transfer,
        max_num_seqs=build.max_num_seqs,
        max_num_batched_tokens=build.max_num_batched_tokens,
        enable_chunked_prefill=True,
        remote_kv_wait=True,
        **_decode_memory_kwargs(spec),
    )
    return prefill, decode, topo
