"""Named policy presets and PD engine construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from simulator.runtime.engine import Engine
from simulator.runtime.engine_config import EngineLinks, EngineRuntime, WorkModel
from simulator.policy.config import EngineConfig, LifecycleSpec, PlacementSpec, build_placement_spec
from simulator.core.memory import Memory
from simulator.policy.policies import EnginePolicies
from simulator.core.request import Request
from simulator.core.resource import BandwidthResource, ComputeResource
from simulator.core.resource_models import build_bandwidth_resource, medium_resource_kind
from simulator.runtime.tasks import TaskPool
from simulator.model.layout import (
    DEFAULT_DECODE_IDS,
    DEFAULT_PREFILL_IDS,
    SimResources,
    Topology,
    TOPOLOGY_TIERS,
    build_eviction_map,
    build_topology,
    engine_ids_for_pd,
    tier_keys_for,
)
from simulator.policy.registry import PolicyContext
from simulator.policy.routing import ROUTING, RoutingPolicy
from simulator.runtime.roles import ROLES


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
    "evict_cost_aware": PresetSpec(
        name="evict_cost_aware",
        description=(
            "LMCache PD: same as evict_lru but cost-aware eviction on all tiers"
        ),
        topology="hbm_dram",
        decode_pull=("npu-0:dram",),
        mirror_tiers=("npu-0:dram",),
        mirror_on_forward=False,
        store_on_complete=("npu-0:dram",),
        hold_kv_on_complete=False,
        decode_retain_prefix_cache=True,
        eviction="cost_aware",
    ),
}

EVICTION_PRESET_NAMES: tuple[str, ...] = (
    "evict_lru",
    "evict_fifo",
    "evict_random",
    "evict_lfu",
    "evict_cost_aware",
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
    medium_models: dict[str, str] | None = None
    medium_model_params: dict[str, dict] = field(default_factory=dict)


def _medium_kind_for_tier(
    tier_key: str,
    medium_models: dict[str, str] | None,
) -> str | None:
    if medium_models is None:
        return None
    if tier_key in medium_models:
        return medium_models[tier_key]
    suffix = tier_key.rsplit(":", 1)[-1]
    if suffix in medium_models:
        return medium_models[suffix]
    colon_suffix = f":{suffix}"
    if colon_suffix in medium_models:
        return medium_models[colon_suffix]
    return medium_resource_kind(tier_key)


def _params_for_tier(
    tier_key: str,
    medium_model_params: dict[str, dict],
) -> dict:
    suffix = tier_key.rsplit(":", 1)[-1]
    for key in (tier_key, suffix, f":{suffix}"):
        if key in medium_model_params:
            return dict(medium_model_params[key])
    return {}


def _tier_bandwidth(
    tier_key: str,
    build: EngineBuildConfig,
    *,
    base_speed: float,
    shared: BandwidthResource | None = None,
) -> BandwidthResource:
    kind = _medium_kind_for_tier(tier_key, build.medium_models)
    if kind is None:
        if shared is not None:
            return shared
        return BandwidthResource(base_speed=base_speed, latency=build.link_latency)
    params = _params_for_tier(tier_key, build.medium_model_params)
    return build_bandwidth_resource(
        kind,
        base_speed=base_speed,
        latency=build.link_latency,
        **params,
    )


def _transfer_links(
    memories: dict[str, Memory],
    pull_sources: list[str],
    build: EngineBuildConfig,
    shared_link: BandwidthResource | None = None,
) -> dict[str, BandwidthResource]:
    if build.medium_models is None:
        link = shared_link or BandwidthResource(
            base_speed=build.link_speed,
            latency=build.link_latency,
        )
        return {src: link for src in pull_sources if src in memories}
    return {
        src: _tier_bandwidth(src, build, base_speed=build.link_speed)
        for src in pull_sources
        if src in memories
    }


def _write_links(
    memories: dict[str, Memory],
    cfg: EngineBuildConfig,
) -> dict[str, BandwidthResource]:
    links: dict[str, BandwidthResource] = {}
    if "npu-0:ssd" in memories:
        links["npu-0:ssd"] = _tier_bandwidth(
            "npu-0:ssd",
            cfg,
            base_speed=cfg.ssd_write_speed,
        )
    return links


def build_topology_for_preset(
    spec: PresetSpec,
    resources: SimResources | None = None,
    *,
    prefill_ids: tuple[str, ...] = DEFAULT_PREFILL_IDS,
    decode_ids: tuple[str, ...] = DEFAULT_DECODE_IDS,
    rng_seed: int = 0,
) -> Topology:
    tier_keys = tier_keys_for(prefill_ids, decode_ids, spec.topology)
    eviction_map = build_eviction_map(
        tier_keys,
        local=spec.local_eviction or spec.eviction,
        downstream=spec.downstream_eviction or spec.eviction,
        rng_seed=rng_seed,
    )
    return build_topology(
        spec.topology,
        prefill_ids=prefill_ids,
        decode_ids=decode_ids,
        resources=resources or SimResources.default(
            prefill_ids=prefill_ids, decode_ids=decode_ids
        ),
        eviction_map=eviction_map,
    )


def _make_engine(
    *,
    engine_id: str,
    requests: list[Request],
    pool: TaskPool,
    topo: Topology,
    policies: EnginePolicies,
    build: EngineBuildConfig,
    compute: ComputeResource,
    link: BandwidthResource,
    remote_kv_wait: bool = False,
) -> Engine:
    pull_sources = list(policies.schedule.config.pull_sources)
    work = WorkModel(
        per_block=build.work_per_block,
        per_transfer=build.work_per_transfer if remote_kv_wait else None,
        per_store=build.work_per_transfer if not remote_kv_wait else None,
    )
    if remote_kv_wait:
        links = EngineLinks(
            compute_res=compute,
            bandwidth_res=link,
            transfer_links=_transfer_links(topo.memories, pull_sources, build, link),
        )
    else:
        links = EngineLinks(
            compute_res=compute,
            write_links=_write_links(topo.memories, build),
        )
    return Engine(
        engine_id=engine_id,
        requests=requests,
        pool=pool,
        memories=topo.memories,
        policies=policies,
        links=links,
        work=work,
        runtime=EngineRuntime(
            max_num_seqs=build.max_num_seqs,
            max_num_batched_tokens=build.max_num_batched_tokens,
            enable_chunked_prefill=True,
            remote_kv_wait=remote_kv_wait,
        ),
    )


def build_engines(
    requests: list[Request],
    pool: TaskPool,
    preset: PresetSpec | str,
    *,
    prefill_ids: tuple[str, ...] = DEFAULT_PREFILL_IDS,
    decode_ids: tuple[str, ...] = DEFAULT_DECODE_IDS,
    routing_name: str = "bijection",
    routing_params: dict | None = None,
    cfg: EngineBuildConfig | None = None,
    resources: SimResources | None = None,
    rng_seed: int = 0,
) -> tuple[dict[str, Engine], Topology, RoutingPolicy]:
    spec = PRESETS[preset] if isinstance(preset, str) else preset
    build = cfg or EngineBuildConfig()
    res = resources or build.resources
    topo = build_topology_for_preset(
        spec, res, prefill_ids=prefill_ids, decode_ids=decode_ids, rng_seed=rng_seed
    )
    compute = ComputeResource(base_speed=build.compute_speed)
    link = BandwidthResource(base_speed=build.link_speed, latency=build.link_latency)

    engines: dict[str, Engine] = {}
    prefill_buckets: dict[str, list[Request]] = {
        pid: [] for pid in topo.prefill_ids
    }
    for index, req in enumerate(requests):
        prefill_buckets[topo.prefill_ids[index % len(topo.prefill_ids)]].append(req)

    for role_name, engine_ids in (
        ("prefill", topo.prefill_ids),
        ("decode", topo.decode_ids),
    ):
        profile = ROLES.create(role_name)
        for engine_id in engine_ids:
            if role_name == "prefill":
                lifecycle = LifecycleSpec(
                    hold_kv_on_complete=spec.hold_kv_on_complete,
                    retain_prefix_cache=spec.prefill_retain_prefix_cache,
                    store_on_complete=spec.store_on_complete,
                )
                engine_requests = prefill_buckets[engine_id]
            else:
                lifecycle = LifecycleSpec(
                    retain_prefix_cache=spec.decode_retain_prefix_cache,
                )
                engine_requests = []

            cfg_kwargs: dict = {
                "engine_id": engine_id,
                "local_tier": topo.local_tier[engine_id],
                "pull_mode": profile.pull_mode,
                "placement": _placement_spec(spec),
                "lifecycle": lifecycle,
            }
            if profile.wants_pull_sources:
                cfg_kwargs["pull_sources"] = topo.decode_pull_sources(spec.decode_pull)
                cfg_kwargs["read_path_name"] = spec.decode_read_path
                cfg_kwargs["read_path_params"] = spec.decode_read_path_params

            engine_cfg = EngineConfig(**cfg_kwargs)
            engines[engine_id] = _make_engine(
                engine_id=engine_id,
                requests=engine_requests,
                pool=pool,
                topo=topo,
                policies=EnginePolicies.from_config(engine_cfg, topo.graph),
                build=build,
                compute=compute,
                link=link,
                remote_kv_wait=profile.remote_kv_wait,
            )

    params = dict(routing_params or {})
    if routing_name == "bijection" and "map" not in params and "spawn_map" not in params:
        if len(topo.prefill_ids) == 1 and len(topo.decode_ids) == 1:
            params["map"] = {topo.prefill_ids[0]: topo.decode_ids[0]}
        else:
            params["decode_ids"] = topo.decode_ids
            routing_name = "round_robin"
    if routing_name in ("round_robin", "hash") and "decode_ids" not in params:
        params["decode_ids"] = topo.decode_ids
    routing = ROUTING.create(routing_name, PolicyContext(params=params))
    return engines, topo, routing


def build_pd_engines(
    requests: list[Request],
    pool: TaskPool,
    preset: PresetSpec | str,
    *,
    cfg: EngineBuildConfig | None = None,
    resources: SimResources | None = None,
    rng_seed: int = 0,
) -> tuple[Engine, Engine, Topology]:
    engines, topo, _routing = build_engines(
        requests,
        pool,
        preset,
        cfg=cfg,
        resources=resources,
        rng_seed=rng_seed,
    )
    return (
        engines[topo.prefill_engine_id],
        engines[topo.decode_engine_id],
        topo,
    )
