"""Engine policy bundle: schedule + execute plugins."""

from __future__ import annotations

from dataclasses import dataclass

from .effects import EffectConfig, EffectPolicy
from .config import EngineConfig, LifecycleSpec, build_placement_spec
from .read_path import ReadPathPolicy, ReadPathStrategy
from .eviction import LRUEviction
from simulator.runtime.plan import EntryPlan, WorkEntry
from simulator.core.request import Request, request_owning_prefix_block
from .schedule import ScheduleConfig, SchedulePolicy
from simulator.model.tier import Tier, TierGraph
from simulator.core.memory import Memory


@dataclass
class EnginePolicies:
    schedule: SchedulePolicy
    effects: EffectPolicy
    config: EngineConfig

    @classmethod
    def from_config(cls, config: EngineConfig, graph: TierGraph) -> EnginePolicies:
        local_eviction = graph.eviction_for(config.local_tier)
        read_path = config.build_read_path_strategy()
        return cls(
            SchedulePolicy(
                ScheduleConfig(
                    local_memory=config.local_tier,
                    pull_sources=config.pull_sources,
                    pull_mode=config.pull_mode,
                    read_path=read_path,
                    local_eviction=local_eviction,
                )
            ),
            EffectPolicy(
                EffectConfig(
                    local_memory=config.local_tier,
                    graph=graph,
                    placement=config.placement,
                )
            ),
            config,
        )

    @classmethod
    def with_graph(cls, policies: EnginePolicies, graph: TierGraph) -> EnginePolicies:
        return cls.from_config(policies.config, graph)

    @classmethod
    def compute_only(
        cls,
        local_memory: str,
        *,
        graph: TierGraph | None = None,
        local_eviction=None,
        mirror_tiers: tuple[str, ...] = (),
        async_write_tiers: frozenset[str] | None = None,
        mirror_on_forward: bool = True,
        retention: str = "unbounded",
        retention_params: dict | None = None,
        hold_kv_on_complete: bool = False,
        retain_prefix_cache: bool = False,
        store_on_complete: tuple[str, ...] = (),
    ) -> EnginePolicies:
        g = graph or TierGraph(tiers={})
        if local_memory not in g.tiers and local_eviction is not None:
            g = TierGraph(
                tiers={
                    **g.tiers,
                    local_memory: Tier(
                        local_memory,
                        Memory(size=1, name=local_memory),
                        local_eviction,
                    ),
                }
            )
        placement = build_placement_spec(
            mirror_tiers=mirror_tiers,
            async_write_tiers=async_write_tiers or frozenset(),
            mirror_on_forward=mirror_on_forward,
            retention_name=retention,
            retention_params=retention_params,
        )
        return cls.from_config(
            EngineConfig(
                engine_id="e0",
                local_tier=local_memory,
                pull_mode="compute_only",
                placement=placement,
                lifecycle=LifecycleSpec(
                    hold_kv_on_complete=hold_kv_on_complete,
                    retain_prefix_cache=retain_prefix_cache,
                    store_on_complete=store_on_complete,
                ),
            ),
            g,
        )

    @classmethod
    def ordered_pull(
        cls,
        local_memory: str,
        pull_sources: tuple[str, ...] | list[str],
        *,
        graph: TierGraph | None = None,
        local_eviction=None,
        mirror_tiers: tuple[str, ...] = (),
        async_write_tiers: frozenset[str] | None = None,
        mirror_on_forward: bool = True,
        retention: str = "unbounded",
        retention_params: dict | None = None,
        hold_kv_on_complete: bool = False,
        retain_prefix_cache: bool = False,
        store_on_complete: tuple[str, ...] = (),
    ) -> EnginePolicies:
        g = graph or TierGraph(tiers={})
        placement = build_placement_spec(
            mirror_tiers=mirror_tiers,
            async_write_tiers=async_write_tiers or frozenset(),
            mirror_on_forward=mirror_on_forward,
            retention_name=retention,
            retention_params=retention_params,
        )
        return cls.from_config(
            EngineConfig(
                engine_id="e0",
                local_tier=local_memory,
                pull_sources=tuple(pull_sources),
                pull_mode="ordered_pull",
                placement=placement,
                lifecycle=LifecycleSpec(
                    hold_kv_on_complete=hold_kv_on_complete,
                    retain_prefix_cache=retain_prefix_cache,
                    store_on_complete=store_on_complete,
                ),
            ),
            g,
        )


def enrich_entry_plan(
    plan: EntryPlan,
    *,
    entry: WorkEntry,
    effects: EffectPolicy,
    memories: dict[str, Memory],
    known_requests: list[Request],
) -> None:
    """Plan-time expansion of async store and spill ops."""

    for block_hash in entry.block_hashes:
        action = plan.blocks.get(block_hash)
        needs_store = action == "compute" or (
            isinstance(action, tuple) and action[0] == "pull"
        )
        if not needs_store:
            continue
        ops = effects.plan_async_stores(memories, block_hash, entry.req)
        if ops:
            plan.store_ops[block_hash] = ops

    for evict_plan in plan.evicts:
        spill_req = request_owning_prefix_block(evict_plan.block.hash, known_requests)
        if spill_req is None:
            continue
        ops = effects.plan_spill_stores(memories, evict_plan.block.hash, spill_req)
        if ops:
            evict_plan.store_ops.extend(ops)
