"""Engine policy bundle: schedule + execute plugins."""

from __future__ import annotations

from dataclasses import dataclass

from .effects import EffectConfig, EffectPolicy
from .engine_config import EngineConfig, build_placement_spec
from .read_path import ReadPathSpec
from .eviction import LRUEviction
from .plan import EntryPlan, WorkEntry
from .request import Request, request_owning_prefix_block
from .schedule import ScheduleConfig, SchedulePolicy
from .tier import Tier, TierGraph


@dataclass
class EnginePolicies:
    schedule: SchedulePolicy
    effects: EffectPolicy

    @classmethod
    def from_config(cls, config: EngineConfig, graph: TierGraph) -> EnginePolicies:
        local_eviction = graph.eviction_for(config.local_tier)
        return cls(
            SchedulePolicy(
                ScheduleConfig(
                    local_memory=config.local_tier,
                    pull_sources=config.pull_sources,
                    pull_mode=config.pull_mode,
                    read_path=config.read_path,
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
        )

    @classmethod
    def with_graph(cls, policies: EnginePolicies, graph: TierGraph) -> EnginePolicies:
        schedule_cfg = policies.schedule.config
        placement = policies.effects.config.placement
        return cls.from_config(
            EngineConfig(
                engine_id="e0",
                local_tier=schedule_cfg.local_memory,
                pull_sources=schedule_cfg.pull_sources,
                pull_mode=schedule_cfg.pull_mode,
                read_path=schedule_cfg.read_path,
                placement=placement,
            ),
            graph,
        )

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
    ) -> EnginePolicies:
        g = graph or TierGraph(tiers={})
        if local_memory not in g.tiers and local_eviction is not None:
            from .memory import Memory
            from .tier import Tier

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
    from .memory import Memory

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

    for victim in plan.evicts:
        spill_req = request_owning_prefix_block(victim.hash, known_requests)
        if spill_req is None:
            continue
        ops = effects.plan_spill_stores(memories, victim.hash, spill_req)
        if ops:
            plan.spill_store_ops[id(victim)] = ops
