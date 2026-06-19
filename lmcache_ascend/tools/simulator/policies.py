"""Engine policy bundle: schedule + execute plugins."""

from __future__ import annotations

from dataclasses import dataclass

from .eviction import EvictionPolicy, LRUEviction
from .effects import EffectConfig, EffectPolicy
from .plan import EntryPlan, WorkEntry
from .request import Request, request_owning_prefix_block
from .schedule import ScheduleConfig, SchedulePolicy


@dataclass
class EnginePolicies:
    schedule: SchedulePolicy
    effects: EffectPolicy

    @classmethod
    def compute_only(
        cls,
        local_memory: str,
        *,
        hbm_eviction: EvictionPolicy | None = None,
        mirror_tiers: tuple[str, ...] = (),
        async_write_tiers: frozenset[str] | None = None,
        mirror_on_forward: bool = True,
        tier_eviction: dict[str, EvictionPolicy] | None = None,
        retention: str = "unbounded",
        global_cap_max: int = 0,
        global_cap_tiers: tuple[str, ...] = (),
        global_cap_per_tier: int | None = 1,
    ) -> EnginePolicies:
        return cls(
            SchedulePolicy(
                ScheduleConfig(
                    local_memory=local_memory,
                    hbm_eviction=hbm_eviction or LRUEviction(),
                )
            ),
            EffectPolicy(
                EffectConfig(
                    local_memory=local_memory,
                    mirror_tiers=mirror_tiers,
                    async_write_tiers=async_write_tiers or frozenset(),
                    mirror_on_forward=mirror_on_forward,
                    tier_eviction=tier_eviction or {},
                    retention=retention,  # type: ignore[arg-type]
                    global_cap_max=global_cap_max,
                    global_cap_tiers=global_cap_tiers,
                    global_cap_per_tier=global_cap_per_tier,
                )
            ),
        )

    @classmethod
    def ordered_pull(
        cls,
        local_memory: str,
        pull_sources: tuple[str, ...] | list[str],
        *,
        hbm_eviction: EvictionPolicy | None = None,
        mirror_tiers: tuple[str, ...] = (),
        async_write_tiers: frozenset[str] | None = None,
        mirror_on_forward: bool = True,
        tier_eviction: dict[str, EvictionPolicy] | None = None,
        retention: str = "unbounded",
        global_cap_max: int = 0,
        global_cap_tiers: tuple[str, ...] = (),
        global_cap_per_tier: int | None = 1,
    ) -> EnginePolicies:
        return cls(
            SchedulePolicy(
                ScheduleConfig(
                    local_memory=local_memory,
                    pull_sources=tuple(pull_sources),
                    pull_mode="ordered_pull",
                    hbm_eviction=hbm_eviction or LRUEviction(),
                )
            ),
            EffectPolicy(
                EffectConfig(
                    local_memory=local_memory,
                    mirror_tiers=mirror_tiers,
                    async_write_tiers=async_write_tiers or frozenset(),
                    mirror_on_forward=mirror_on_forward,
                    tier_eviction=tier_eviction or {},
                    retention=retention,  # type: ignore[arg-type]
                    global_cap_max=global_cap_max,
                    global_cap_tiers=global_cap_tiers,
                    global_cap_per_tier=global_cap_per_tier,
                )
            ),
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

    for victim in plan.evicts:
        spill_req = request_owning_prefix_block(victim.hash, known_requests)
        if spill_req is None:
            continue
        ops = effects.plan_spill_stores(memories, victim.hash, spill_req)
        if ops:
            plan.spill_store_ops[id(victim)] = ops
