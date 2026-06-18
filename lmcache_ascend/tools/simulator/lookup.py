"""Compatibility shims — prefer ``schedule.SchedulePolicy`` and ``policies.EnginePolicies``."""

from __future__ import annotations

from .eviction import EvictionPolicy, LRUEviction
from .schedule import (
    BlockResolution,
    ScheduleConfig,
    SchedulePolicy,
    first_resident_pull_source,
    local_satisfied,
    remote_wait_source,
    slots_needed,
)


class ComputeOnlyLookupPolicy(SchedulePolicy):
    def __init__(
        self,
        local_memory: str,
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(
            ScheduleConfig(
                local_memory=local_memory,
                hbm_eviction=eviction_policy or LRUEviction(),
            )
        )


class OrderedPullLookupPolicy(SchedulePolicy):
    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str],
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(
            ScheduleConfig(
                local_memory=local_memory,
                pull_sources=tuple(pull_sources),
                pull_mode="ordered_pull",
                hbm_eviction=eviction_policy or LRUEviction(),
            )
        )


class LookupPolicy(SchedulePolicy):
    """Deprecated alias."""


__all__ = [
    "BlockResolution",
    "ComputeOnlyLookupPolicy",
    "LookupPolicy",
    "OrderedPullLookupPolicy",
    "SchedulePolicy",
    "first_resident_pull_source",
    "local_satisfied",
    "remote_wait_source",
    "slots_needed",
]
