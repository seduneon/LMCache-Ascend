"""Schedule-time policy: block resolution + local-tier eviction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .eviction import EvictionPolicy, LRUEviction
from .memory import Memory
from .plan import BlockAction, BlockActions, EntryPlan
from .request import Request

_LOCAL = Literal["local"]
BlockResolution = BlockAction | _LOCAL | None


def local_satisfied(local: Memory, block_hash: str) -> bool:
    return (
        local.best_resident(block_hash) is not None
        or local.inflight_incoming(block_hash) is not None
    )


def slots_needed(actions: BlockActions) -> int:
    return sum(
        1
        for action in actions.values()
        if action == "compute"
        or (isinstance(action, tuple) and action[0] == "pull")
    )


def remote_wait_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    from .kv_content import tier_has_block, tier_has_inflight

    for src_key in pull_sources:
        src = memories[src_key]
        if tier_has_block(src, req, block_hash):
            continue
        if tier_has_inflight(src, req, block_hash):
            return src_key
    return None


def first_resident_pull_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    from .kv_content import tier_has_block

    for src_key in pull_sources:
        src = memories[src_key]
        if tier_has_block(src, req, block_hash):
            return src_key
    return None


@dataclass(frozen=True)
class ScheduleConfig:
    local_memory: str
    pull_sources: tuple[str, ...] = ()
    pull_mode: Literal["compute_only", "ordered_pull"] = "compute_only"
    local_eviction: EvictionPolicy = field(default_factory=LRUEviction)


class SchedulePolicy:
    """Resolve per-block actions and HBM evictions at admit time."""

    def __init__(self, config: ScheduleConfig):
        self.config = config

    @property
    def local_memory(self) -> str:
        return self.config.local_memory

    @property
    def pull_sources(self) -> list[str]:
        return list(self.config.pull_sources)

    @property
    def eviction_policy(self) -> EvictionPolicy:
        return self.config.local_eviction

    def lookup(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
    ) -> EntryPlan | None:
        del block_size
        actions = self.resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
        )
        if actions is None:
            return None
        evicts = self.config.local_eviction.plan(
            memories[self.local_memory], slots_needed(actions), set(block_hashes)
        )
        if evicts is None:
            return None
        return EntryPlan(evicts=list(evicts), blocks=dict(actions))

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
    ) -> BlockActions | None:
        actions: BlockActions = {}
        for block_hash in block_hashes:
            resolution = self.resolve_block(
                memories,
                block_hash,
                allow_compute=allow_compute,
                req=req,
            )
            if resolution is None:
                return None
            if resolution == "local":
                continue
            actions[block_hash] = resolution
        return actions

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"

        if self.config.pull_mode == "compute_only":
            if allow_compute:
                return "compute"
            return None

        if remote_wait_source(memories, self.pull_sources, block_hash, req=req) is not None:
            return "wait"

        src_key = first_resident_pull_source(
            memories, self.pull_sources, block_hash, req=req
        )
        if src_key is not None:
            return ("pull", src_key)

        if allow_compute:
            return "compute"
        return None
