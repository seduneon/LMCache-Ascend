"""Lookup policies: resolve per-block compute, pull, or local hit."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    """Pull source with in-flight chunk (resident copy not ready yet)."""
    from .chunk_hash import tier_covers_hbm_block, tier_inflight_hbm_block

    for src_key in pull_sources:
        src = memories[src_key]
        if tier_covers_hbm_block(src, req, block_hash):
            continue
        if tier_inflight_hbm_block(src, req, block_hash):
            return src_key
    return None


def first_resident_pull_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    from .chunk_hash import tier_covers_hbm_block

    for src_key in pull_sources:
        src = memories[src_key]
        if tier_covers_hbm_block(src, req, block_hash):
            return src_key
    return None


class LookupPolicy(ABC):
    """Resolve per-block actions: local hit, pull from a tier, or recompute."""

    def __init__(
        self,
        local_memory: str,
        eviction_policy: EvictionPolicy | None = None,
    ):
        self.local_memory = local_memory
        self.eviction_policy = eviction_policy or LRUEviction()

    @property
    @abstractmethod
    def pull_sources(self) -> list[str]:
        pass

    @abstractmethod
    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        pass

    def lookup(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
    ) -> EntryPlan | None:
        """Try to allocate ``block_hashes`` without preemption (SSOT for actions + evicts)."""
        actions = self.resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            block_size=block_size,
        )
        if actions is None:
            return None
        evicts = self.eviction_policy.plan(
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
        block_size: int = 1,
    ) -> BlockActions | None:
        actions: BlockActions = {}
        for block_hash in block_hashes:
            resolution = self.resolve_block(
                memories,
                block_hash,
                allow_compute=allow_compute,
                req=req,
                block_size=block_size,
            )
            if resolution is None:
                return None
            if resolution == "local":
                continue
            actions[block_hash] = resolution
        return actions


class ComputeOnlyLookupPolicy(LookupPolicy):
    """Local hit or recompute. No remote tiers."""

    @property
    def pull_sources(self) -> list[str]:
        return []

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"
        if allow_compute:
            return "compute"
        return None


class OrderedPullLookupPolicy(LookupPolicy):
    """First ``pull_sources`` entry with a resident copy, else recompute."""

    def __init__(
        self,
        local_memory: str,
        pull_sources: list[str],
        eviction_policy: EvictionPolicy | None = None,
    ):
        super().__init__(local_memory, eviction_policy)
        self._pull_sources = list(pull_sources)

    @property
    def pull_sources(self) -> list[str]:
        return self._pull_sources

    def resolve_block(
        self,
        memories: dict[str, Memory],
        block_hash: str,
        *,
        allow_compute: bool,
        req: Request | None = None,
        block_size: int = 1,
    ) -> BlockResolution:
        local = memories[self.local_memory]
        if local_satisfied(local, block_hash):
            return "local"

        if remote_wait_source(memories, self._pull_sources, block_hash, req=req) is not None:
            return "wait"

        src_key = first_resident_pull_source(
            memories, self._pull_sources, block_hash, req=req
        )
        if src_key is not None:
            return ("pull", src_key)

        if allow_compute:
            return "compute"
        return None
