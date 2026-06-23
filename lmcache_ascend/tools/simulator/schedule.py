"""Schedule-time policy: block resolution + local-tier eviction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .block_resolve import BlockResolution
from .eviction import EvictionPolicy, LRUEviction
from .memory import Memory
from .plan import BlockActions, EntryPlan, EvictPlan
from .read_path import READ_PATH, ReadPathPolicy, ReadPathStrategy, read_path_from_pull_mode
from .block_resolve import local_satisfied
from .policy_registry import PolicyContext
from .request import Request


def slots_needed(actions: BlockActions) -> int:
    return sum(
        1
        for action in actions.values()
        if action == "compute"
        or (isinstance(action, tuple) and action[0] == "pull")
    )


@dataclass(frozen=True)
class ScheduleConfig:
    local_memory: str
    pull_sources: tuple[str, ...] = ()
    pull_mode: Literal["compute_only", "ordered_pull"] = "compute_only"
    read_path: ReadPathStrategy | None = None
    # Default used only when ScheduleConfig is built directly in tests;
    # EnginePolicies.from_config always sets eviction from the graph.
    local_eviction: EvictionPolicy = field(default_factory=LRUEviction)

    def resolved_read_path(self) -> ReadPathStrategy:
        if self.read_path is not None:
            return self.read_path
        return READ_PATH.create(read_path_from_pull_mode(self.pull_mode))


class SchedulePolicy:
    """Resolve per-block actions and local-tier evictions at admit time."""

    def __init__(self, config: ScheduleConfig):
        self.config = config
        self._read_path = ReadPathPolicy(config.resolved_read_path())

    @property
    def local_memory(self) -> str:
        return self.config.local_memory

    @property
    def pull_sources(self) -> list[str]:
        return list(self.config.pull_sources)

    @property
    def eviction_policy(self) -> EvictionPolicy:
        return self.config.local_eviction

    @property
    def read_path_policy(self) -> ReadPathPolicy:
        return self._read_path

    def lookup(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        block_size: int = 1,
        cost_ctx=None,
        trace_writer=None,
        engine_id: str | None = None,
    ) -> EntryPlan | None:
        del block_size
        actions = self.resolve_actions(
            memories,
            block_hashes,
            allow_compute=allow_compute,
            req=req,
            cost_ctx=cost_ctx,
            trace_writer=trace_writer,
            engine_id=engine_id,
        )
        if actions is None:
            return None
        evicts = self.config.local_eviction.plan(
            memories[self.local_memory], slots_needed(actions), set(block_hashes)
        )
        if evicts is None:
            return None
        return EntryPlan(
            evicts=[EvictPlan(block=block) for block in evicts],
            blocks=dict(actions),
        )

    def resolve_actions(
        self,
        memories: dict[str, Memory],
        block_hashes: list[str],
        *,
        allow_compute: bool = True,
        req: Request | None = None,
        cost_ctx=None,
        trace_writer=None,
        engine_id: str | None = None,
    ) -> BlockActions | None:
        actions: BlockActions = {}
        for block_hash in block_hashes:
            resolution = self.resolve_block(
                memories,
                block_hash,
                allow_compute=allow_compute,
                req=req,
                cost_ctx=cost_ctx,
            )
            if trace_writer is not None and req is not None:
                trace_writer.on_decision(
                    now=cost_ctx.now if cost_ctx is not None else 0.0,
                    engine_id=engine_id or "",
                    req_id=req.req_id,
                    decision=self._read_path.last_decision,
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
        cost_ctx=None,
    ) -> BlockResolution:
        return self._read_path.resolve(
            memories,
            local_memory=self.local_memory,
            pull_sources=self.pull_sources,
            block_hash=block_hash,
            allow_compute=allow_compute,
            req=req,
            cost_ctx=cost_ctx,
        )
