"""Read-path policies: pull vs recompute vs wait."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from simulator.observability.estimate import BlockCostEstimate, CostContext
from simulator.core.memory import Memory
from simulator.core.request import Request
from .block_resolve import (
    BlockResolution,
    first_resident_pull_source,
    local_satisfied,
    remote_wait_source,
)
from .registry import PolicyContext, Registry

READ_PATH = Registry["ReadPathStrategy"]("read_path")

ReadPathKind = Literal[
    "ordered_pull",
    "compute_only",
    "min_cost",
    "min_cost_with_wait",
    "threshold",
]


@dataclass(frozen=True)
class ReadPathDecision:
    """Provenance for tracing and analysis."""

    block_hash: str
    chosen: str
    reason: str
    estimate: BlockCostEstimate | None = None


def read_path_from_pull_mode(
    pull_mode: Literal["compute_only", "ordered_pull"],
) -> str:
    if pull_mode == "compute_only":
        return "compute_only"
    return "ordered_pull"


def ordered_pull_fallback(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    allow_compute: bool,
    req: Request | None,
) -> tuple[BlockResolution, ReadPathDecision | None]:
    src_key = first_resident_pull_source(
        memories, pull_sources, block_hash, req=req
    )
    if src_key is not None:
        return ("pull", src_key), ReadPathDecision(
            block_hash, f"pull:{src_key}", "ordered_pull_fallback"
        )
    if allow_compute:
        return "compute", ReadPathDecision(block_hash, "compute", "fallback")
    return None, None


class ReadPathStrategy(ABC):
    """Resolve pull vs compute vs wait for one block."""

    skip_remote_wait_gate: bool = False

    def resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        local = memories[local_memory]
        if local_satisfied(local, block_hash):
            return "local", ReadPathDecision(block_hash, "local", "resident")

        if not self.skip_remote_wait_gate:
            if remote_wait_source(memories, pull_sources, block_hash, req=req) is not None:
                return "wait", ReadPathDecision(
                    block_hash, "wait", "inflight_remote"
                )

        return self._resolve(
            memories,
            local_memory=local_memory,
            pull_sources=pull_sources,
            block_hash=block_hash,
            allow_compute=allow_compute,
            req=req,
            cost_ctx=cost_ctx,
        )

    @abstractmethod
    def _resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        pass


class ComputeOnlyStrategy(ReadPathStrategy):
    skip_remote_wait_gate = True

    def _resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        del memories, local_memory, pull_sources, req, cost_ctx
        if allow_compute:
            return "compute", ReadPathDecision(block_hash, "compute", "compute_only")
        return None, None


class OrderedPullStrategy(ReadPathStrategy):
    def _resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        del local_memory, cost_ctx
        src_key = first_resident_pull_source(
            memories, pull_sources, block_hash, req=req
        )
        if src_key is not None:
            return ("pull", src_key), ReadPathDecision(
                block_hash, f"pull:{src_key}", "ordered_pull"
            )
        if allow_compute:
            return "compute", ReadPathDecision(block_hash, "compute", "fallback")
        return None, None


def _resolve_min_cost(
    pull_sources: list[str],
    block_hash: str,
    allow_compute: bool,
    estimate: BlockCostEstimate,
    *,
    reason: str,
) -> tuple[BlockResolution, ReadPathDecision | None]:
    candidates: list[tuple[str, float, BlockResolution]] = []
    for src in pull_sources:
        if src in estimate.pull:
            candidates.append((f"pull:{src}", estimate.pull[src], ("pull", src)))
    if allow_compute:
        candidates.append(("compute", estimate.compute, "compute"))
    if not candidates:
        return None, None
    chosen_label, _, action = min(candidates, key=lambda c: (c[1], c[0]))
    return action, ReadPathDecision(block_hash, chosen_label, reason, estimate)


class MinCostStrategy(ReadPathStrategy):
    def _resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        if cost_ctx is None or req is None:
            return ordered_pull_fallback(
                memories, pull_sources, block_hash, allow_compute, req
            )
        estimate = cost_ctx.estimate_block(pull_sources, block_hash, req=req)
        return _resolve_min_cost(
            pull_sources,
            block_hash,
            allow_compute,
            estimate,
            reason="min_cost",
        )


class MinCostWithWaitStrategy(ReadPathStrategy):
    skip_remote_wait_gate = True

    def _resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        if cost_ctx is None or req is None:
            return ordered_pull_fallback(
                memories, pull_sources, block_hash, allow_compute, req
            )
        estimate = cost_ctx.estimate_block(pull_sources, block_hash, req=req)
        if estimate.wait:
            best_wait = min(estimate.wait.items(), key=lambda kv: (kv[1], kv[0]))
            if allow_compute or estimate.pull:
                candidates: list[tuple[str, float, BlockResolution]] = []
                for src, t in estimate.pull.items():
                    candidates.append((f"pull:{src}", t, ("pull", src)))
                if allow_compute:
                    candidates.append(("compute", estimate.compute, "compute"))
                candidates.append((f"wait:{best_wait[0]}", best_wait[1], "wait"))
                chosen_label, _, action = min(
                    candidates, key=lambda c: (c[1], c[0])
                )
                return action, ReadPathDecision(
                    block_hash, chosen_label, "min_cost_with_wait", estimate
                )
            return "wait", ReadPathDecision(
                block_hash, "wait", "min_cost_with_wait", estimate
            )
        return _resolve_min_cost(
            pull_sources,
            block_hash,
            allow_compute,
            estimate,
            reason="min_cost_with_wait",
        )


class ThresholdStrategy(ReadPathStrategy):
    def __init__(self, threshold_ratio: float = 1.0):
        self.threshold_ratio = threshold_ratio

    def _resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> tuple[BlockResolution, ReadPathDecision | None]:
        if cost_ctx is None or req is None:
            return ordered_pull_fallback(
                memories, pull_sources, block_hash, allow_compute, req
            )
        estimate = cost_ctx.estimate_block(pull_sources, block_hash, req=req)
        ratio = self.threshold_ratio
        if estimate.pull:
            best_src = min(
                pull_sources,
                key=lambda s: (
                    estimate.pull.get(s, float("inf")),
                    s,
                ),
            )
            if best_src in estimate.pull:
                t_pull = estimate.pull[best_src]
                if t_pull < ratio * estimate.compute:
                    return ("pull", best_src), ReadPathDecision(
                        block_hash,
                        f"pull:{best_src}",
                        "threshold",
                        estimate,
                    )
        if allow_compute:
            return "compute", ReadPathDecision(
                block_hash, "compute", "threshold", estimate
            )
        return None, None


class ReadPathPolicy:
    """Thin wrapper holding last decision for tracing."""

    def __init__(self, strategy: ReadPathStrategy):
        self.strategy = strategy
        self.last_decision: ReadPathDecision | None = None

    def resolve(
        self,
        memories: dict[str, Memory],
        *,
        local_memory: str,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
        cost_ctx: CostContext | None,
    ) -> BlockResolution:
        action, decision = self.strategy.resolve(
            memories,
            local_memory=local_memory,
            pull_sources=pull_sources,
            block_hash=block_hash,
            allow_compute=allow_compute,
            req=req,
            cost_ctx=cost_ctx,
        )
        self.last_decision = decision
        return action


@READ_PATH.register("ordered_pull")
def _ordered_pull(_ctx: PolicyContext) -> ReadPathStrategy:
    return OrderedPullStrategy()


@READ_PATH.register("compute_only")
def _compute_only(_ctx: PolicyContext) -> ReadPathStrategy:
    return ComputeOnlyStrategy()


@READ_PATH.register("min_cost")
def _min_cost(_ctx: PolicyContext) -> ReadPathStrategy:
    return MinCostStrategy()


@READ_PATH.register("min_cost_with_wait")
def _min_cost_with_wait(_ctx: PolicyContext) -> ReadPathStrategy:
    return MinCostWithWaitStrategy()


@READ_PATH.register("threshold")
def _threshold(ctx: PolicyContext) -> ReadPathStrategy:
    ratio = float(ctx.params.get("threshold_ratio", 1.0))
    return ThresholdStrategy(threshold_ratio=ratio)


# Keep ``ReadPathKind`` in sync with registered strategy names (fail fast on drift).
assert frozenset(READ_PATH.names()) == frozenset(
    ("compute_only", "min_cost", "min_cost_with_wait", "ordered_pull", "threshold")
)
