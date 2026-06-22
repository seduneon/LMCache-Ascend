"""Read-path policies: pull vs recompute vs wait."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .estimate import BlockCostEstimate, CostContext
from .memory import Memory
from .request import Request
from .block_resolve import (
    BlockResolution,
    first_resident_pull_source,
    local_satisfied,
    remote_wait_source,
)

ReadPathKind = Literal[
    "ordered_pull",
    "compute_only",
    "min_cost",
    "min_cost_with_wait",
    "threshold",
]


@dataclass(frozen=True)
class ReadPathSpec:
    kind: ReadPathKind = "ordered_pull"
    threshold_ratio: float = 1.0


@dataclass(frozen=True)
class ReadPathDecision:
    """Provenance for tracing and analysis."""

    block_hash: str
    chosen: str
    reason: str
    estimate: BlockCostEstimate | None = None


def read_path_from_pull_mode(
    pull_mode: Literal["compute_only", "ordered_pull"],
) -> ReadPathSpec:
    if pull_mode == "compute_only":
        return ReadPathSpec(kind="compute_only")
    return ReadPathSpec(kind="ordered_pull")


class ReadPathPolicy:
    def __init__(self, spec: ReadPathSpec):
        self.spec = spec
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
        local = memories[local_memory]
        if local_satisfied(local, block_hash):
            self.last_decision = ReadPathDecision(block_hash, "local", "resident")
            return "local"

        kind = self.spec.kind
        if kind == "compute_only":
            if allow_compute:
                self.last_decision = ReadPathDecision(block_hash, "compute", "compute_only")
                return "compute"
            self.last_decision = None
            return None

        if remote_wait_source(memories, pull_sources, block_hash, req=req) is not None:
            if kind != "min_cost_with_wait":
                self.last_decision = ReadPathDecision(block_hash, "wait", "inflight_remote")
                return "wait"

        if kind == "ordered_pull":
            src_key = first_resident_pull_source(
                memories, pull_sources, block_hash, req=req
            )
            if src_key is not None:
                self.last_decision = ReadPathDecision(
                    block_hash, f"pull:{src_key}", "ordered_pull"
                )
                return ("pull", src_key)
            if allow_compute:
                self.last_decision = ReadPathDecision(block_hash, "compute", "fallback")
                return "compute"
            self.last_decision = None
            return None

        if cost_ctx is None or req is None:
            return self._ordered_pull_fallback(
                memories, pull_sources, block_hash, allow_compute, req
            )

        estimate = cost_ctx.estimate_block(pull_sources, block_hash, req=req)

        if kind == "min_cost_with_wait" and estimate.wait:
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
                self.last_decision = ReadPathDecision(
                    block_hash, chosen_label, kind, estimate
                )
                return action
            self.last_decision = ReadPathDecision(block_hash, "wait", kind, estimate)
            return "wait"

        if kind in ("min_cost", "min_cost_with_wait"):
            return self._resolve_min_cost(
                pull_sources,
                block_hash,
                allow_compute,
                req,
                estimate,
                reason=kind,
            )

        if kind == "threshold":
            return self._resolve_threshold(
                pull_sources,
                block_hash,
                allow_compute,
                estimate,
            )

        return self._ordered_pull_fallback(
            memories, pull_sources, block_hash, allow_compute, req
        )

    def _resolve_min_cost(
        self,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request,
        estimate: BlockCostEstimate,
        *,
        reason: str,
    ) -> BlockResolution:
        del req
        candidates: list[tuple[str, float, BlockResolution]] = []
        for src in pull_sources:
            if src in estimate.pull:
                candidates.append((f"pull:{src}", estimate.pull[src], ("pull", src)))
        if allow_compute:
            candidates.append(("compute", estimate.compute, "compute"))
        if not candidates:
            self.last_decision = None
            return None
        chosen_label, _, action = min(candidates, key=lambda c: (c[1], c[0]))
        self.last_decision = ReadPathDecision(
            block_hash, chosen_label, reason, estimate
        )
        return action

    def _resolve_threshold(
        self,
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        estimate: BlockCostEstimate,
    ) -> BlockResolution:
        ratio = self.spec.threshold_ratio
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
                    self.last_decision = ReadPathDecision(
                        block_hash,
                        f"pull:{best_src}",
                        "threshold",
                        estimate,
                    )
                    return ("pull", best_src)
        if allow_compute:
            self.last_decision = ReadPathDecision(
                block_hash, "compute", "threshold", estimate
            )
            return "compute"
        self.last_decision = None
        return None

    def _ordered_pull_fallback(
        self,
        memories: dict[str, Memory],
        pull_sources: list[str],
        block_hash: str,
        allow_compute: bool,
        req: Request | None,
    ) -> BlockResolution:
        src_key = first_resident_pull_source(
            memories, pull_sources, block_hash, req=req
        )
        if src_key is not None:
            self.last_decision = ReadPathDecision(
                block_hash, f"pull:{src_key}", "ordered_pull_fallback"
            )
            return ("pull", src_key)
        if allow_compute:
            self.last_decision = ReadPathDecision(block_hash, "compute", "fallback")
            return "compute"
        self.last_decision = None
        return None
