"""Shared block resolution helpers for schedule and read-path policies."""

from __future__ import annotations

from typing import Literal

from simulator.core.memory import Memory
from simulator.runtime.plan import BlockAction
from simulator.core.request import Request

_LOCAL = Literal["local"]
BlockResolution = BlockAction | _LOCAL | None


def local_satisfied(local: Memory, block_hash: str) -> bool:
    return (
        local.best_resident(block_hash) is not None
        or local.inflight_incoming(block_hash) is not None
    )


def remote_wait_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    from simulator.core.kv_content import tier_has_block, tier_has_inflight

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
    from simulator.core.kv_content import tier_has_block

    for src_key in pull_sources:
        src = memories[src_key]
        if tier_has_block(src, req, block_hash):
            return src_key
    return None
