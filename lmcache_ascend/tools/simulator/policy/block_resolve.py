"""Shared block resolution helpers for schedule and read-path policies."""

from __future__ import annotations

from typing import Literal

from simulator.core.memory import Memory
from simulator.runtime.plan import BlockAction
from simulator.core.request import Request, RequestPD

_LOCAL = Literal["local"]
BlockResolution = BlockAction | _LOCAL | None


def local_satisfied(local: Memory, block_hash: str) -> bool:
    return (
        local.best_resident(block_hash) is not None
        or local.inflight_incoming(block_hash) is not None
    )


def pull_sources_for_request(
    pull_sources: list[str],
    req: Request | None,
) -> list[str]:
    """Prefer the paired prefill HBM for PD decode; keep downstream tier order."""
    if req is None or req.pd != RequestPD.DECODE or not req.prefill_engine_id:
        return pull_sources
    preferred = f"{req.prefill_engine_id}:hbm"
    if preferred not in pull_sources:
        return pull_sources
    return [preferred] + [src for src in pull_sources if src != preferred]


def skip_foreign_prefill_wait(req: Request | None, src_key: str) -> bool:
    """Do not wait on another prefill engine's HBM in-flight reservation."""
    if req is None or req.pd != RequestPD.DECODE or not req.prefill_engine_id:
        return False
    if not src_key.endswith(":hbm"):
        return False
    return not src_key.startswith(f"{req.prefill_engine_id}:")


def remote_wait_source(
    memories: dict[str, Memory],
    pull_sources: list[str],
    block_hash: str,
    *,
    req: Request | None = None,
) -> str | None:
    from simulator.core.kv_content import tier_has_block, tier_has_inflight

    for src_key in pull_sources_for_request(pull_sources, req):
        if skip_foreign_prefill_wait(req, src_key):
            continue
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

    for src_key in pull_sources_for_request(pull_sources, req):
        src = memories[src_key]
        if tier_has_block(src, req, block_hash):
            return src_key
    return None
