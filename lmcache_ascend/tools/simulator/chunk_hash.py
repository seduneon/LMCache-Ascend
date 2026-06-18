"""LMCache-aligned chunk keys: coarse tier slots cover N consecutive HBM block hashes."""

from __future__ import annotations

from .content_key import ContentKey, lmcache_chunk_hash
from .memory import Memory
from .request import Request


def chunk_hbm_group(ordered_hbm: list[str], block_hash: str, chunk_blocks: int) -> list[str]:
    """``chunk_blocks`` consecutive prefix blocks covering ``block_hash``."""
    if chunk_blocks <= 1:
        return [block_hash]
    try:
        idx = ordered_hbm.index(block_hash)
    except ValueError:
        return [block_hash]
    start = (idx // chunk_blocks) * chunk_blocks
    return ordered_hbm[start : start + chunk_blocks]


def chunk_key_for_hbm_block(
    req: Request | None,
    block_hash: str,
    chunk_blocks: int,
) -> str:
    """Storage key on a tier with ``chunk_blocks`` HBM blocks per slot."""
    if chunk_blocks <= 1:
        return block_hash
    if req is None:
        return lmcache_chunk_hash([block_hash])
    prefix = req.block_hashes[: req.prefix_block_count]
    if block_hash not in prefix:
        return lmcache_chunk_hash([block_hash])
    return lmcache_chunk_hash(chunk_hbm_group(prefix, block_hash, chunk_blocks))


def tier_covers_hbm_block(mem: Memory, req: Request | None, block_hash: str) -> bool:
    key = chunk_key_for_hbm_block(req, block_hash, mem.chunk_blocks)
    return mem.best_resident(key) is not None


def tier_inflight_hbm_block(mem: Memory, req: Request | None, block_hash: str) -> bool:
    key = chunk_key_for_hbm_block(req, block_hash, mem.chunk_blocks)
    return mem.inflight_incoming(key) is not None


def transfer_work_units(mem: Memory, req: Request | None, block_hash: str) -> int:
    """Pull work multiplier: one unit per HBM block covered by the source chunk."""
    if mem.chunk_blocks <= 1 or req is None:
        return 1
    prefix = req.block_hashes[: req.prefix_block_count]
    if block_hash not in prefix:
        return 1
    return len(chunk_hbm_group(prefix, block_hash, mem.chunk_blocks))


def chunk_transfer_work(mem: Memory, req: Request | None, block_hashes: list[str]) -> float:
    """Transfer work for a batch of HBM blocks pulled/stored together (latency once)."""
    if not block_hashes:
        return 0.0
    return float(transfer_work_units(mem, req, block_hashes[0]))


def pull_dedupe_key(
    src_key: str,
    req: Request | None,
    block_hash: str,
    memories: dict[str, Memory],
) -> tuple[str, ContentKey]:
    """Shared pull identity: one transfer per (source tier, logical content)."""
    src = memories[src_key]
    storage = chunk_key_for_hbm_block(req, block_hash, src.chunk_blocks)
    return (src_key, ContentKey.for_storage_key(storage))


def group_pull_blocks(
    block_hashes: list[str],
    actions: dict,
    req: Request | None,
    memories: dict[str, Memory],
) -> list[tuple[str, list[str]]]:
    """Group consecutive blocks with the same pull source into chunk-aligned ranges."""
    groups: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(block_hashes):
        block_hash = block_hashes[i]
        action = actions.get(block_hash)
        if not isinstance(action, tuple) or action[0] != "pull":
            i += 1
            continue
        src_key = action[1]
        src_mem = memories[src_key]
        chunk = (
            chunk_hbm_group(
                req.block_hashes[: req.prefix_block_count] if req else [block_hash],
                block_hash,
                src_mem.chunk_blocks,
            )
            if req is not None
            else [block_hash]
        )
        group = [block_hash]
        j = i + 1
        while j < len(block_hashes):
            nxt = block_hashes[j]
            nxt_action = actions.get(nxt)
            if (
                isinstance(nxt_action, tuple)
                and nxt_action[0] == "pull"
                and nxt_action[1] == src_key
                and nxt in chunk
            ):
                group.append(nxt)
                j += 1
            else:
                break
        groups.append((src_key, group))
        i = j
    return groups
