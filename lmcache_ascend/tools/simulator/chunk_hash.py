"""LMCache-aligned chunk keys: coarse tier slots cover N consecutive HBM block hashes."""

from __future__ import annotations

from memory import Memory
from request import Request


def lmcache_chunk_hash(hbm_hashes: list[str]) -> str:
    """Deterministic chunk content id (LMCache ``chunk_hashes`` shape, block-grain)."""
    if len(hbm_hashes) == 1:
        return hbm_hashes[0]
    return "chunk:" + "|".join(hbm_hashes)


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
