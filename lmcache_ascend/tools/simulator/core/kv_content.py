"""Logical KV content identity and tier slot keys (LMCache chunk alignment)."""

from __future__ import annotations

from dataclasses import dataclass

from .memory import Memory
from .request import Request


def content_id(hbm_blocks: list[str]) -> str:
    """Encode one or more HBM block hashes as a logical content token."""
    if len(hbm_blocks) == 1:
        return hbm_blocks[0]
    return "chunk:" + "|".join(hbm_blocks)


def parse_content_id(token: str) -> list[str]:
    if not token.startswith("chunk:"):
        return [token]
    return token.split(":", 1)[1].split("|")


@dataclass(frozen=True, order=True)
class ContentKey:
    """Tier-independent logical KV content."""

    token: str

    @classmethod
    def from_block(cls, hbm_hash: str) -> ContentKey:
        return cls(content_id([hbm_hash]))

    @classmethod
    def from_slot(cls, storage_key: str) -> ContentKey:
        return cls(content_id(parse_content_id(storage_key)))

    def hbm_blocks(self) -> list[str]:
        return parse_content_id(self.token)

    def anchor(self) -> str:
        blocks = self.hbm_blocks()
        return blocks[0] if blocks else self.token

    def __str__(self) -> str:
        return self.token


def aligned_blocks(
    prefix: list[str], block_hash: str, chunk_blocks: int
) -> list[str]:
    """``chunk_blocks`` consecutive prefix blocks covering ``block_hash``."""
    if chunk_blocks <= 1:
        return [block_hash]
    try:
        idx = prefix.index(block_hash)
    except ValueError:
        return [block_hash]
    start = (idx // chunk_blocks) * chunk_blocks
    return prefix[start : start + chunk_blocks]


def storage_key(
    req: Request | None,
    block_hash: str,
    chunk_blocks: int,
) -> str:
    """``Memory.blocks`` dict key on a tier (coarser when ``chunk_blocks > 1``)."""
    if chunk_blocks <= 1:
        return block_hash
    if req is None:
        return content_id([block_hash])
    prefix = req.block_hashes[: req.prefix_block_count]
    if block_hash not in prefix:
        return content_id([block_hash])
    return content_id(aligned_blocks(prefix, block_hash, chunk_blocks))


def tier_has_block(mem: Memory, req: Request | None, block_hash: str) -> bool:
    key = storage_key(req, block_hash, mem.chunk_blocks)
    return mem.best_resident(key) is not None


def tier_has_inflight(mem: Memory, req: Request | None, block_hash: str) -> bool:
    key = storage_key(req, block_hash, mem.chunk_blocks)
    return mem.inflight_incoming(key) is not None


def transfer_block_count(mem: Memory, req: Request | None, block_hash: str) -> int:
    """HBM blocks covered by one tier-slot transfer."""
    if mem.chunk_blocks <= 1 or req is None:
        return 1
    prefix = req.block_hashes[: req.prefix_block_count]
    if block_hash not in prefix:
        return 1
    return len(aligned_blocks(prefix, block_hash, mem.chunk_blocks))


def transfer_work(mem: Memory, req: Request | None, block_hashes: list[str]) -> float:
    if not block_hashes:
        return 0.0
    return float(transfer_block_count(mem, req, block_hashes[0]))


def shared_pull_key(
    src_key: str,
    req: Request | None,
    block_hash: str,
    memories: dict[str, Memory],
) -> tuple[str, ContentKey]:
    src = memories[src_key]
    slot = storage_key(req, block_hash, src.chunk_blocks)
    return (src_key, ContentKey.from_slot(slot))


def group_pulls(
    block_hashes: list[str],
    actions: dict,
    req: Request | None,
    memories: dict[str, Memory],
) -> list[tuple[str, list[str]]]:
    """Group consecutive same-source pulls into chunk-aligned ranges."""
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
            aligned_blocks(
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
