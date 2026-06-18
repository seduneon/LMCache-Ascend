"""Tier-independent logical KV content identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import Memory
    from .request import Request


def lmcache_chunk_hash(hbm_hashes: list[str]) -> str:
    """Deterministic chunk content id (LMCache ``chunk_hashes`` shape, block-grain)."""
    if len(hbm_hashes) == 1:
        return hbm_hashes[0]
    return "chunk:" + "|".join(hbm_hashes)


def hbm_blocks_in_token(token: str) -> list[str]:
    if not token.startswith("chunk:"):
        return [token]
    return token.split(":", 1)[1].split("|")


@dataclass(frozen=True, order=True)
class ContentKey:
    """Tier-independent logical KV content id."""

    token: str

    @classmethod
    def for_hbm_block(cls, hbm_hash: str) -> ContentKey:
        """Finest-grain content id for one HBM block."""
        return cls(lmcache_chunk_hash([hbm_hash]))

    @classmethod
    def for_storage_key(cls, storage_key: str) -> ContentKey:
        """Normalize a tier ``Memory`` slot key to logical content."""
        return cls(lmcache_chunk_hash(hbm_blocks_in_token(storage_key)))

    def hbm_blocks(self) -> list[str]:
        return hbm_blocks_in_token(self.token)

    def anchor_hbm(self) -> str:
        blocks = self.hbm_blocks()
        return blocks[0] if blocks else self.token

    def __str__(self) -> str:
        return self.token


def tier_storage_key(
    content: ContentKey,
    mem: Memory,
    req: Request | None,
    hbm_hash: str,
) -> str:
    """Key in ``Memory.blocks`` for ``mem`` (coarser slots on downstream tiers)."""
    from .chunk_hash import chunk_key_for_hbm_block

    if mem.chunk_blocks <= 1:
        return hbm_hash
    return chunk_key_for_hbm_block(req, hbm_hash, mem.chunk_blocks)
