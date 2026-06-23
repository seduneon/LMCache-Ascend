"""Convert tier capacity in GiB to simulator block/slot counts."""

from __future__ import annotations

from dataclasses import dataclass

from simulator.policy.eviction import EvictionPolicy, LRUEviction
from simulator.core.memory import Memory

GIB = 1024**3

# Full-model K+V bytes per token (bf16), reference models for --kv-model.
KV_BYTES_PER_TOKEN: dict[str, float] = {
    "toy": 256.0,
    "llama3-8b": 32 * 2 * 8 * 128 * 2,
    "llama3-70b": 80 * 2 * 8 * 128 * 2,
}


@dataclass(frozen=True)
class TierSpec:
    """One memory tier: capacity in GiB and slot granularity."""

    tier_key: str
    capacity_gib: float
    chunk_blocks: int = 1


def kv_bytes_per_token(model: str = "llama3-8b") -> float:
    key = model.lower().replace("_", "-")
    if key not in KV_BYTES_PER_TOKEN:
        known = ", ".join(sorted(KV_BYTES_PER_TOKEN))
        raise ValueError(f"unknown kv model {model!r} (choose: {known})")
    return KV_BYTES_PER_TOKEN[key]


def default_tiers(
    *,
    hbm_gib: float = 32.0,
    dram_gib: float = 64.0,
    ssd_gib: float = 256.0,
    dram_chunk_blocks: int = 4,
    ssd_chunk_blocks: int = 4,
    prefill_ids: tuple[str, ...] = ("npu-0",),
    decode_ids: tuple[str, ...] = ("npu-1",),
) -> tuple[TierSpec, ...]:
    return default_tiers_for(
        prefill_ids,
        decode_ids,
        hbm_gib=hbm_gib,
        dram_gib=dram_gib,
        ssd_gib=ssd_gib,
        dram_chunk_blocks=dram_chunk_blocks,
        ssd_chunk_blocks=ssd_chunk_blocks,
    )


def default_tiers_for(
    prefill_ids: tuple[str, ...],
    decode_ids: tuple[str, ...],
    *,
    hbm_gib: float = 32.0,
    dram_gib: float = 64.0,
    ssd_gib: float = 256.0,
    dram_chunk_blocks: int = 4,
    ssd_chunk_blocks: int = 4,
) -> tuple[TierSpec, ...]:
    """Tier specs for each engine HBM plus downstream tiers on the primary prefill."""
    specs: list[TierSpec] = []
    for eid in prefill_ids + decode_ids:
        specs.append(TierSpec(f"{eid}:hbm", hbm_gib, 1))
    primary = prefill_ids[0]
    specs.append(TierSpec(f"{primary}:dram", dram_gib, dram_chunk_blocks))
    specs.append(TierSpec(f"{primary}:ssd", ssd_gib, ssd_chunk_blocks))
    return tuple(specs)


def bytes_per_kv_block(*, tokens_per_block: int, kv_bytes_per_token: float) -> int:
    """Bytes for one logical HBM KV block."""
    return max(1, int(tokens_per_block * kv_bytes_per_token))


def bytes_per_slot(
    *,
    tokens_per_block: int,
    kv_bytes_per_token: float,
    chunk_blocks: int = 1,
) -> int:
    return bytes_per_kv_block(
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
    ) * max(1, chunk_blocks)


def blocks_from_gib(
    gib: float,
    *,
    tokens_per_block: int,
    kv_bytes_per_token: float,
    chunk_blocks: int = 1,
) -> int:
    """Floor capacity in GiB to a positive slot count."""
    if gib <= 0:
        raise ValueError(f"tier capacity must be positive GiB, got {gib}")
    slot_bytes = bytes_per_slot(
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
        chunk_blocks=chunk_blocks,
    )
    return max(1, int(gib * GIB // slot_bytes))


def resolve_tier_slots(
    tier: TierSpec,
    *,
    tokens_per_block: int,
    kv_bytes_per_token: float,
) -> int:
    return blocks_from_gib(
        tier.capacity_gib,
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
        chunk_blocks=tier.chunk_blocks,
    )


def build_tier(
    spec: TierSpec,
    *,
    tokens_per_block: int,
    kv_bytes_per_token: float,
    eviction: EvictionPolicy | None = None,
    slots: int | None = None,
    role: "TierRole" = "downstream",
) -> "Tier":
    from .tier import Tier

    resolved = (
        slots
        if slots is not None
        else resolve_tier_slots(
            spec,
            tokens_per_block=tokens_per_block,
            kv_bytes_per_token=kv_bytes_per_token,
        )
    )
    memory = Memory(
        size=resolved,
        name=spec.tier_key,
        chunk_blocks=spec.chunk_blocks,
    )
    return Tier(
        key=spec.tier_key,
        memory=memory,
        eviction=eviction or LRUEviction(),
        role=role,
    )


def gib_for_blocks(
    blocks: int,
    *,
    tokens_per_block: int,
    kv_bytes_per_token: float,
    chunk_blocks: int = 1,
) -> float:
    """Inverse of ``blocks_from_gib`` (for tests and migration)."""
    slot_bytes = bytes_per_slot(
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
        chunk_blocks=chunk_blocks,
    )
    return blocks * slot_bytes / GIB


def format_tier_capacity(tier: TierSpec, *, blocks: int, kv_bytes_per_token: float, tokens_per_block: int) -> str:
    slot_bytes = bytes_per_slot(
        tokens_per_block=tokens_per_block,
        kv_bytes_per_token=kv_bytes_per_token,
        chunk_blocks=tier.chunk_blocks,
    )
    return (
        f"{tier.tier_key}={blocks} slots ({tier.capacity_gib:g} GiB @ "
        f"{tokens_per_block} tok/block, {int(kv_bytes_per_token)} B/token-kv, "
        f"{slot_bytes} B/slot)"
    )
