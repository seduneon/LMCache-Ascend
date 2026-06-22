"""Memory tier topology for PD experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from .capacity import TierSpec, default_tiers, resolve_tier_slots
from .memory import Memory

TOPOLOGY_TIERS: dict[str, tuple[str, ...]] = {
    "hbm_only": ("npu-0:hbm", "npu-1:hbm"),
    "hbm_dram": ("npu-0:hbm", "npu-1:hbm", "npu-0:dram"),
    "hbm_dram_ssd": ("npu-0:hbm", "npu-1:hbm", "npu-0:dram", "npu-0:ssd"),
}


@dataclass(frozen=True)
class Topology:
    """Named memory tiers and PD local/pull wiring."""

    memories: dict[str, Memory]
    prefill_hbm: str = "npu-0:hbm"
    decode_hbm: str = "npu-1:hbm"

    @property
    def prefill_engine_id(self) -> str:
        return self.prefill_hbm.split(":")[0]

    @property
    def decode_engine_id(self) -> str:
        return self.decode_hbm.split(":")[0]

    def decode_pull_sources(self, sources: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(s for s in sources if s in self.memories)


@dataclass(frozen=True)
class SimResources:
    """Resolved tier slot counts."""

    slots: dict[str, int] = field(default_factory=dict)
    chunk_blocks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_tiers(
        cls,
        tiers: tuple[TierSpec, ...],
        *,
        tokens_per_block: int,
        kv_bytes_per_token: float,
    ) -> SimResources:
        slots: dict[str, int] = {}
        chunks: dict[str, int] = {}
        for tier in tiers:
            slots[tier.tier_key] = resolve_tier_slots(
                tier,
                tokens_per_block=tokens_per_block,
                kv_bytes_per_token=kv_bytes_per_token,
            )
            chunks[tier.tier_key] = tier.chunk_blocks
        return cls(slots=slots, chunk_blocks=chunks)

    @classmethod
    def default(
        cls,
        *,
        tokens_per_block: int = 512,
        kv_bytes_per_token: float | None = None,
    ) -> SimResources:
        from .capacity import kv_bytes_per_token as resolve_kv_model

        kv_bpt = kv_bytes_per_token if kv_bytes_per_token is not None else resolve_kv_model("llama3-8b")
        return cls.from_tiers(
            default_tiers(),
            tokens_per_block=tokens_per_block,
            kv_bytes_per_token=kv_bpt,
        )

    def size(self, tier_key: str) -> int:
        return self.slots[tier_key]

    @property
    def hbm_size(self) -> int:
        return self.slots["npu-0:hbm"]

    @property
    def dram_size(self) -> int:
        return self.slots.get("npu-0:dram", 0)

    @property
    def ssd_size(self) -> int:
        return self.slots.get("npu-0:ssd", 0)


def build_topology(
    kind: str,
    *,
    resources: SimResources | None = None,
) -> Topology:
    tier_keys = TOPOLOGY_TIERS.get(kind)
    if tier_keys is None:
        raise ValueError(f"unknown topology {kind!r}")

    cfg = resources or SimResources.default()
    memories: dict[str, Memory] = {}
    for tier_key in tier_keys:
        memories[tier_key] = Memory(
            size=cfg.size(tier_key),
            name=tier_key,
            chunk_blocks=cfg.chunk_blocks.get(tier_key, 1),
        )
    return Topology(memories=memories)
