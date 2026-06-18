"""Memory tier topology for PD experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .memory import Memory


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
    """Sizing knobs shared across presets."""

    hbm_size: int = 40
    dram_size: int = 80
    ssd_size: int = 160
    dram_chunk_blocks: int = 4
    ssd_chunk_blocks: int = 4


def build_topology(
    kind: str,
    *,
    resources: SimResources | None = None,
) -> Topology:
    cfg = resources or SimResources()
    if kind == "hbm_only":
        memories = {
            "npu-0:hbm": Memory(size=cfg.hbm_size, name="npu-0:hbm"),
            "npu-1:hbm": Memory(size=cfg.hbm_size, name="npu-1:hbm"),
        }
    elif kind == "hbm_dram":
        memories = build_topology("hbm_only", resources=cfg).memories
        memories["npu-0:dram"] = Memory(
            size=cfg.dram_size,
            name="npu-0:dram",
            chunk_blocks=cfg.dram_chunk_blocks,
        )
    elif kind == "hbm_dram_ssd":
        memories = build_topology("hbm_dram", resources=cfg).memories
        memories["npu-0:ssd"] = Memory(
            size=cfg.ssd_size,
            name="npu-0:ssd",
            chunk_blocks=cfg.ssd_chunk_blocks,
        )
    else:
        raise ValueError(f"unknown topology {kind!r}")
    return Topology(memories=memories)
