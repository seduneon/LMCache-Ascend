"""Engine construction bundles: work costs and resource links."""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.core.resource import BandwidthResource, ComputeResource


@dataclass(frozen=True)
class WorkModel:
    """Forward/transfer/evict work units for one engine."""

    per_block: float = 1.0
    per_transfer: float | None = None
    per_store: float | None = None
    per_evict: float = 0.0
    per_prefill_token: float | None = None
    per_decode_req: float | None = None

    def resolved_transfer(self) -> float:
        return self.per_block if self.per_transfer is None else self.per_transfer

    def resolved_store(self) -> float:
        transfer = self.resolved_transfer()
        return transfer if self.per_store is None else self.per_store

    def resolved_prefill_token(self) -> float:
        return self.per_block if self.per_prefill_token is None else self.per_prefill_token

    def resolved_decode_req(self) -> float:
        return self.per_block if self.per_decode_req is None else self.per_decode_req


@dataclass(frozen=True)
class EngineLinks:
    """Compute, bandwidth, and per-tier transfer/write resources."""

    compute_res: ComputeResource
    bandwidth_res: BandwidthResource | None = None
    transfer_links: dict[str, BandwidthResource] = field(default_factory=dict)
    write_links: dict[str, BandwidthResource] = field(default_factory=dict)
    interconnect: BandwidthResource | None = None

    @classmethod
    def from_bandwidth(
        cls,
        compute_res: ComputeResource,
        bandwidth_res: BandwidthResource | None,
        *,
        pull_sources: list[str],
        write_links: dict[str, BandwidthResource] | None = None,
        interconnect: BandwidthResource | None = None,
    ) -> EngineLinks:
        transfer: dict[str, BandwidthResource] = {}
        if bandwidth_res is not None and pull_sources:
            transfer = {src: bandwidth_res for src in pull_sources}
        return cls(
            compute_res=compute_res,
            bandwidth_res=bandwidth_res,
            transfer_links=transfer,
            write_links=dict(write_links or {}),
            interconnect=interconnect,
        )
