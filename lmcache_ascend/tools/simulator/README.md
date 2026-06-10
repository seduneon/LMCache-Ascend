# Simulator

## v0 scope

- Single HBM, compute-only loads
- FIFO admission, head-of-line blocking on failed lookup
- `ComputeAllLookup` + `FirstAvailableEviction`
- Multiple concurrent active requests

## Next (v1 ideas)

- `pull` action and multiple memory tiers
- LRU / better eviction policies (`EvictionPolicy` subclass)
- Separate bandwidth resource for transfer/evict
- `run()` done vs stuck status
- Prefill/decode (`RequestPD`) in the workflow
- Tests
