# Simulator

## v0

### Requirements

- no KV cache, everything is compute
- a single instance HBM, no PD disaggregation

### Structure

- Resources (ABC)
    - current workload info
    - add, del functions
- ComputeResource, MemoryResources
    - override speed information
- TaskPool
    - tasks 
    - add
    - next
- Task (ABC)
    - dependents
    - dependencies
    - left
- TransferTask, ComputeTask, EvictTask
    - onStart, onEnd, other info
- Memory
    - size
    - kv blocks
- KV block
    - size
    - hash
    - info (LRU, etc)
    - dependents