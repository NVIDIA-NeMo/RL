# Register mode (`data_plane.backend: transfer_engine`)

Publish an address on `put`; move the bytes once, on `get`.

`mooncake_cpu` copies every payload into a store before `put` returns — through
host staging, and out of GPU memory first if that is where the tensor lived.
Register mode does neither: it registers the producer's own allocation with the
local NIC, publishes `{endpoint, base, offset, size}` through TransferQueue's
`custom_backend_meta`, and the bytes move exactly once, when a consumer pulls
them one-sided. There is no store, no master, and no staging buffer.

## Quick start

```yaml
data_plane:
  enabled: true
  impl: transfer_queue
  backend: transfer_engine
  claim_meta_poll_interval_s: 0.5
  transfer_engine:
    use_gdr: true                  # receive into HBM where CUDA is initialized
    offload_source_to_host: false  # true: register a host copy instead of HBM
    rpc_port: 0                    # 0 = the engine picks a free port
```

Requires RDMA (`rdma_devices()` must find a device) and, for `use_gdr`, a GPU.
It is otherwise a drop-in for `mooncake_cpu` — nothing above `DataPlaneClient`
changes.

## The three shapes

`use_gdr` decides where a **read** lands in the process issuing it;
`offload_source_to_host` decides where a **published source** lives. They are
independent, and both are resolved per process, so a CPU-only SingleController
driver and GPU workers can share one config.

| config | put | get |
|---|---|---|
| `use_gdr: true` | register HBM in place, no copy | producer HBM → consumer HBM, one hop |
| `use_gdr: true`, `offload_source_to_host: true` | one D2H, register the host copy | producer host → consumer HBM, one hop |
| `use_gdr: false` | register host memory in place | producer host → consumer host |

All three are verified on hardware; see the design doc's *Measured* table.

## What it costs, measured

On a GB300 node (RoCE), Qwen3-30B-A3B GRPO, per call:

| op | register mode | `mooncake_cpu` |
|---|---|---|
| `put_samples` | 15.3 ms (pin 1.6 ms + 13.7 ms TQ) | 28.3 ms |
| `get_samples` | 46.9 ms (client 19.3 ms + 27.6 ms TQ) | 37.1 ms |
| `clear_samples` | 0.2 ms | 13.3 ms |

Three things that table does not say on its own:

- **More than half of both backends is TransferQueue overhead** — controller
  round trip, schema extraction, TensorDict rebuild — which neither controls.
  At this payload (~6 MB per put, ~25 MB per get) the whole data plane is under
  1 % of a ~100 s step, so backend choice does not move step time.
- **`get` is slower, structurally.** Every fetch reports `peers=1`: a step's
  reads all funnel through the producing process's NIC (~1.6 GB/s effective),
  while a store spreads them over segments cluster-wide. Re-seeding — consumers
  registering what they pulled — is the fix, and it is not built.
- **The `clear` gap is not a fair fight.** Register mode defers its
  `ibv_dereg_mr`; mooncake still makes a synchronous master RPC, which is
  deferrable by the same argument. What *is* architectural is that register mode
  has no master in the release path at all.

Registration costs ~270 µs per `ibv_reg_mr` **regardless of size** (0.5 MB and
64 MB measure the same), so the pin cost tracks the number of distinct
allocations, not bytes — a put covering six fields pays six registrations. See
`tools/bench_pin_vs_copy.py`.

## Lifetime

A producer's registration lives from `put` until `clear`, and **only the owner
can unregister its own memory**. Since `clear` runs on the driver in both
trainers, each client serves a small ZMQ `PULL` socket (its `release_port` rides
in the meta beside `endpoint`) and `clear` forwards each foreign key to whoever
published it. Without that, registrations accumulate every step for the life of
the process — pinned memory, and an MR per publication against a finite NIC
table.

Releases carry `(key, seq)` rather than addresses: TQ recycles
`<global_idx>@<field>` keys across steps, so a replayed `clear` naming a
republished key must not unpin live data, and matching on the address cannot
tell them apart because the caching allocator hands the same block back.

Two rules for callers: **the producer must outlive the keys it published**, and
**must not mutate a registered buffer until `clear`**.

## Operational notes

- `WITH_NVIDIA_PEERMEM` (no `MC_` prefix) defaults to *true* in the pinned
  mooncake wheel, which registers GPU memory via `ibv_reg_mr` and needs the
  `nvidia_peermem` kernel module. Where that module is absent — GB200/GB300
  nodes — every CUDA registration fails with `ERR_CONTEXT` (-202).
  `transfer_queue_env` sets it to `0` (the DMA-BUF route) when a run registers
  GPU memory and the module is missing.
- Metadata is `P2PHANDSHAKE` only: no metadata server, no master, nothing for
  bootstrap to start.
- Each client logs one line at startup naming its endpoint, transport, receive
  device and source policy. That line is the only way to tell a GDR client from
  one that silently fell back to host buffers.

## Where to look

| Concern | File |
|---|---|
| Backend (client, manager, bootstrap) | `tq_register_mode.py` |
| Config schema | `../interfaces.py` (`TransferEngineConfig`) |
| Backend selection, engine env | `transfer_queue.py`, `transfer_queue_env.py` |
| Design, measurements, open work | `../../../docs/design-docs/tq-register-mode.md` |
| Unit tests (faked engine) | `../../../tests/unit/data_plane/test_tq_register_mode.py` |
| Real-NIC smoke | `../../../tools/smoke_register_mode_gdr.py` |
| Registration-vs-copy benchmark | `../../../tools/bench_pin_vs_copy.py` |
| GB200/GB300 recipe + launcher | `../../../examples/configs/recipes/llm/grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_register.yaml` |
