# TransferQueue Register Mode

> **Status: implemented, experimental.** Available as
> `data_plane.backend: "transfer_engine"`; the backend lives in
> `nemo_rl/data_plane/adapters/tq_register_mode.py` and registers itself into
> TransferQueue's factories. The go/no-go measurement at the end has **not**
> been run — it is still the gate for using this at scale, because the read
> fan-out risk below is unresolved.

Today every `put_samples` into TransferQueue **copies bytes** out of the
producer and into storage before it returns. Register mode replaces that copy
with an RDMA memory registration: the producer publishes the address of a
buffer it already holds, the controller records where it lives, and the bytes
move only when a consumer actually reads — pulled one-sided, straight from the
producer's memory.

This is the same primitive Mooncake's P2P Store already exposes; the proposal is
to reach it from TransferQueue.

---

## Background — what `put` does today

Both TQ backends move data eagerly. Neither has a lazy path.

**SimpleStorage** (`transfer_queue/storage/managers/simple_storage_manager.py`)
routes each sample to a storage unit by `global_idx % num_su`
(`_group_by_hash`, `:127-144`) and ships the payload over ZMQ
(`_put_to_single_storage_unit`, `:288-334`). CUDA tensors are forced to host
first — `if obj.device.type != "cpu": obj = obj.cpu()`
(`transfer_queue/utils/serial_utils.py:186-187`, `:206-207`). Storage units
never talk to each other; the topology is a star with the producer at one leaf.

**MooncakeStore** (`transfer_queue/storage/clients/mooncake_client.py`) is the
backend NeMo-RL uses at scale. Per put it:

1. registers the source buffer with the local NIC — `register_buffer` (`:683-689`),
   which lands on `Client::RegisterLocalMemory(..., remote_accessible=false, ...)`
   (`mooncake-store/src/real_client.cpp:3326-3342`). Note `false`: the buffer is
   pinned as an RDMA *source*, never published for anyone to read;
2. calls `batch_upsert_from`, which runs `StartBatchUpsert` → **`SubmitTransfers`
   → `WaitForTransfers`** → `FinalizeBatchUpsert`
   (`mooncake-store/src/client_service.cpp:2156-2169`) — the write completes
   before the call returns;
3. **unregisters the buffer in a `finally`** (`:261`, `:291`).

Step 3 is the proof that the data was copied: the source region is released
immediately, and the GDR path goes further and reuses a *single* staging buffer
for every group (`GdrStaging`, `transfer_queue/utils/mooncake_utils.py:90-101`).
Neither would be safe if the store were holding a reference.

The transfer itself picks one of exactly two strategies
(`mooncake-store/src/transfer_task.cpp:1480-1486`), and both copy:

| Condition | Strategy | Mechanism |
|---|---|---|
| target segment in the **same process** (`:1494-1503`) | `LOCAL_MEMCPY` | `std::memcpy` (`:675`) or an accelerator copy (`:688-700`) |
| anything else | `TRANSFER_ENGINE` | one-sided RDMA `WRITE` (`:1097-1101`) |

There is no third branch — no path that adopts the caller's pointer as the
storage location. TQ also cannot steer the allocation local: `prefer_alloc_in_same_node`
is rejected outright (`real_client.cpp:4292-4296`).

---

## What Mooncake already provides

**P2P Store `Register()` is exactly the semantics we want**, and it already
exists — in Go, without a master:

> "A `Register` is equivalent to seeding in BitTorrent, where a local file is
> registered with the global metadata **without any data transfer occurring**;
> it merely registers metadata in etcd."
> — `Mooncake/docs/source/design/p2p-store.md:6`

The implementation (`mooncake-p2p-store/src/p2pstore/core.go:123-183`) registers
the MR locally, builds shard descriptors whose location is *the registrar's own
address* (`goldLocation := Location{SegmentName: localServerName, Offset: addr+offset}`,
`:159-168`), and writes the catalog to etcd. No transfer call appears in the
function. Movement is deferred to `GetReplica` (`:330-363`), which issues a
one-sided `OPCODE_READ` from a holder — and then registers the puller as a *new*
source (`updatePayloadMetadata`, `:362`), so fan-out doesn't saturate the origin.

**Mooncake Store cannot do this**, and the `pub_*` API family is a red herring
despite the name: `pub_tensor` is literally
`return put_tensor_impl(key, tensor, config);`
(`mooncake-integration/store/store_py.cpp:1692-1703`) — "publish" here means
"put with a caller-supplied `ReplicateConfig`", not publish-by-reference. The
store's master owns allocation, eviction, pinning and replication over pooled
segments; an object whose only copy sits in a caller's buffer cannot be evicted
or replicated, so the semantics are structurally incompatible.

**The Transfer Engine's Python bindings expose the primitives directly**, which
is what makes this buildable without touching C++ or Go
(`mooncake-integration/transfer_engine/transfer_engine_py.cpp`):

| Need | Binding |
|---|---|
| register, no copy | `register_memory(buffer_addr, capacity, location)` (`:1275-1277`) — `remote_accessible` defaults to **true** (`mooncake-transfer-engine/include/transfer_engine.h:127-130`) |
| bulk register | `batch_register_memory(addrs, capacities, location)` (`:1279-1282`) |
| consumer pull | `batch_transfer_sync_read(target_hostname, buffers, peer_addrs, lengths)` (`:1213-1217`) |
| pull into GPU | `batch_transfer_read_on_cuda(..., stream_ptr, ...)` (`:1257-1261`) |
| release | `unregister_memory` / `batch_unregister_memory` (`:1278-1284`) |

---

## Proposed design

A new `StorageKVClient` registered through the existing factory, sitting beside
`MooncakeStoreClient`:

```python
@StorageClientFactory.register("TransferEngineClient")
class TransferEngineClient(StorageKVClient):
    def put(self, keys, values) -> list[dict | None]: ...
    def get(self, keys, shapes, dtypes, custom_backend_meta) -> list: ...
    def clear(self, keys, custom_backend_meta) -> None: ...
```

### Nothing above the client changes

This is what keeps the proposal small. `KVStorageManager.put_data` already
scatters whatever the client returns into per-field, per-sample
`custom_backend_meta` and forwards it to the controller
(`transfer_queue/storage/managers/base.py:751-774`), and
`_get_shape_type_custom_backend_meta_list` (`:695-723`) hands it back on both
`get_data` and `clear_data` (`:803-813`).

So the address book is already plumbed end to end. Register mode just puts
different contents in it:

```python
{"endpoint": "<producer TE host:rpc_port>", "base": <registered base addr>,
 "offset": <byte offset of this row>, "size": <nbytes>}
```

Shapes and dtypes already ride the existing `field_schema`. The controller,
`BatchMeta`, the samplers, and every NeMo-RL layer above `DataPlaneClient` are
untouched.

### The overlap constraint forces the shape

The Transfer Engine **rejects overlapping registrations**:

```
"Transfer Engine does not support overlapped memory region"  → ERR_ADDRESS_OVERLAPPED
```
— `mooncake-transfer-engine/src/transfer_engine_impl.cpp:613-617`

This rules out the naive one-line swap. `KVStorageManager._generate_values`
(`base.py:559-580`) flattens a batched TensorDict into per-row values — either
`results.extend(field_data)` over rows of one tensor, or `field_data.unbind()`
for nested — so **every value handed to `put` is a view into a shared
allocation.** Registering per key fails on the second row of every field.

NeMo-RL's codec makes this concrete: per-token fields become `torch.jagged`
nested tensors via `pack_per_token_field` → `to_nested_by_length`
(`nemo_rl/data_plane/codec.py:185-212`), and those are precisely the bulk
columns (`input_ids`, `generation_logprobs`, `token_mask`, `advantages`).
Non-token fields get `.detach().contiguous()` (`:176`) and are individually
clean.

The design is therefore forced into **register the base allocation once, record
per-key offsets**:

```python
storage = t.untyped_storage()
base = storage.data_ptr()
if base not in self._pinned:
    engine.register_memory(base, storage.nbytes())
    self._pinned[base] = [storage, 0]          # strong ref + refcount
self._pinned[base][1] += 1
meta = {"endpoint": self._endpoint, "base": base,
        "offset": t.data_ptr() - base, "size": t.nbytes}
```

This is also the fix for registration cost — one `ibv_reg_mr` per batch instead
of per sample, which matters because pinning is the expensive part (it is the
entire reason `GdrStaging` registers one buffer for the process lifetime rather
than per put).

The **strong reference is mandatory, not defensive**: holding only the pointer
lets PyTorch's caching allocator reissue the block after the tensor is collected,
and consumers would silently read another tensor's data.

### Read path

`get` cannot be a one-liner either, because the Transfer Engine requires the
*destination* to be registered before it can be a transfer target. So:

1. group keys by `endpoint`;
2. allocate + register a receive buffer — reuse the `GdrStaging` pattern
   (`transfer_queue/utils/mooncake_utils.py:90`), which already handles
   register-once, lock-serialize, and oversized-payload chunking;
3. `batch_transfer_sync_read(endpoint, [dst_ptrs], [base+offset], [sizes])`;
4. slice out into per-key tensors.

### Non-tensor fields

Cheaper than it first appears. Python objects have no stable buffer, but the
existing path already serializes them into a *freshly allocated* contiguous
uint8 region (`allocate_empty_tensors`, `mooncake_client.py:270-280`). A fresh
region has no overlap problem, so it registers under exactly the same model as
tensors — no second backend, no hybrid dispatch.

### Lifecycle

`clear_data` already carries `custom_backend_meta`, so it becomes the
`unregister` trigger for free: decrement the refcount on `base`, and at zero
call `unregister_memory` and drop the strong reference.

---

## Fit with the NeMo-RL sync GRPO step

The relevant flow is in `nemo_rl/data_plane/README.md` ("E2E flow — one sync
GRPO step") and `nemo_rl/algorithms/grpo_sync.py`.

**Producer liveness: good.** The lifetime contract — "do not modify or unmap the
region before `Unregister`" (`p2p-store.md:58`) — is satisfiable here. The
rollout actor performs `kv_first_write` at step start and stays alive for the
whole step; `policy.finish_step(meta)` drops the step's bulk at the end. Worker
leader write-backs (`prev_lp`, `ref_lp`) and the driver's `write_to_dataplane`
(`advantages`, `sample_mask`) come from processes that likewise persist across
the step. Registration lifetime maps cleanly onto the existing step boundary.

**Read fan-out: this is the risk.** Within one step the same keys are read
repeatedly:

| Phase | Reader |
|---|---|
| `get_logprobs_from_meta` | DP workers, sharded ~64 keys each |
| `get_reference_policy_logprobs_from_meta` | ref-policy workers, same keys again |
| `read_from_dataplane` | driver, `generation_logprobs` + `token_mask`, all rows |
| `train_from_meta` | DP workers, all keys again |

Each individual read is a *disjoint shard*, so no single key is broadcast. But
every one of those reads targets bulk that **one rollout actor produced**. Today
with `mooncake_cpu` those bytes were copied at put time into segments spread
across the whole cluster, so reads are served by many NICs. Register mode
concentrates every read for the step onto the producing actor's single NIC.

That is the central open question, and it is the inverse of the usual
motivation: register mode saves one copy at write, then risks serializing three
to four read phases through one link. P2P Store's answer is re-seeding
(`core.go:362`); a first cut without it may well be slower than what exists.

**Backend context.** NeMo-RL currently runs `backend: "mooncake_cpu"` with
`protocol: rdma` and no `use_gdr` (`nemo_rl/data_plane/README.md` configuration
block; `nemo_rl/data_plane/adapters/transfer_queue.py:99`) — i.e. the CPU RDMA
path, where put bytes are host-resident. That is favourable: host DRAM is
plentiful, so pinning a packed batch for a step is cheap, and the saved D2H +
copy is real. Register mode is materially riskier for GPU-resident puts, where
holding a strong reference to `untyped_storage()` pins a whole caching-allocator
block and the producer's HBM is not reclaimed mid-step.

---

## As implemented

Everything lives in NeMo-RL — `nemo_rl/data_plane/adapters/tq_register_mode.py`
— and plugs into TransferQueue through its existing decorator registries, so no
TQ fork is needed:

| Piece | Registered as |
|---|---|
| `TransferEngineClient(StorageKVClient)` | `StorageClientFactory` → `"TransferEngineClient"` |
| `TransferEngineStorageManager(KVStorageManager)` | `StorageManagerFactory` → `"TransferEngine"` |
| `initialize_transfer_engine_storage` | `StorageBootstrapProvider` → `"TransferEngine"` |

`nemo_rl/data_plane/adapters/transfer_queue.py` imports the module (so every
process that builds a TQ storage manager has the registrations) and adds the
`transfer_engine` overlay to `_init_tq`. TQ code is reused as-is:
`allocate_empty_tensors` / `compute_stride` / `get_nbytes` for receive buffers
and `serial_utils.batch_encode_into` / `batch_decode_from` for non-tensor
payloads. Transport selection is shared with `mooncake_cpu`: the same
`rdma_devices()` all-rail list and the same `MC_ENABLE_DEST_DEVICE_AFFINITY`
peer-rail pinning from `transfer_queue_env`.

### GDR: the payload never leaves HBM until a consumer asks

With `data_plane.transfer_engine.use_gdr: true` (the default) the whole path is
device-resident:

| Stage | `mooncake_cpu` (`use_gdr: true`) | register mode (`use_gdr: true`) |
|---|---|---|
| producer → put | D2H into host staging, then RDMA into the store's segment | `ibv_reg_mr` over the tensor's own HBM range; **no copy** |
| at rest | a second copy in the store | the producer's own tensor |
| consumer → get | RDMA out of the store into GDR staging, then D2D into the result | one-sided READ producer HBM → consumer HBM |

Mooncake classifies a registered range by probing the pointer
(`getMemoryLocation` → `cudaPointerGetAttributes`, `memory_location.cpp:71-75`),
so a CUDA allocation registers as `cuda:N` and inherits that GPU's affine rail
from the topology every other transfer already uses — no separate GDR transport
configuration.

**One knob is not optional.** Mooncake chooses between two GPU-registration
routes by environment variable rather than by probing, and `WITH_NVIDIA_PEERMEM`
(no `MC_` prefix) defaults to **true** in the pinned wheel
(`mooncake-common/src/environ.cpp:216`) — that route calls `ibv_reg_mr()` on the
GPU pointer and needs the `nvidia_peermem` kernel module. These GB200/GB300
nodes have no such module (`/proc/modules` has no `nvidia_peermem`), so every
CUDA registration fails with `ERR_CONTEXT` (-202) at
`rdma_transport.cpp:257`. Setting it to `0` selects the DMA-BUF route
(`cuMemGetHandleForAddressRange`), which is the supported path on this
generation and works. `transfer_queue_env` therefore sets
`WITH_NVIDIA_PEERMEM=0` when a run registers GPU memory *and* the module is
absent, leaving hosts that do have it on upstream's behaviour. Note the upstream
docs claim the opposite default; the code is what runs.

`use_gdr` is resolved **per process**, not per cluster: it takes effect where
`torch.cuda.is_initialized()`, so a CPU-only client (the SingleController
driver) keeps host receive buffers while GPU workers receive into HBM. A source
is registered wherever it lives regardless, so a mixed pairing still works.

### Host source, HBM destination

The source and destination halves are configured separately, because the useful
middle ground is host-resident sources with HBM landings:

```
offload_source_to_host: true, use_gdr: true
    put:  D2H once, register the host buffer  (no HBM held until clear)
    get:  producer HOST --RDMA--> consumer HBM   (still one hop)
```

Compare `mooncake_cpu` with GDR, where the same payload takes two RDMA hops
through a host-resident store object and a D2D copy at each end:

```
put:  producer HBM --D2D--> GPU staging --RDMA--> store segment (host)
get:  store segment --RDMA--> GPU staging --D2D--> consumer HBM
```

The offload variant costs one D2H per put and moves what a producer holds
registered between `put` and `clear` out of HBM into host memory. Leave it off
to keep the payload in HBM end to end.

Two call sites outside the client complete the no-copy contract:
`DataPlaneClient.put_device` (default `"cpu"`, `"cuda"` under register+GDR) and
`TQWorkerMixin._write_back_result_field`, which used to force `.to("cpu")` on
every write-back — under register mode that copy was pure loss.

### Deltas from the proposal above

- **`P2PHANDSHAKE` only.** No metadata server and no master, so bootstrap starts
  nothing; any other `metadata_server` value is rejected loudly.
- **Local reads bypass the fabric.** Keys whose endpoint is this process are
  filled with `Tensor.copy_` from the still-pinned source storage, which also
  gets the device pairing right when producer and consumer differ.
- **Receive regions are registered per `get`, not pooled.** A pooled destination
  would need a second copy out of the pool, which is the copy this backend
  exists to remove. If `ibv_reg_mr` on the receive side shows up in profiles,
  that is the first thing to revisit.
- **No re-seeding.** Consumers do not register what they pulled, so the read
  fan-out concern is live exactly as described above.
- **Release travels to the owner.** Only the owner can `unregister_memory` its
  own address space, but the moment a key becomes releasable is known centrally:
  it is TQ's `clear`, which runs on the driver in both trainers. Each client
  therefore serves a small ZMQ `PULL` socket (its `release_port` rides in the
  meta beside `endpoint`), and `clear` forwards each foreign key to whoever
  published it. Without this a producer's registrations accumulate every step
  for the life of the process — pinned memory that the caching allocator cannot
  reclaim, and, sooner, an `ibv_reg_mr` per publication against a finite NIC MR
  table.

  Two hazards shape the message format. Releases carry `(key, seq)`, not
  addresses: TQ recycles `<global_idx>@<field>` keys across steps, so a replayed
  `clear` naming a since-republished key would otherwise unpin live data, and
  matching on the address cannot distinguish them either — the caching allocator
  routinely returns the same block, so old and new publications commonly share a
  base. A per-client sequence number makes a replay or a duplicate a no-op.
  Re-publishing a key TQ never cleared drops the superseded pin as the new one
  lands, so an upsert cannot strand a registration.

  Sends are fire-and-forget with `IMMEDIATE` and a send timeout: a producer that
  has already exited must not block the caller of `clear`, and its registrations
  died with it anyway. `TransferEngineClient.pinned_keys` exposes the live count
  if it ever regrows.

### Measured

Coverage is in `tests/unit/data_plane/test_tq_register_mode.py`, which fakes the
Transfer Engine (`memmove` for a one-sided read) so the publish / group / pull /
release paths run without an RDMA NIC; the GDR cases assert on the transfer
descriptors — source and destination both HBM addresses — since a real transfer
needs a real NIC. `tools/smoke_register_mode_gdr.py` is the real-NIC counterpart:
two live engines in one process, distinct segments, byte comparison after the
pull.

On a GB300 node (RoCE, `mlx5_8`; `mlx5_17` present but with no active port):

| Path | Result |
|---|---|
| unit tests, CPU node | 43 passed |
| unit tests incl. the 3 CUDA cases, GPU node | 13 passed |
| real engine, host-resident sources | PASS — 4 tensor keys + a non-tensor payload, byte-exact |
| real engine, GDR, mooncake defaults | FAIL, `ERR_CONTEXT` (-202) — the peermem route above |
| real engine, GDR, DMA-BUF route | PASS — published base is a CUDA VA, every key returns `device=cuda` and byte-exact |

### NVLink (MNNVL) is 40x RDMA, and register mode cannot reach it as it stands

Measured on GB300, `tools/nvlink_crossnode.sbatch` + `tools/smoke_nvlink_normal_mode.py`:

| Path | Warm bandwidth (512MB) |
|---|---|
| RDMA, point-to-point, best case | 18.2 GB/s |
| RDMA, in situ during a real run | 0.6–3.9 GB/s (`peers=1` fan-out concentration) |
| MNNVL NVLink, same node | **765 GB/s** |
| MNNVL NVLink, cross node, same clique | **768 GB/s** |

Cross-node costs nothing in steady state; the whole difference is first touch
(76ms vs 6ms), which is `openSegment` + fabric handle import + address mapping,
paid once per peer. Both nodes reported `CliqueId 3406`, `ClusterUUID
cecca9d5-…`.

Four facts govern whether this is reachable, and three of them are traps:

1. **The transport is chosen by env var at install time, not by `protocol`.**
   `transfer_engine_impl.cpp`: `MC_INTRANODE_NVLINK` → `nvlink_intra`; else
   `MC_FORCE_MNNVL` or no HCAs → `nvlink`; else → `rdma`. Passing
   `protocol="nvlink"` only swaps the *memory allocator*, which is why an early
   run measured "nvlink" at RDMA speed. `nvlink_intra` is not compiled into the
   pinned wheel — setting `MC_INTRANODE_NVLINK` fails engine init outright.
2. **Register mode's sources are torch-allocated, and both torch modes failed.**
   `NvlinkTransport::allocatePinnedLocalMemory` builds its own buffers with
   `cuMemCreate` + `CU_MEM_HANDLE_TYPE_FABRIC`, so they are exportable by
   construction. Register mode instead registers whatever torch allocated:

   - **default caching allocator** — fails with a *proven* mechanism: mooncake
     logs `Memory region ... is not allocated by cuMemCreate` and returns 0
     having published nothing, so the reader hits `Requested address ... not
     found!`.
   - **`expandable_segments`** — also fails, mechanism **unverified**. The only
     log line is the reader's `not found!`; there is no `cuMemCreate` warning
     and no `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` /
     `cuMemAddressReserve` failure, so registration appears to have succeeded
     and something later did not. A plausible reading is that torch's VMM
     allocation never requests a FABRIC handle type, but that was inferred, not
     observed, and has not been checked against the wheel's actual source.

   So: not reachable with torch-allocated sources as they are today. That is
   weaker than "impossible" — allocating register mode's sources through
   mooncake's own allocator (see the `_prepare_source` note below) would sidestep
   the question entirely, and has not been tried.
3. **Registration fails silently.** `registerLocalMemory` calls
   `cuMemRetainAllocationHandle`, and when the memory did not come from
   `cuMemCreate` it logs a warning and returns **0** — success, having published
   nothing. The failure surfaces much later and on a different node as
   `Requested address … not found!` → `batch_transfer_sync_read … status -1`.
4. **`MC_FORCE_MNNVL` replaces RDMA rather than joining it.** A peer outside the
   NVLink clique gets no path at all, not a slower one. `ENABLE_MULTI_PROTOCOL`
   (comma-separated segments, per-buffer routing) is not compiled into the
   pinned wheel, so there is no adaptive rdma/nvlink fallback to lean on.

Also: `transfer_sync_read` returns **before** the copy completes on this path —
unsynchronised repeats measured 0.00–0.02ms for 512MB. Any adopter must
synchronise explicitly; this is a correctness issue, not just a benchmarking
artifact.

Consequence: adopting NVLink means moving sources into engine-allocated buffers
(`allocate_managed_buffer`, which appears nowhere in the shipped code today),
i.e. reintroducing the put-side copy register mode exists to avoid. It is a
choice between the two, not a combination. At current payloads the data plane is
0.12% of step time, so the case rests on headroom at larger scale rather than
on today's throughput.

## Effort estimate (original, pre-implementation)

Assumes someone fluent in both repos.

| Component | LOC | Effort |
|---|---|---|
| TE bootstrap + config plumbing (mirrors `MooncakeStoreClient.__init__`, `:56-135`) | ~80 | 0.5d |
| Pin table — base dedupe, refcount, strong refs, lock | ~60 | 1d |
| `put` | ~60 | 0.5d |
| Non-tensor path (reuses `allocate_empty_tensors`) | ~30 | 0.5d |
| `get` — endpoint grouping, registered receive buffer, batch read, slice out | ~120 | **2-3d** |
| `clear` — refcount → `unregister_memory` | ~30 | 0.5d |
| Bootstrap provider registration | ~50 | 0.5d |
| Tests (unit + 2-process e2e) | ~250 | 1.5d |
| Docs | ~100 | 0.5d |

**~700 LOC, 8-9 days to a prototype passing e2e on one node.** Add ~3 days for
multi-node bring-up (RDMA endpoint resolution and transport selection is where
this reliably overruns). **≈2.5 weeks.**

Multipliers if the open questions go badly:

- **Read hotspot needs re-seeding** — +1 week (consumers must register their
  receive buffers as new sources and the controller must track multiple holders
  per key).
- **Producer liveness turns out not to hold** in some path — +1-2 weeks, since
  the fallback ("register, but copy into the pool on producer teardown") is a
  materially larger design.

---

## Go/no-go measurement — run this first

Two days of instrumentation on a real GRPO step, before any implementation:

1. **Read amplification per produced byte.** Count total bytes read across
   `prev_lp` / `ref_lp` / `read_from_dataplane` / `train_from_meta` divided by
   bytes written at `kv_first_write`. This is the factor that would be forced
   through the producer's single NIC.
2. **Producer concentration.** How many distinct actors perform
   `kv_first_write` per step. One actor writing all 512 rows is the worst case;
   a wide producer set spreads the read load naturally and largely removes the
   hotspot concern.
3. **Device residency at put.** Fraction of put bytes that are CUDA-resident,
   and whether they are slices of larger arenas. Determines whether the HBM
   pinning risk is live.
4. **Baseline.** Current put/get wall-clock via
   `TransferQueue/scripts/performance_test/perftest.py`, to have something to
   beat.

**Proceed** if reads are host-resident, amplification is low, and producers are
plural. **Redesign first** if a single actor produces the step's bulk and
amplification is 3× or higher — in that case re-seeding is not an optimization,
it is a prerequisite.

---

## Related

- `nemo_rl/data_plane/README.md` — data-plane boundary, GRPO step flow, config
- `TransferQueue/docs/storage_backends/mooncake_gdr.md` — the existing GDR path
  and its staging-buffer pattern
- `Mooncake/docs/source/design/p2p-store.md` — register-without-copy precedent
- `Mooncake/docs/source/design/transfer-engine/index.md` — Segment / BatchTransfer model
