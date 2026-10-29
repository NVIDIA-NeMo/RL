# NIXL Data Plane (`data_plane.backend: nixl`)

> **Experimental.** A TransferQueue storage backend (`NixlStore`) that moves
> rollout and training columns with one-sided NIXL RDMA.

The NIXL data plane separates storage from compute. CPU `NixlStorageUnit` Ray
actors each own one registered DRAM slab. Trainer, vLLM and controller processes
are clients: they move bytes with NIXL READ/WRITE straight into or out of unit
slabs, and send small control messages to units over ZMQ. Ray only starts the
units. TransferQueue keeps keys, fields and readiness, so the NeMo-RL adapter is
unchanged apart from selecting the backend.

```
client (trainer / vLLM / controller)                 NixlStorageUnit gidx % N
  put:  ZMQ alloc ───────────────────────────────►    slab allocator
        NIXL WRITE (zero-copy or pooled buffer) ──►   registered DRAM slab
        ZMQ commit ──────────────────────────────►    publish rows in the row index
  get:  ZMQ resolve_keys ────────────────────────►    look up rows, pin their blobs
        NIXL READ entries ◄──────────────────────     (one READ per unit), then unpin
  clear: ZMQ release_rows (awaited) ─────────────►    drop rows; free unpinned blobs
```

## Code layout

Same split as the Mooncake backend: TransferQueue is imported only under
`adapters/`, store-side code has no TQ dependency.

```
nemo_rl/data_plane/
├── interfaces.py                   NixlStoreConfig (data_plane.nixl block)
├── adapters/
│   ├── transfer_queue.py           backend: nixl branch -> imports tq_nixl
│   ├── tq_nixl.py                  TQ client, storage manager, bootstrap provider
│   └── tq_nixl_checkpoint.py       storage save / restore (cf. tq_mooncake_checkpoint.py)
├── nixl_storage_unit.py            CPU Ray actor owning one registered DRAM slab
└── nixl/                           store side
    ├── blob_format.py              blob byte layout (index + footer with blob tag)
    ├── blobstore.py                unit-slab store: put, resolve + read, clear
    ├── control.py                  ZMQ ROUTER/DEALER control plane
    ├── nixl_io.py                  NIXL endpoint, RDMA-only policy check
    └── allocator.py, directory.py, bufpool.py, errors.py
nemo_rl/distributed/numa_utils.py   socket detection and binding (numa: auto)
nemo_rl/utils/checkpoint_engines/nixl.py   shared NIXL agent factory
```

## Enabling it

```yaml
data_plane:
  enabled: true
  impl: transfer_queue
  backend: nixl
  nixl:
    num_storage_units: null        # default: one per Ray node (2x measured no faster)
    unit_slab_bytes: 17179869184   # 16 GiB pinned per unit
```

The transport is RDMA only. Every client and unit checks the UCX environment
before creating a NIXL backend and refuses TCP:

```bash
export UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm      # the default when unset
export UCX_NET_DEVICES=<rdma dev>:1,<rdma dev>:1   # RDMA devices, never eth0
```

NIXL's UCX backend honours `UCX_*` env vars over its own parameters, so the
environment is the only reliable lever.

The `UCX_TLS` default is the list the NIXL checkpoint-engine refit uses
(`refit_transport: nixl`, see the checkpoint-engine refit guide). The driver's
`UCX_TLS` reaches every Ray worker, and the refit needs `cuda_copy` to register
its GPU buffers (without it, refit hangs). `cuda_copy` and `cuda_ipc` never
carry the data plane's CPU bytes off the node, so its cross-node traffic still
goes over `rc` (RDMA) only. If you set `UCX_TLS` yourself with both features
on, keep `cuda_copy,cuda_ipc` in it.

### Images with HPC-X: set `NCCL_NET_PLUGIN=none`

On images that ship HPC-X's NCCL network plugin
(`/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so`), NIXL can fail to start
in the trainer processes with `NIXL_ERR_BACKEND`, after UCX warns that `rc`,
`cuda_copy`, `cuda_ipc` and the `mlx5_*` devices are "not available":

```
NCCL first collective ─▶ loads HPC-X SHARP net plugin ─▶ HPC-X libucs in process
NIXL starts its bundled UCX ─▶ finds its plugin dir via the loaded HPC-X libucs
  ─▶ HPC-X's IB/CUDA plugins register with the wrong UCX
  ─▶ NIXL's UCX keeps only tcp/shm ─▶ NIXL_ERR_BACKEND
```

Fix: launch with `NCCL_NET_PLUGIN=none`. NCCL then uses its built-in
InfiniBand transport over the same rails, HPC-X's UCX is never loaded, and
NIXL finds its own plugins. Seen on GB200 with the `nightly-gym.2026-10-19`
image (verified on 1 and 64 nodes); the GB300 runs in this doc did not need it.
Effect on NCCL collective speed (no SHARP plugin) was not measured.

## Design points

- **Index-addressed rows.** Like TransferQueue's SimpleStorage, row `gidx`
  lives on unit `unit_ids[gidx % N]`, which keeps the row's
  `field -> (blob, entry)` index. A put returns no location meta, so the TQ
  controller holds no more than it does for SimpleStorage, and a new client
  needs only the unit table.
- **Pinned reads.** `resolve_keys` pins the blobs it returns for `read_pin_s`
  (at least `transfer_timeout_s` + margin); a freed region is reused only
  after its pins end.
- **Put path.** Values of 1 to 32 MiB go zero-copy from the caller's memory.
  Larger values use torch's parallel copy into a registered buffer pool.
  Registering a fresh 1 GiB mapping costs about 70 ms, while copying it costs
  about 10 ms.
- **Sockets.** With `numa: auto`, the k-th unit on a multi-socket node pins its
  CPUs and slab memory to socket `k % sockets`. NICs are not restricted per
  socket: doing so broke cross-node UCX endpoint creation at 86 nodes.
- **Checkpointing.** `save_data_plane` writes one shard per unit, its live
  rows, and a manifest. With the same units, each shard restores in place;
  with a different count, the live rows are put again through a client.

## Measured (GB300, 2 nodes, 4 RDMA rails; ms, lower is better)

| op, size | Mooncake CPU | NIXL |
|---|---|---|
| put 4 KB | 7.0 | 5.6 |
| put 256 MB | 17.3 | 15.1 |
| get same-node 1 GB | 99 | 89 |
| get cross-node 1 GB | 136 | 99–102 |

At 86 nodes with RDMA, data-plane time per RL step was about 11 s for NIXL
against 16.5 s for Mooncake CPU.
