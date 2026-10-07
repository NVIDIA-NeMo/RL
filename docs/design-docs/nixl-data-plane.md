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
client (trainer / vLLM / controller)                 NixlStorageUnit (CPU actor)
  put:  ZMQ alloc ───────────────────────────────►    slab allocator
        NIXL WRITE (zero-copy or pooled buffer) ──►   registered DRAM slab
  get:  NIXL READ data + blob footer ◄─────────────   (no control message when the
        (footer tag checked against the blob id)       location hint is live)
  clear: ZMQ release_many (fire-and-forget) ───────►  quarantine, then free
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
    ├── blobstore.py                unit-slab and file stores, hinted reads
    ├── control.py                  ZMQ ROUTER/DEALER control plane
    ├── nixl_io.py                  NIXL endpoint, RDMA-only policy check
    └── allocator.py, directory.py, placement.py, bufpool.py, errors.py
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
    num_storage_units: 68          # e.g. one per Ray node
    unit_slab_bytes: 17179869184   # 16 GiB pinned per unit
```

The transport is RDMA only. Every client and unit checks the UCX environment
before creating a NIXL backend and refuses TCP:

```bash
export UCX_TLS=rc,self,sm
export UCX_NET_DEVICES=<rdma dev>:1,<rdma dev>:1   # RDMA devices, never eth0
```

NIXL's UCX backend honours `UCX_*` env vars over its own parameters, so the
environment is the only reliable lever.

## Design points

- **Placement-free locations.** TransferQueue stores per-key location meta
  `{b, o, n, k, d, s}` plus a hint `{u, so, z, g}` (unit, slab offset, blob
  size, unit-instance stamp). A live hint lets reads skip every control
  message; after a checkpoint restore the stamp no longer matches, and reads
  fall back to a directory lookup and a pinned `resolve`.
- **Safe reads without pins.** Units quarantine freed regions for
  `read_pin_s`, and each blob's footer carries its id, which the reader checks
  in the same READ.
- **Put path.** Values of 1 to 32 MiB go zero-copy from the caller's memory.
  Larger values use torch's parallel copy into a registered buffer pool.
  Registering a fresh 1 GiB mapping costs about 70 ms, while copying it costs
  about 10 ms.
- **Sockets.** With `numa: auto`, the k-th unit on a multi-socket node pins its
  CPUs and slab memory to socket `k % sockets`. NICs are not restricted per
  socket: doing so broke cross-node UCX endpoint creation at 86 nodes.
- **Checkpointing.** `save_data_plane` writes one shard per unit plus a
  manifest. Restore re-places blobs on whichever units exist then.

## Measured (GB300, 2 nodes, 4 RDMA rails; ms, lower is better)

| op, size | Mooncake CPU | NIXL |
|---|---|---|
| put 4 KB | 7.0 | 5.6 |
| put 256 MB | 17.3 | 15.1 |
| get same-node 1 GB | 99 | 89 |
| get cross-node 1 GB | 136 | 99–102 |

At 86 nodes with RDMA, data-plane time per RL step was about 11 s for NIXL
against 16.5 s for Mooncake CPU.
