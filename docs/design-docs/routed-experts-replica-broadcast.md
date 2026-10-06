# Router-Replay Routes on the Replica-Group Broadcast (Design)

> **Status: proposed (2026-10-05).** Nothing in this document is implemented yet.
> Numbers marked *estimate* still need to be measured.

**Summary.** With router replay on, the training fetch builds one dense,
padded `routed_experts` table for a DP rank's whole batch on the replica-group
leader, then NCCL-broadcasts it to the other TP×CP×PP ranks. At 86 nodes that
table is 47 GiB per rank, and the broadcast stages up to 4× that on a
non-leader GPU, which OOMs. This design broadcasts the small **route fragments**
instead and lets **every rank assemble its routes locally**. It moves bytes
with a **bucketed, streamed NCCL broadcast**, and later assembles routes **per
packed microbatch** inside the training iterator, so no rank ever holds the
padded table.

```
today                                        proposed
leader: fetch fragments → dense table        leader: fetch fragments
        [rows, pad_seq, L, K] int16                  │ bucketed byte broadcast (≈ valid tokens only)
        │ broadcast (int16 → int32 wire)             ▼
        ▼                                    every rank: assemble locally
all ranks: 47 GiB table (GPU staging         (phase 1: CPU table; phase 2: per packed
           up to ~188 GiB) → OOM              microbatch on GPU, prefetched)
```

---

## 1. Problem

### 1.1 Where it happens

`TQWorkerMixin._fetch` (`nemo_rl/data_plane/worker_mixin.py`) runs on every
policy-worker rank for each training call:

```
_fetch(meta)                                   one call per DP rank per train step
├─ leader (replica-group rank 0)
│   ├─ dp_client.get_samples(...)              per-row fields, materialized "padded"
│   │     pad_to_seqlen = GLOBAL_FORWARD_PAD_SEQLEN
│   │     (max sequence over ALL DP ranks, rounded to sequence_length_round)
│   └─ _maybe_assemble_routed_experts          deferred routes (ROUTE_PASSTHROUGH)
│         _route_fragments_by_row → _fetch_route_fragments  (fragments from data plane)
│         routed = full([rows, pad_seq, num_moe_layers, top_k], SENTINEL, int16)
│         execute_route_plan(...) per row → routed[row, :len]
└─ _broadcast_batched_data_dict(group=replica_group)
      descriptor via broadcast_object_list, then per tensor:
        non-leader: torch.empty(shape)  on GPU
        int16: wire = tensor.to(int32); broadcast(wire); tensor = wire.to(int16)
then: _attach_or_repack_pack_metadata → train_microbatch → get_microbatch_iterator
      (each packed microbatch moves to GPU via data_dict.to("cuda"))
```

Sequence packing does not shrink this. Packing only attaches
`micro_batch_indices`, the row groups of each bin of up to `train_mb_tokens`
tokens. The padded table is built and broadcast before any bin exists.

### 1.2 Observed failure

Job 1053591: 86 nodes, `rlvr_sc_86n_mxfp8_capture_rdma.yaml`, Megatron
TP 4 × CP 4 × PP 1 over 128 training GPUs, so DP 8;
`train_global_batch_size` 8192; packing on with `train_mb_tokens` 73,728.
It trained steps 1–3, then failed in step 4:

```
torch.OutOfMemoryError: Tried to allocate 47.13 GiB. GPU 2 ... 44.60 GiB free,
this process has 231.33 GiB in use (156.90 GiB allocated, 49.04 GiB reserved-unallocated)
  at _broadcast_batched_data_dict  (tensor = wire.to(torch.int16))
```

The size matches the routes table exactly:

```
  1024 rows        8192 rows / 8 DP ranks, one training call per rank
× 73,536 tokens    longest sequence in the step, rounded to 64
× 56 layers        router_replay_dimensions(): num_layers with moe_layer_freq=1
× 6                moe_router_topk
× 2 B              int16
= 50,602,180,608 B = 47.127 GiB
```

The failure is in shared NeMo-RL code and does not depend on the data-plane
backend. The data plane only carries the per-row fragments.

### 1.3 Why it is so large

| Factor | Effect |
|---|---|
| Whole-rank batch | every row of the rank's step share is materialized in one call |
| Global padding | every row is padded to the longest sequence across all DP ranks |
| Layer count | 56 route layers, while the model has 23 MoE layers + 1 MTP. The vLLM capture uses the same 56, so this cannot change on one side only |
| int16 on NCCL | NCCL has no int16, so the code widens to int32: 2× wire bytes and 2 extra copies |
| Broadcast staging | non-leader peak ≈ 47 (receive) + 94 (int32 wire) + 47 (narrow back) ≈ 188 GiB |

Valid tokens are a small part of the table. *Estimate:* step 2 reported
41.8M valid tokens over 2 steps, about 20.9M per step, so about 2.6M per DP
rank against 1024 × 73,536 = 75.3M padded slots. That is about 3.5% filled,
and about 1.75 GB of real routes per rank at 336 B per token.

---

## 2. Use cases and requirements

| Use case | Requirement |
|---|---|
| Megatron train with router replay, leader-fetch + replica broadcast (TP/CP/PP > 1) | primary target: no rank stages the padded table on GPU |
| prev-logprob pass with router replay | same path and treatment (reference-logprob skips routes) |
| CP > 1 | routes are CP-sharded after packing (`_shard_routed_experts_for_cp`); the new path must feed the same input |
| PP > 1 | all stages in the replica group still receive identical routes |
| Sequence packing on / off, dynamic batching | bins are defined after the fetch; phase 2 must use the same bins |
| Fallback rows (missing fragments) | stay all-sentinel, model uses its own router; fallback counts recorded **once per replica group** (leader), as today |
| No router replay | unchanged |
| Small runs (12-node smoke) | no regression in step time or correctness |
| Every backend (simple, mooncake_cpu, nixl) | backend-agnostic; only `get_samples` is called |

Correctness invariant: each rank's routes are byte-identical to what the
leader's dense table would have held for the same rows and positions.

---

## 3. Options considered

| Option | GPU peak per non-leader | Bytes on the wire per rank | Change size | Verdict |
|---|---|---|---|---|
| A. Chunked dense broadcast (CPU → GPU chunk → bcast → CPU) | 1 chunk | 47 GiB, or 94 GiB if widened | small | stopgap only: still ships ~30× padding |
| B. Byte-view dense broadcast (`view(uint8)`, no int32) | 47 GiB | 47 GiB | tiny | not enough: 47 GiB > the 44.6 GiB that was free |
| **C. Broadcast fragments, assemble locally** | 1 bucket | ≈ fragment bytes (≈ valid tokens) | medium | **chosen, phase 1** |
| **D. C + per-microbatch assembly in the iterator, with prefetch** | ~2 packed microbatches | same as C | larger | **phase 2** |
| E. Every rank fetches fragments independently (`fetch_policy="independent"`) | 1 microbatch | 0 broadcast, 16× data-plane reads | small | possible fallback; multiplies data-plane load by the group size |

A and B remain useful as building blocks. The streamed broadcast in §5 is
option A's mechanism, generalized to any large host-resident byte buffer.

---

## 4. Design

### 4.1 Phase 1: broadcast fragments, assemble on every rank

```
leader                                   every rank in the replica group
_fetch_route_fragments(keys)             meta (with route plans in tags) is already
  → {staging_key: RouteFragment}         replicated to all ranks by
     routes, encoding, extras_metadata   run_all_workers_sharded_data
        │
        ▼ serialize
  index:  [(key, offset, nbytes, dtype, shape, encoding, meta_len)]  (broadcast_object_list)
  blob:   concatenated raw bytes of all fragments                    (bucketed byte broadcast, §5)
        │
        ▼ every rank: rebuild {staging_key: RouteFragment} from index + blob
_maybe_assemble_routed_experts(meta, data, fragments=...)   ← runs locally on each rank
  routed table on CPU (phase 1) — never staged whole on GPU
```

Code changes, all in `nemo_rl/data_plane/worker_mixin.py`:

1. Split `_maybe_assemble_routed_experts` into fetch (`_route_fragments_by_row`,
   leader only) and assemble (pure, given plans + fragments, every rank).
2. In `_fetch`, when the route-passthrough flag is set: the leader fetches
   fragments; the fragments go through the descriptor + bucketed broadcast; all
   ranks assemble. `routed_experts` is then dropped from
   `_broadcast_batched_data_dict` for this call.
3. `_route_fallback_counts.update(...)` runs on the leader only, preserving
   today's once-per-replica-group metric semantics.
4. Errors: the leader's fetch error travels in the existing `("error", ...)`
   payload of the first collective, so peers never hang.

Phase 1 still builds the padded table **on CPU** on every rank. Host memory
cost is ranks per node × table size, e.g. 4 × 47 GiB = 188 GiB per node in the
job above. That is the main limit of phase 1, and phase 2 removes it.

### 4.2 Phase 2: assemble per packed microbatch, with prefetch

Routes are only needed one packed microbatch at a time, in the packed layout
`[T ≤ train_mb_tokens, L, K]` that `process_microbatch` produces.

```
train_microbatch(data)                         fragments for the whole call: on CPU, every rank
  get_microbatch_iterator(data, ...)
    for bin i in micro_batch_indices:
      side stream:  assemble bin i+1 routes → pinned host → GPU (async)   ← prefetch
      main stream:  wait(event i); attach routed_experts[bin i]; forward/backward(i)
                                                GPU peak ≈ 2 bins × 73,728 × 56 × 6 × 2 B ≈ 100 MB
```

Requirements:
- The routes source must follow every row reorder and slice that
  `BatchedDataDict` applies between fetch and iteration. Implement it as a
  lazy column keyed by **sample id**, not row position, and resolve it against
  each bin's sample ids.
- Assemble directly in the layout `process_microbatch` expects. That is padded
  `[rows_in_bin, bin_pad_len, L, K]` today, or packed `[T, L, K]` once the packer
  consumes it. CP sharding stays where it is.
- Prefetch depth 1 with two buffers is enough. No extra collectives are
  needed, because every rank already holds the fragments after phase 1.

---

## 5. Streamed byte broadcast (the "chunked NCCL broadcast")

A reusable primitive used to move the fragment blob in phase 1, or any large
host-resident buffer.

### 5.1 NCCL facts it relies on

- **Device memory only.** NCCL sends from GPU buffers, so host data is staged
  through the GPU. Host↔GPU copies are fast and asynchronous only from
  **pinned** memory.
- **No int16 type.** NCCL types are int8/uint8, int32, int64, fp16, bf16, fp32,
  fp64 and fp8. Broadcast does no arithmetic, so any buffer can be viewed as
  `uint8` (`t.view(torch.uint8)` on a contiguous tensor) and sent unchanged.
  That removes today's int32 widening.
- **Ordering.** Collectives must be issued in the same order on every rank of
  the group. All buckets go on one comm stream, in index order.

### 5.2 Pipeline

```
leader:  pinned host ──H2D (copy stream)──► GPU buf[k%B] ──bcast (comm stream)──►
others:  ◄──bcast (comm stream)── GPU buf[k%B] ──D2H (copy stream)──► pinned host
         bucket k+1 copying          bucket k on the wire          bucket k-1 draining
         B = 2–3 rotating device buffers; CUDA events order copy ↔ comm per buffer
```

- Bucket boundaries pack **whole fragments**, so no fragment is split. A
  fragment larger than the bucket size gets a bucket of its own.
- Pinned host buffers are allocated once and reused across steps.
- GPU memory is fixed at B × bucket size, independent of batch size.

### 5.3 Choosing the bucket size

Each broadcast costs `t(n) = α + n/β`: a fixed cost α (kernel launch,
cross-rank sync, Python) and a per-byte cost 1/β. Efficiency is
`(n/β) / (α + n/β)`, which reaches 90% at `n ≈ 9·α·β`.

```
efficiency
100% ┤                         ________________   ← bandwidth plateau
 90% ┤                  ___----
 50% ┤         _-'                 knee ≈ 9·α·β
  0% ┼────┬─────┬──────┬──────┬──────┬──────► bucket size
         1MB  8MB   32MB  128MB  512MB  2GB
       latency-bound │  sweet spot │ no gain; more memory and start-up delay
```

| Bucket size | Throughput | Cost |
|---|---|---|
| below ~8 MB | fixed costs dominate, link idle | many collectives, Python overhead |
| knee to ~4× knee (≈ 32–256 MB, *estimate*) | ≥ 90% of link bandwidth | B × bucket of GPU staging |
| ≥ 1 GB | ≤ 1–2% more | memory; first bucket delays everything (bad for phase-2 prefetch) |

Bigger is better only up to the knee.

*Estimates* for the 86-node layout: α ≈ 50 µs per inter-node NCCL call plus
Python; β ≈ 25–50 GB/s per rank effective, if the 16-rank TP 4 × CP 4 group
spans 4 nodes over InfiniBand. That puts the knee near 10–25 MB, and 64–256 MB
buckets sit on the plateau. Moving ~1.75 GB of routes per rank then takes
about 35–70 ms per step, against a step of about 560 s.

How to set it:
1. Measure the knee with `nccl-tests` `broadcast_perf` on the real replica-group
   layout, sweeping 1 MB to 1 GB.
2. Default to about 2× the measured knee. Expose it as an env var
   (`NRL_ROUTE_BCAST_BUCKET_BYTES`) and record the chosen value in step metrics.
3. With phase-2 prefetch, prefer bucket boundaries aligned to bins even when
   that leaves some buckets small. Starting compute earlier beats the last few
   percent of link use.

---

## 6. Memory and time comparison (86-node job, per rank, estimates)

| Path | GPU peak for routes | Host memory for routes | Bytes broadcast |
|---|---|---|---|
| Today | ~188 GiB on non-leaders (OOM) | 47 GiB on the leader | 94 GiB (int32) |
| A. chunked dense | ~3 GiB at 1 GiB chunks | 47 GiB on every rank | 94 GiB, or 47 as uint8 |
| C / phase 1 | B × bucket (≤ 0.75 GB) | fragments + 47 GiB table on every rank | ≈ 1.75 GB |
| D / phase 2 | ~100 MB (2 bins) | fragments only | ≈ 1.75 GB |

---

## 7. Testing

| Level | Test | Pass criterion |
|---|---|---|
| Unit, 1 node / 4 GPUs (torchrun, NCCL) | streamed byte broadcast: random buffers, odd sizes, bucket smaller and larger than one fragment, B = 1/2/3 | byte-exact on all ranks; GPU peak ≤ B × bucket + ε |
| Unit | fragment serialize / rebuild round trip, every `encoding` | identical `RouteFragment`s |
| Unit | phase 1 assembly on every rank vs today's leader dense table, including fallback rows | `torch.equal` per row; fallback counts only on the leader |
| Unit (phase 2) | lazy routes column under reorder, slice and packing | per-bin routes equal the dense table's rows |
| E2E 12 nodes | router-replay capture smoke (`rlvr_sc_smoke_small_nixl.yaml`, seg 4) | 5 steps, masked ≈ 0, mult_prob_error ≈ 1.01, no step-time regression |
| E2E 86 nodes | `rlvr_sc_86n_mxfp8_capture_rdma.yaml` | 4+ steps without OOM; route broadcast time and bytes in step metrics |

---

## 8. Rollout

- `NRL_ROUTE_BCAST=dense|fragments` (default `dense` until both E2E tests
  pass), plus `NRL_ROUTE_BCAST_BUCKET_BYTES`.
- Step metrics: `route_bcast/bytes`, `route_bcast/buckets`, `route_bcast/ms`,
  and host memory used by assembly.
- Phase 2 behind its own flag, after phase 1 has run at 86 nodes.

## 9. Open questions

1. Real fragment volume per rank per step, as encoded on the wire. The
   1.75 GB figure assumes raw int16 per valid token.
2. Is the 56-layer route width needed? Shrinking it to the real MoE layers
   (23 + 1 MTP) needs the vLLM capture and the Megatron replay to change
   together.
3. Host-memory headroom for phase 1 on GB300 nodes running 4 ranks plus
   data-plane slabs.
4. Should `GLOBAL_FORWARD_PAD_SEQLEN` stay global for the other padded
   fields, or become per rank? This only matters once routes no longer
   dominate.
