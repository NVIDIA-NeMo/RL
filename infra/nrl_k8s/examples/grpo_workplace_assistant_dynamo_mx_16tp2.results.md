# Dynamo + ModelExpress v2 (NIXL RDMA) weight-transfer benchmark — 16×TP2

Measures mid-training weight-transfer (refit) duration for the **Dynamo + MX**
generation backend at a realistic fan-out: a 4-GPU DTensor trainer pushing
weights to 16 decode workers (TP=2 each, 32 inference GPUs).

## Setup

| | |
|---|---|
| Model | Qwen/Qwen3-4B-Thinking-2507 (dense, ~8.8 GB bf16) |
| Trainer | 4 GPU, DTensor TP=2 × DP=2 (1 GB300 node) |
| Generation | 16 × `VllmDecodeWorker`, `--tensor-parallel-size 2` (32 GPU, 8 nodes) |
| Weight sync | `cluster.weight_sync.method=mx` (ModelExpress v2, NIXL RDMA) |
| Recipe | `examples/nemo_gym/grpo_workplace_assistant_dynamo_mx_16tp2.yaml` |
| Infra | `infra/nrl_k8s/examples/grpo_workplace_assistant_dynamo_mx_16tp2.gb300.infra.yaml` |
| DGD | `infra/nrl_k8s/examples_dgd/qwen3_4b_thinking_gb300_mx_16tp2.yaml` |
| MX server | `jwillthomson/mx-8594fd6` (kavink/nemo_rl_moe build, arm64) |

Metric: the driver-log timing `prepare_for_generation/transfer_and_update_weights`
(nemo_rl/algorithms/grpo.py), which wraps the full refit (discover → RDMA pull →
load_weights) across all 16 workers.

## Result

### Dynamo + MX (NIXL RDMA)

| Config | weight transfer (`transfer_and_update_weights`) |
|---|---|
| **16 × TP2**, cold refit (version 1) | **31.78 s** |
| **16 × TP2**, warm refit (version 2) | **20.67 s** |
| **16 × TP2 + tree_scale_out**, cold refit (v=1) | **37.47 s** (+5.7 s vs no-fanout) |
| **16 × TP2 + tree_scale_out**, warm refit (v=2) | **23.90 s** (+3.2 s vs no-fanout) |
| **16 × TP2 + wave-parallel dispatcher (FANOUT=4)**, cold (v=1) | **15.25 s** (-16.5 s, **2.08× vs serial**) |
| **16 × TP2 + wave-parallel dispatcher (FANOUT=4)**, warm (v=2) | **8.27 s** (-12.4 s, **2.50× vs serial**) |
| 1 × TP1 smoke (reference) | 5.07 s (RDMA 8.82 GB @ 386 Gbps on the wire) |

#### Tree fan-out result (2026-06-03)

Re-ran with the post-fix overlay (modelexpress @ b4636c1 + dynamo @ bfc6adab) so the
inference_replica self-publish actually works (heartbeat-backed,
`model.named_parameters()` registered with NIXL at init). The MX server confirms
fan-out is structurally live — at v=2 discover, receivers see multiple source_ids
(trainer + 16 replicas, distinguished by tensor count in the GetMetadata log).

But the warm refit got **slower**, not faster. Two reasons:

1. **`pick_best_source` ranks trainer first** (`(role_priority, -training_step,
   -updated_at)` with trainer=0, inference_replica=1). With `same_rank_only=True`,
   every receiver finds the same-rank trainer at the requested version and picks
   it. The inference_replicas published at v=K are visible but never chosen —
   the trainer is always "better" by role.
2. **Per-pod publish + heartbeat overhead is paid on every refit**. The receiver
   has to register named_parameters with NIXL at init (cold path), then issue a
   `publish_self_as_source` RPC and start the heartbeat thread after each pull.
   That's net additional work per refit with no offsetting bandwidth win.

To actually exploit fan-out you'd need the picker to either round-robin among
trainer + replicas, prefer topology-local sources, or load-balance by
trainer-side counts. Tree fan-out as it stands today is correctly enabled for
**autoscaling boot** (where peers are the only source) but does not help the
static-pool refit path.

#### Wave-parallel dispatcher (2026-06-03)

The real win: parallelize the dispatcher itself. Replaced the sequential
`for inst in new_instances` in
`_dispatch_update_weights_via_mx_remote`
(`nemo-rl/nemo_rl/models/generation/dynamo/dynamo_generation.py:249`) with
exponential waves backed by a `ThreadPoolExecutor`: wave 1 = 4 pods in
parallel, wave 2 = 16, etc. Picker preference for inference_replicas + (later)
direct-copy load path for replica sources is staged for the tree-fanout
follow-up.

Even without tree fan-out, the per-NIC concurrency wins: at 16 × TP=2 all
pulling from one trainer NIC, NIXL/UCX multiplex the concurrent reads
better than the serial overhead they replace (per-receiver RPC + scratch
allocation + load_weights + KV cache reset). Net 2-2.5× cold/warm speedup
vs the serial baseline.

#### Tree fan-out followups

Wave-parallel works. Tree fan-out on top of it does not, for a chain of
reasons each of which needs a real fix:

1. **`publish_self_as_source` doesn't emit a `shape_registry`.** Receivers
   pulling from replicas have no per-tensor global_shape to view scratch
   bytes against. A receiver-side fallback (read shapes from the
   receiver's own `named_parameters`) compiles but doesn't actually
   describe what's on the wire — there's a hidden 2× size mismatch
   (777,912,320 bytes when 388,956,160 expected) that suggests the
   replica's NIXL descriptor isn't what we think.
2. **MX server sources have `TTL=-1` in Redis.** Sources from prior runs
   (different fleets, different TP sizes) linger forever. Stale sources
   pass `list_sources(status_filter=READY)` and pollute discover. A
   60-second-recency filter in the picker helps but doesn't catch
   sources that are getting heartbeat'd by something we can't see.
3. **Picker has no TP-size compatibility check.** A TP=2 receiver can
   pick a TP=1 replica from a different DGD (proven during this debug
   pass — the long-running autoscaling-demo TP=1 DGD was the smoking
   gun). The `publish_self_as_source` identity sets
   `tensor_parallel_size=0` as a sentinel, so even if we filter by it,
   replicas don't carry the real value.

Each of these is a small fix in isolation, but together they're a
quality-of-implementation sweep through `nemo_rl_v2.py` and the
dynamo extension. Punted as separate work — wave-parallel alone is
the deliverable here.

### vLLM + NCCL (collective broadcast) — baseline

Same model + fan-out (4-GPU DTensor trainer → 16 TP2 vLLM engines, 32 inference
GPUs), but non-colocated vLLM with the stock NCCL collective weight transfer
(`broadcast_weights_for_collective` → `update_weights_from_collective`).
Recipe/infra: `grpo_workplace_assistant_vllm_16tp2.{yaml,gb300.infra.yaml}`.

| Config | weight transfer (`transfer_and_update_weights`) |
|---|---|
| **16 × TP2**, cold refit (version 1) | **0.22 s** |
| **16 × TP2**, warm refit (version 2) | **0.18 s** |

### MX vs NCCL

NCCL is ~100× faster here (**0.18–0.22 s** vs **20.67–31.78 s**). NCCL is a single
optimized collective broadcast — all workers receive in parallel over
NVLink/RoCE. The MX path, as implemented for this benchmark, pays for: sequential
per-worker HTTP dispatch (pause → pull → resume × 16), per-worker scratch
allocation + vLLM `load_weights`, and the `full_tensor()` allgather added to make
sharded-trainer publishing correct. So this compares a mature collective against
an unoptimized rank-to-rank RDMA path — the MX number is an upper bound, not the
floor of what NIXL RDMA can do (the wire transfer itself was 8.82 GB @ 386 Gbps =
0.18 s in the 1-worker smoke).

Notes:
- Both MX mid-training refits of the 3-step run succeeded (cold 31.78 s includes
  NIXL/publisher/scratch init; warm 20.67 s is steady state).
- vLLM+NCCL: weight transfer is a negligible fraction of step time (0.3%).

## Getting it working required a chain of fixes (see memory
`project_mx_dynamo_ucx_cuda_blocker`)

The MX path worked at TP1/1-GPU but not at 16×TP2; each blocker below was masked
by the single-worker smoke:

1. **MX server image** must be the `kavink/nemo_rl_moe` build (`jwillthomson/mx-8594fd6`,
   arm64-pinned), not stock `modelexpress-server:0.3.0` (the stock build drops the
   v2 `extra_parameters` round-trip).
2. **`device_id`** in `dtensor_policy_worker.py:stream_weights_via_mx` →
   `torch.cuda.current_device()` (was `self.rank`; broke for >1 training GPU).
3. **Publish-before-pull + dispatcher retry** in the MX refit path
   (`grpo.py`, `dynamo_generation.py`).
4. **Do NOT set `HF_HUB_OFFLINE` on the DGD.** Offline makes vLLM resolve `--model`
   to the local snapshot *path*, so the receiver's `discover_v2_sources` model_name
   filter no longer matches the trainer's published HF-id → "no v2 source
   available". Keep `HF_HOME` at the populated shared cache (online, cache-backed)
   to avoid the 16-pod HF-429 storm while preserving the HF-id model name.
5. **`full_tensor()` publish** in `stream_weights_via_mx` (was `to_local()`): a
   sharded multi-GPU trainer's local shard didn't match the global shape the
   receiver reshapes to ("shape '[151936, 2560]' invalid for input of size
   97239040"). Gathering the full tensor fixes it (trades away the no-allgather
   optimization; revisit for MoE/EP).
6. **Recreate the DGD (fresh pods)** before a real run — stale NIXL agent state
   from prior failed refits causes `loadRemoteMD … NIXL_ERR_NOT_ALLOWED`.
