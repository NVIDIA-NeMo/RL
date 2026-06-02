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

| Config | weight transfer (`transfer_and_update_weights`) |
|---|---|
| **16 × TP2**, cold refit (version 1) | **31.78 s** |
| **16 × TP2**, warm refit (version 2) | **20.67 s** |
| 1 × TP1 smoke (reference) | 5.07 s (RDMA 8.82 GB @ 386 Gbps on the wire) |

Notes:
- Both mid-training refits of the 3-step run succeeded. The **cold** refit
  (31.78 s) includes NIXL agent init, publisher init, and per-worker scratch
  allocation; the **warm** refit (20.67 s, publisher cached) is the steady-state
  figure. The dispatcher refits the 16 instances sequentially (pause → pull →
  resume), and with the `full_tensor()` publish each receiver pulls the complete
  ~8.8 GB model.
- vLLM + NCCL comparison was **not run** (descoped by request).

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
