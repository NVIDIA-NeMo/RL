# MXFP8→MXFP8 Refit Enablement for `nccl_reshard_refit` — Upstreaming Handoff

**Status: fully implemented and validated end-to-end on GB200 (2026-07-20).**
This document is a self-contained work order. Given this file and the repos below, an
agent should be able to (1) re-apply and verify the changes, and (2) create all
upstream PRs / issues without any other context.

- Working branch with all changes (uncommitted at time of writing): `youngeunk/new-refit-mxfp8`
  in the NeMo-RL checkout at
  `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/youngeunk/mount/nemo-rl-backup`
  (branched from `origin/youngeunk/new-refit-downsize` @ `097850d1f "canonical version"`).
- Megatron-Bridge checkout (contains 2 required uncommitted changes):
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` @ `554c7b932`.
- Related design doc already upstream: `docs/design-docs/nccl-reshard-refit.md`.

---

## 1. What this enables and why

`nccl_reshard_refit` (a.k.a. the xferdtensor bulk-reshard refit) byte-copies training
weights from Megatron train workers directly into vLLM generation workers. Before this
work, FP8 support was **blockwise-only** (`fp8_recipe=blockwise`, Hopper-era). On
Blackwell (GB200), the required FP8 recipe is **MXFP8** (E8M0 block scales, 32-element
blocks), and the goal recipe
`examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off-mxfp8-rollout.yaml`
previously supported only **BF16-train → MXFP8-rollout** (gen quantizes on receive).

This work makes **MXFP8-train → MXFP8-gen** refit work: the train side stores TE
`MXFP8Tensor` params (`fp8_param=true`), and the refit transfers the raw FP8 bytes plus
their E8M0 scales with **no requantization anywhere** (bit-exact train→gen weights).

Key architectural facts an implementer must know:

- The reshard splits params into two paths: **bulk** (FFN gate/up/down weights via
  xferdtensor P2P; whitelist in `is_nccl_reshard_param`) and **misc**
  (everything else — attention, embeddings, norms, **and all FP8 scale tensors** — via
  `packed_broadcast` + vLLM `load_weights`).
- For FP8 export, the Bridge's `build_export_fp8_tasks` emits a *pair* of tasks per FP8
  weight: the FP8 data tensor and a scale tensor named `<weight_name><suffix>`.
- vLLM's MXFP8 mode (`vllm_cfg.is_mx=true`, quantization `modelopt_mxfp8`) names its
  scale receive-buffers `<name>_scale_from_checkpoint` (an **unswizzled** E8M0 uint8
  param; `process_weights_after_loading` re-swizzles into the live `weight_scale`).
  The blockwise mode uses `<name>_scale_inv`. The suffix MUST match or the misc
  producer/consumer desynchronize.
- TE scale-tensor layouts differ: blockwise `_rowwise_scale_inv` is **2D-scaled**
  (rows AND cols divided by 128); MXFP8 `_rowwise_scale_inv` is **1D-scaled**
  (cols ÷ 32, rows untouched, dtype uint8/E8M0).

---

## 2. Validated results (acceptance criteria for re-verification)

| Run | Config | Result |
|---|---|---|
| 4B canary (`script/new_refit/4b_mxfp8_to_mxfp8_dp_dp.sh`, 4 nodes GB200) | Qwen3-4B, train DP8 fp8_param mxfp8 → gen DP8 is_mx, reshard | **Step 10/10**, weight_sync 0.23–0.31 s ×10, **KL 0.0055–0.0080** |
| Goal recipe (`script/new_refit/qwen3_32b_mxfp8_rollout_reshard.sh`, 8 nodes GB200) | qwen3-32b mxfp8-rollout recipe + reshard + train fp8_param mxfp8 | **Step 10/10 in 33 min**, xfer_frac 71.3%, weight_sync 0.63–0.94 s ×10, **KL 0.0055–0.0059** |
| Reference baseline (recipe stock, `nccl_reshard_refit=false`) | BF16 train → mxfp8 rollout, collective refit | Step 10/10, KL 0.0043–0.0047 |

Interpretation: KL in the 0.005–0.008 range is the expected MXFP8 quantization-error
band (the reshard variant is slightly above the baseline because it quantizes the
attention projections too — see §4 config notes). Anything ≫ 0.01 indicates broken
scales; identical-to-BF16 (≈0.0007 at 4B) indicates FP8 was silently not used.

---

## 3. The changes — split by upstream destination

### 3.1 PR #1: NeMo-RL (4 files, ~41 lines)

Apply to the NeMo-RL repo (github.com/NVIDIA-NeMo/RL). Full diff:

```diff
diff --git a/nemo_rl/models/generation/vllm/quantization/fp8.py b/nemo_rl/models/generation/vllm/quantization/fp8.py
--- a/nemo_rl/models/generation/vllm/quantization/fp8.py
+++ b/nemo_rl/models/generation/vllm/quantization/fp8.py
@@ -436,6 +436,13 @@ def load_weights(weights, model_runner):
         if not _is_fp8_weight(k, model):
             weights_quantized.append((k, v))
             continue
+        # Already-quantized input (fp8_param training + nccl_reshard misc path):
+        # the FP8 weight bytes pass through as-is and the matching E8M0/scale
+        # tensor arrives as its own "<name>_scale_from_checkpoint"/"_scale_inv"
+        # entry in the same stream — do NOT requantize.
+        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
+            weights_quantized.append((k, v))
+            continue
         # Cast the weight into fp8 and its scale factor
         if global_fp8_config.is_mx:
             from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
diff --git a/nemo_rl/models/generation/vllm/vllm_worker_async.py b/nemo_rl/models/generation/vllm/vllm_worker_async.py
--- a/nemo_rl/models/generation/vllm/vllm_worker_async.py
+++ b/nemo_rl/models/generation/vllm/vllm_worker_async.py
@@ -1536,7 +1536,10 @@ class VllmAsyncGenerationWorkerImpl(BaseVllmGenerationWorker):
             import traceback
 
             traceback.print_exc()
-            return False
+            # Re-raise so the driver's ray.get surfaces the failure immediately;
+            # returning False here left the train side broadcasting misc params
+            # to consumers that had already died (silent cluster-wide hang).
+            raise
 
     async def reset_prefix_cache_async(self):
         """Async version of reset_prefix_cache."""
diff --git a/nemo_rl/models/policy/workers/megatron_policy_worker.py b/nemo_rl/models/policy/workers/megatron_policy_worker.py
--- a/nemo_rl/models/policy/workers/megatron_policy_worker.py
+++ b/nemo_rl/models/policy/workers/megatron_policy_worker.py
@@ -1895,12 +1895,12 @@ class MegatronPolicyWorkerImpl(
         self._require_remote_sparse_refit().finish(succeeded)
 
     def _is_fp8_export(self) -> bool:
-        """Return True if the train side stores weights as TE blockwise FP8."""
+        """Return True if the train side stores weights as TE FP8 (blockwise or MXFP8)."""
         if self.fp8_cfg is None:
             return False
         return bool(
             self.fp8_cfg.get("fp8_param", False)
-            and self.fp8_cfg.get("fp8_recipe") == "blockwise"
+            and self.fp8_cfg.get("fp8_recipe") in ("blockwise", "mxfp8")
         )
 
     def _build_refit_conversion_tasks(self) -> list:
@@ -1912,8 +1912,18 @@ class MegatronPolicyWorkerImpl(
         scale tensor).
         """
         if self._is_fp8_export():
+            # vLLM's MXFP8 (is_mx) loader expects scales named
+            # "<name>_scale_from_checkpoint" (unswizzled E8M0 receive buffer);
+            # the blockwise loader expects "<name>_scale_inv".
+            scale_suffix = (
+                "_scale_from_checkpoint"
+                if self.fp8_cfg.get("fp8_recipe") == "mxfp8"
+                else "_scale_inv"
+            )
             return self.megatron_bridge._model_bridge.build_export_fp8_tasks(
-                self.megatron_bridge.hf_pretrained, [self.model]
+                self.megatron_bridge.hf_pretrained,
+                [self.model],
+                scale_inv_suffix=scale_suffix,
             )
         return [
             task
@@ -2082,7 +2092,7 @@ class MegatronPolicyWorkerImpl(
             if local_tensor is None:
                 continue  # non-local PP rank
             # FP8 scale siblings take the misc path.
-            if task.global_param_name.endswith("_scale_inv"):
+            if task.global_param_name.endswith(("_scale_inv", "_scale_from_checkpoint")):
                 continue
 
             if isinstance(task.mapping, GatedMLPMapping):
@@ -2348,7 +2358,7 @@ class MegatronPolicyWorkerImpl(
         def _task_is_misc(task) -> bool:
             # FP8 scale siblings carry the suffix on global_param_name and are
             # always misc (packed_broadcast).
-            if task.global_param_name.endswith("_scale_inv"):
+            if task.global_param_name.endswith(("_scale_inv", "_scale_from_checkpoint")):
                 return True
             # Compound mappings (QKV/GatedMLP) export homogeneous sub-params
             # (all nccl-reshard or all misc), so the first HF name is representative.
diff --git a/nemo_rl/weight_sync/nccl_reshard_utils.py b/nemo_rl/weight_sync/nccl_reshard_utils.py
--- a/nemo_rl/weight_sync/nccl_reshard_utils.py
+++ b/nemo_rl/weight_sync/nccl_reshard_utils.py
@@ -223,6 +223,11 @@ _STR_TO_DTYPE = {
     "float32": torch.float32,
     "float8_e4m3fn": torch.float8_e4m3fn,
     "float8_e5m2": torch.float8_e5m2,
+    # E8M0 block-scale tensors (MXFP8 *_scale_from_checkpoint) are uint8.
+    "torch.uint8": torch.uint8,
+    "uint8": torch.uint8,
+    "torch.int32": torch.int32,
+    "int32": torch.int32,
 }
 
 
@@ -630,11 +635,18 @@ def check_nccl_reshard_refit_support(master_config: dict) -> None:
                     "policy.megatron_cfg.fp8_cfg.fp8_param=True "
                     "(BF16→FP8 train-side quantization is not implemented yet)."
                 )
-            elif fp8_recipe != "blockwise":
+            elif fp8_recipe not in ("blockwise", "mxfp8"):
                 violations.append(
                     "policy.megatron_cfg.fp8_cfg.fp8_recipe must be 'blockwise' "
-                    f"when fp8_param=True (got {fp8_recipe!r}); other recipes "
-                    "don't produce export-ready scale_inv tensors."
+                    f"or 'mxfp8' when fp8_param=True (got {fp8_recipe!r}); other "
+                    "recipes don't produce export-ready scale_inv tensors."
+                )
+            elif fp8_recipe == "mxfp8" and not vllm_cfg.get("is_mx", False):
+                violations.append(
+                    "policy.megatron_cfg.fp8_cfg.fp8_recipe='mxfp8' requires "
+                    "policy.generation.vllm_cfg.is_mx=True so the vLLM side "
+                    "builds MXFP8 (E8M0 block-scale) params to receive the "
+                    "transferred weights."
                 )
```

Per-change rationale (use in the PR description):

| Change | Failure it fixes (observed empirically) |
|---|---|
| `fp8.py` fp8-dtype passthrough | `NotImplementedError: mxfp8_quantize only supports input tensor with dtypes fp16/bf16` — the misc consumer re-quantized already-FP8 attention weights. Also independently makes the **collective** refit path viable for fp8_param training. |
| `vllm_worker_async.py` re-raise | Any gen-side refit exception was swallowed into `return False`; because `sync_weights` waits on **train** futures first, the train side then broadcast misc params to dead consumers forever → cluster-wide idle-GPU hang, jobs reaped by watchdogs with no visible error. This single change turned a 40-min silent hang into an 8-min loud failure. |
| `_is_fp8_export` + `scale_inv_suffix` | Without the recipe check: no FP8 export at all. Without the suffix: gen expects `_scale_from_checkpoint`, train exported `_scale_inv` → misc-phase name mismatch. |
| Misc routing (2 sites) | The new suffix bypassed both hardcoded `endswith("_scale_inv")` checks, so scale tensors entered the **bulk** reshard planner and failed shape validation (`Source rank local shape (2560, 304) does not match planned shape (2560, 9728)`). |
| `_STR_TO_DTYPE` uint8 | `KeyError: 'torch.uint8'` in gen's `_receive_and_load_misc_params` on the E8M0 scale metadata — the proximate cause of the silent hang above. |
| Gate (`check_nccl_reshard_refit_support`) | Fails fast with actionable messages instead of undefined behavior on unsupported / half-configured setups. |

Suggested extra hardening for the PR (not implemented, recommended): in the gen-side
`_receive_and_load_misc_params`, replace the silent `if not misc_meta: return` with an
assertion when the train side reported nonzero misc bytes — a silently skipped misc
phase leaves garbage weights and corrupts KL invisibly.

### 3.2 PR #2: Megatron-Bridge (2 files, ~25 lines)

Apply to github.com/NVIDIA-NeMo/Megatron-Bridge (checkout used: `554c7b932`).
NeMo-RL must bump its Bridge pin after this merges — **the NeMo-RL PR does not work
against a stock Bridge.**

```diff
diff --git a/src/megatron/bridge/models/conversion/model_bridge.py b/src/megatron/bridge/models/conversion/model_bridge.py
--- a/src/megatron/bridge/models/conversion/model_bridge.py
+++ b/src/megatron/bridge/models/conversion/model_bridge.py
@@ -1687,6 +1687,14 @@ class MegatronModelBridge(
             from transformer_engine.pytorch.tensor import Float8BlockwiseQTensor
         except Exception:
             Float8BlockwiseQTensor = None
+        try:
+            from transformer_engine.pytorch.tensor import MXFP8Tensor
+        except Exception:
+            try:
+                from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
+            except Exception:
+                MXFP8Tensor = None
+        _fp8_tensor_types = tuple(t for t in (Float8BlockwiseQTensor, MXFP8Tensor) if t is not None)
 
         for vp_stage, model in enumerate(megatron_model):
             for local_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
@@ -1707,9 +1715,9 @@ class MegatronModelBridge(
                 # - Some initialization paths may leave scale tensors unset; we should not emit
                 #   a scale task in that case (would break deterministic export/consumer assumptions).
                 is_blockwise_fp8 = False
-                if Float8BlockwiseQTensor is not None:
+                if _fp8_tensor_types:
                     try:
-                        is_blockwise_fp8 = isinstance(local_weights, Float8BlockwiseQTensor)
+                        is_blockwise_fp8 = isinstance(local_weights, _fp8_tensor_types)
                     except Exception:
                         is_blockwise_fp8 = False
 
diff --git a/src/megatron/bridge/models/conversion/param_mapping.py b/src/megatron/bridge/models/conversion/param_mapping.py
--- a/src/megatron/bridge/models/conversion/param_mapping.py
+++ b/src/megatron/bridge/models/conversion/param_mapping.py
@@ -3299,15 +3299,23 @@ def split_qkv_weights(
                     f"Cannot infer block divisor for qkv tensor: "
                     f"provider.hidden_size={orig_hidden_size} is not divisible by qkv.shape[-1]={current_last_dim}"
                 )
-            divisor = orig_hidden_size // current_last_dim
-            if head_size % divisor != 0:
+            hidden_size = current_last_dim
+            # Rows may or may not be compressed: blockwise FP8 scale_inv is
+            # 2D-scaled (rows AND cols divided by the block), while MXFP8
+            # rowwise scale_inv is 1D-scaled (cols divided by 32, rows full).
+            # Derive the row compression from the actual row count instead of
+            # assuming it matches the column divisor.
+            rows = qkv.shape[0]
+            if rows == qkv_total_dim * head_size:
+                scaled_head_size = head_size
+            elif rows % qkv_total_dim == 0:
+                scaled_head_size = rows // qkv_total_dim
+            else:
                 raise ValueError(
-                    f"Cannot scale head_size for qkv tensor: "
-                    f"head_size={head_size} is not divisible by divisor={divisor} "
-                    f"(provider.hidden_size={orig_hidden_size}, qkv.shape[-1]={current_last_dim})"
+                    f"Cannot infer per-head rows for qkv scale tensor: "
+                    f"rows={rows} is not divisible by qkv_total_dim={qkv_total_dim} "
+                    f"(head_size={head_size}, qkv.shape[-1]={current_last_dim})"
                 )
-            hidden_size = current_last_dim
-            scaled_head_size = head_size // divisor
 
         qkv_reshaped = qkv.view(qkv_total_dim, scaled_head_size, hidden_size)
```

Per-change rationale:

| Change | Failure it fixes |
|---|---|
| `model_bridge.py` MXFP8Tensor detection | `_detect_fp8_params` isinstance-checked only `Float8BlockwiseQTensor`, so MXFP8 params were never detected as FP8 → no scale tasks emitted. The rest of the export path (`_rowwise_data` + `_fp8_dtype` view) was already format-generic and needs no changes. |
| `param_mapping.py` QKV scale split | `shape '[48, 4, 80]' is invalid for input of size 491520`: the fused-QKV splitter assumed the scale tensor's rows were compressed by the same block divisor as its columns (true for blockwise 2D scales, false for MXFP8 1D scales). New logic infers row compression from the actual row count — strictly more general, blockwise-compatible. |

Bridge note: `_trim_blockwise_fp8_scale_inv_padding` currently warns and no-ops for
MXFP8 (`is_2d_scaled` is falsy). This was benign for all tested shapes (Qwen3-4B/32B —
all scale dims naturally aligned); a model with non-128-multiple row dims or
non-4-multiple `K/32` may need MXFP8-aware trimming there. Mention in the PR as a
known follow-up.

### 3.3 Issue #3: Megatron-LM bug report (workaround only, no patch)

**Title:** `fp8_param=True + distributed optimizer + overlap_param_gather crashes at
the second train step (freed param buffer)`

**Symptom:** first train step succeeds; on the next forward, the DDP overlap
param-gather pre-forward hook fails:

```
File megatron/core/distributed/param_and_grad_buffer.py, line 291, in _post_param_sync
    param_slice = bucket.param_data.view(-1)[param_start:param_end]
RuntimeError: setStorage: sizes [15728640], strides [1], storage offset 3607101440,
and itemsize 2 requiring a storage size of 7245660160 are out of bounds for storage of size 0
```

i.e. `bucket.param_data`'s storage has been freed. Reproduced identically on two
completely independent weight-sync paths (NeMo-RL collective refit and nccl_reshard
refit), Qwen3-4B and Qwen3-32B, GB200, `fp8_recipe=mxfp8`, `fp8_param=true`,
`use_distributed_optimizer=true`, `overlap_param_gather=true`,
`use_precision_aware_optimizer=false`. Megatron-LM version: the submodule pinned by
Megatron-Bridge `554c7b932` (see `3rdparty/Megatron-LM` therein).

**Verified workaround:** `distributed_data_parallel_config.overlap_param_gather=false`
(params gather synchronously; modest perf cost). With the workaround, 10/10 training
steps complete at both scales.

### 3.4 Config-level requirements (goes into the NeMo-RL PR as recipe/doc changes)

A working mxfp8→mxfp8 reshard run needs these overrides (validated combination):

```yaml
policy:
  megatron_cfg:
    fp8_cfg:
      enabled: true
      fp8: e4m3
      fp8_recipe: mxfp8
      fp8_param: true
    optimizer:
      use_precision_aware_optimizer: false   # inherited from blockwise fp8 setup; =true untested
    distributed_data_parallel_config:
      overlap_param_gather: false            # WORKAROUND for the Megatron-LM bug in §3.3
  generation:
    vllm_cfg:
      precision: fp8
      is_mx: true
      quantization_ignored_layer_kws: []     # see note below
  nccl_reshard_refit: true                   # (+prefix if key absent from the config chain)
```

**`quantization_ignored_layer_kws: []` note:** the stock mxfp8-rollout recipe keeps
q/k/v/o in BF16 on the gen side for accuracy. With `fp8_param=true`, TE quantizes ALL
train-side linears, so gen must quantize them too or the misc transfer of attention
weights would land FP8 bytes in BF16 params. Preserving attn-BF16 under reshard would
require per-layer dequantization at export — listed as future work.

Consider shipping a recipe variant, e.g.
`grpo-qwen3-32b-8n4g-async-1off-mxfp8-rollout-reshard.yaml`, encoding the above
(validated launcher: `script/new_refit/qwen3_32b_mxfp8_rollout_reshard.sh` in the
working checkout).

---

## 4. Verification runbook

1. **4B canary** (fast, 4 GB200 nodes, ~15 min): run
   `script/new_refit/4b_mxfp8_to_mxfp8_dp_dp.sh` (Qwen3-4B, train DP8, gen DP8,
   `grpo.max_num_steps=10`). PASS = Step 10/10, `weight_sync` ≤ ~0.5 s,
   `Generation KL Error` in **0.005–0.008** every step, log line
   `reshard path: xferdtensor_python (exact-transfer)`, payload line
   `[xferd-payload] ... xfer_frac=59.4%`.
2. **Goal recipe** (8 GB200 nodes, ~35 min): run
   `script/new_refit/qwen3_32b_mxfp8_rollout_reshard.sh`. PASS = Step 10/10,
   KL **0.005–0.006**, xfer_frac 71.3%.
3. **Regression check**: the BF16 L0 matrix (`script/new_refit/{4b_*,30b_*,1b_dp_dp}.sh`)
   must stay green — all changes are gated on fp8_recipe / dtype and are no-ops for BF16.
   Expected: 4b KL 0.0006–0.0008, 30b KL 0.0016–0.0032, 1b = intended gate-reject.

Cluster-specific gotchas for re-running on this GB200 cluster (not needed for the PRs
themselves): `ray.sub` needs `unset SLURM_CPUS_PER_TASK` re-applied after any checkout
(SLURM 23.11 env conflict); compute-node GitHub egress is flaky — every launcher passes
a `SETUP_COMMAND` installing git `insteadOf` redirects to local mirrors of
`NVIDIA/Model-Optimizer` and `NVIDIA/Megatron-LM` under
`/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/youngeunk/mirrors/`
plus `git config --global protocol.file.allow always`; account `coreai_dlalgo_nemorl`;
job walltimes ≤ 1 h.

---

## 5. Known limitations / follow-up work (mention in PR descriptions)

1. **Megatron-LM training bug** (§3.3) — until fixed upstream, mxfp8 `fp8_param`
   training requires `overlap_param_gather=false`.
2. **Collective-path mxfp8→mxfp8**: the `fp8.py` passthrough removes the
   `mxfp8_quantize` blocker, but the full collective path with `fp8_param=true` has
   only been validated up to training start — finish validating if that path matters.
3. **Attention-BF16 preservation under reshard**: needs per-layer dequant at export
   (see §3.4). The stock rollout recipe's accuracy advantage of BF16 attention is
   given up by the current reshard variant.
4. **`use_precision_aware_optimizer=true`** untested with mxfp8 fp8_param.
5. **MXFP8 scale padding**: `_trim_blockwise_fp8_scale_inv_padding` no-ops for MXFP8;
   fine for tested shapes, may need work for models with unaligned dims (§3.2 note).
6. **Silent misc-skip hardening** (§3.1 suggestion).
7. **MTP + reshard** (orthogonal, discovered earlier on nemotron-super): mcore's
   `mtp_on_this_rank` treats `mtp_num_layers=0` as MTP-enabled (`is not None` check),
   and refit cannot handle the resulting embedding duplicated across first/last PP
   stages. Workaround: `mtp_num_layers=null`. Real MTP refit support needs a
   multi-owner rule in `broadcast_obj_from_pp_rank`.

## 6. Debugging playbook (if a PR reviewer hits a hang)

The failure mode of this subsystem is *silent distributed hangs*, and the July-2026
debugging campaign (28 iterations) produced these reusable lessons:

- A gen-side exception inside the EngineCore RPC used to surface **nowhere**: EngineCore
  stderr is not relayed to the Ray driver log, the async worker swallowed exceptions
  (fixed by this PR), and `sync_weights` blocks on train futures first. If you see
  train workers stuck in `_broadcast_misc_params_packed` with idle GPUs, suspect a
  dead gen consumer and grep the log for `Exception during nccl_reshard_refit` and
  `ERROR .* core.py`.
- Ray deduplicates identical log lines across workers (`[repeated Nx across cluster]`)
  — never count raw matching lines to infer per-rank behavior.
- On clusters that block both `srun --jobid` attach and ptrace/py-spy, the working
  stack-capture technique is **in-process**: a daemon thread armed at worker init that
  periodically `faulthandler.dump_traceback(file=sys.stdout)`.
