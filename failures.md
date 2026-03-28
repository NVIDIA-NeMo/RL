# NeMo-RL Nightly — Failure Details
**Branch**: `hemil/automodel-transformers-v5`
**Container**: `nemo-rl:hemil-automodel-transformers-v5-9db945aa4`
**Last updated**: 2026-03-27

---

## 1. grpo-nanov3-30BA3B-2n8g-fsdp2-lora

**Status**: PENDING RESUBMIT (fixes applied 2026-03-27)
**Last failed job**: 10417454 (float != BFloat16 at lora.py:239)

**Error history**:
1. `AttributeError: 'FSDPNemotronHForCausalLM' object has no attribute 'model'` (jobs 10360713, 10417454 first run)
   - Fixed: `object.__setattr__(result, "model", inner_model)` in `parallelizer.py`
2. `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`
   - At `lora.py:239` (`F.linear(x, self.weight, bias)` in `LinearLoRA.forward` for `lm_head`)
   - Root cause: `NemotronHParallelizationStrategy` passed `mp_policy` (with `output_dtype=float32`) to per-layer `fully_shard_by_dtype` → backbone hidden_states became float32. ROOT `fully_shard` cast `lm_head.weight` to bfloat16 via `param_dtype` → mismatch.
   - Fixed: In `NemotronHParallelizationStrategy.parallelize`, create `layer_mp_policy` with `output_dtype=param_dtype` (bfloat16) for per-layer shards; keep ROOT `fully_shard` with original `mp_policy`.

**All fixes applied in `parallelizer.py`**:
1. `object.__setattr__(result, "model", inner_model)` — restores `self.model` after FSDP2 removes it from `_modules`
2. `layer_mp_policy` override — sets `output_dtype=param_dtype` for per-layer FSDP2 to keep hidden_states in bfloat16

**Next step**: Snapshot deleted. Resubmit from nightly runner.

---

## 2. sft-nanov3-30BA3B-2n8g-fsdp2-lora

**Status**: PENDING RESUBMIT (fixes applied 2026-03-27)
**Last failed job**: 10417458

**Root cause / fixes**: Same as `grpo-nanov3-30BA3B-2n8g-fsdp2-lora` above — both `object.__setattr__` fix and `layer_mp_policy` output_dtype override applied in `parallelizer.py`.

**Next step**: Snapshot deleted. Resubmit from nightly runner.

---

## 3. dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel

**Status**: FAIL (metric — convergence regression)
**Job**: 10402105 (resubmit of original failure)
**Snapshot log**: `code_snapshots_v5_nightly/dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel/tests/test_suites/llm/dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel/logs/exp_001/wandb/wandb/run-20260326_225059-0k633c08/files/output.log`

**Metric check output**:
```
PASS  train/loss at step 1        = 0.6931  < 0.69316
FAIL  train/loss at step 10       = 0.6438  < 0.63263
PASS  train/preference_loss step1 = 0.6931  > 0.69314
PASS  train/preference_loss step1 = 0.6931  < 0.69316
FAIL  train/preference_loss step10= 0.6438  < 0.63263
PASS  timing/train step (last 5)  = 1.473s  < 5s
```

**Loss progression (selected steps)**:
```
Step  1: loss=0.6931, preference_loss=0.6931, accuracy=0.0000
Step  2: loss=0.6867, preference_loss=0.6867, accuracy=0.4375
Step 10: loss=0.6438, preference_loss=0.6438  ← FAIL (threshold < 0.6326)
```

**Root cause**: Convergence regression with NanoV3 FSDP8+EP8 automodel DPO. Loss is not decreasing fast enough — 0.6438 vs threshold <0.6326 at step 10. Confirmed on retry (not a fluke). Likely caused by transformers-v5 changes affecting NanoV3 DPO learning dynamics.

**Next step**: Compare loss curves vs main branch; check if optimizer state, learning rate, or weight initialization is affected by automodel/transformers-v5 changes.

---

## 4. vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-megatrontp2.v1

**Status**: FAIL (metric — loss regression)
**Job**: (recent run)
**Snapshot log**: `code_snapshots_v5_nightly/vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-megatrontp2.v1/tests/test_suites/vlm/vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-megatrontp2.v1/logs/exp_001/wandb/wandb/run-20260325_233919-ba2s8gvg/files/output.log`

**Metric check output**:
```
PASS  train/reward at step 200 = 0.943   > 0.9
FAIL  train/loss   at step 200 = 0.494   < 0.1
```

**Throughput at step 200**:
```
E2E (Tokens/sec/gpu):               82.11
Policy Training (Tokens/sec/gpu): 1388.99
Policy+Ref Logprobs (Tok/sec/gpu): 1670.34
Generation (Tok/sec/gpu):          3995.85
Mean total tokens per sample:        35.12
```

**Root cause**: Reward converged (0.943 > 0.9, PASS) but loss is high (0.494 vs threshold <0.1). High reward + high loss is unusual — loss metric may be measuring something different from the reward (e.g., language modeling loss vs GRPO objective). Possible logging regression or genuine convergence issue in the Megatron VLM path.

**Next step**: Investigate what `train/loss` tracks for VLM GRPO Megatron — check if loss logging definition changed in this branch vs main.

---

## 5. grpo-qwen2.5-32b-32n8g-fsdp2tp8-actckpt.v3

**Status**: FAIL (OOM)
**Job**: 10360716
**Snapshot log**: `code_snapshots_v5_nightly/grpo-qwen2.5-32b-32n8g-fsdp2tp8-actckpt.v3/tests/test_suites/llm/grpo-qwen2.5-32b-32n8g-fsdp2tp8-actckpt.v3/logs/exp_001/wandb/wandb/run-20260326_015403-fqnca7p2/files/output.log`

**Error**:
```
ray.exceptions.RayTaskError(OutOfMemoryError):
  ray::DTensorPolicyWorkerV2.train()
  (pid=3900548, ip=10.65.29.143, actor_id=7c33dde8bfc662c0efbe335501000000)

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB.
GPU 0 has a total capacity of 79.11 GiB of which 177.38 MiB is free.
Process 3894014 has 5.37 GiB memory in use.
Including non-PyTorch memory, this process has 73.50 GiB memory in use.
Of the allocated memory 43.07 GiB is allocated by PyTorch,
and 26.44 GiB is reserved by PyTorch but unallocated.
If reserved but unallocated memory is large try setting
PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.

Traceback (most recent call last):
  File ".../nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py", line 430, in train
    mb_results = automodel_forward_backward(...)
  File ".../nemo_rl/models/automodel/train.py", line 432, in automodel_forward_backward
    result, metrics, _ = forward_with_post_processing_fn(...)
  File ".../transformers/models/qwen2/modeling_qwen2.py", line 312, in forward
    hidden_states = self.post_attention_layernorm(hidden_states)
  File ".../torch/utils/checkpoint.py", line 512, in checkpoint
    ret = function(*args, **kwargs)
  File ".../transformers/models/qwen2/modeling_qwen2.py", line 264, in forward
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
               ^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB.
```

**Memory snapshot at failure**:
| | |
|---|---|
| GPU total capacity | 79.11 GiB |
| Free | 177.38 MiB |
| PyTorch allocated | 43.07 GiB |
| PyTorch reserved (unallocated) | 26.44 GiB |
| Non-PyTorch process memory | 30.43 GiB |
| Attempted allocation | 320.00 MiB |

**Config**:
```
model:           Qwen/Qwen2.5-32B
nodes:           32 (256 GPUs total)
TP:              8 (training), 4 (generation/vLLM)
activation_ckpt: true
max_seq_len:     16384
train_micro_bs:  1
max_new_tokens:  16384
```

**Root cause**: OOM on step 2 forward pass during activation checkpoint recomputation (`post_attention_layernorm`). PyTorch has 26.44 GiB reserved but fragmented — cannot satisfy a 320 MiB contiguous allocation. Generation uses TP=4 while training uses TP=8; memory from generation phase may not be fully released before training, contributing to fragmentation.

**Next step**: Add `PYTORCH_ALLOC_CONF=expandable_segments:True` to `EXTRA_ENV` and resubmit.
