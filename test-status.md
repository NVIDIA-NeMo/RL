# NeMo-RL Nightly Test Status
**Branch**: `zhiyul/bump_up_mbridge`
**Container**: `nemo-rl:hemil-automodel-transformers-v5-9db945aa4`
**NRT_REBUILD_VENVS**: `true`
**Last updated**: 2026-03-28

## Summary

| | Count |
|---|---|
| PASS | 51 |
| FAIL (code bug — awaiting automodel team) | 2 |
| FAIL (metric — convergence regression) | 2 |
| FAIL (OOM) | 1 |
| **Total failing** | **5** |

---

## Failing Tests

### 1. grpo-nanov3-30BA3B-2n8g-fsdp2-lora
### 2. sft-nanov3-30BA3B-2n8g-fsdp2-lora
**Status**: FAIL (code bug — awaiting automodel team)
**Jobs**: 10453627, 10453673 (latest — reward=0 across all steps)
**Logs**:
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/nemo-rl-nightly/code_snapshots_v5_nightly/grpo-nanov3-30BA3B-2n8g-fsdp2-lora/tests/test_suites/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora/logs/exp_001/wandb/wandb/run-20260327_210227-mno5uth4/files/output.log`
- `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/nemo-rl-nightly/code_snapshots_v5_nightly/sft-nanov3-30BA3B-2n8g-fsdp2-lora/tests/test_suites/llm/sft-nanov3-30BA3B-2n8g-fsdp2-lora/logs/exp_001/wandb/wandb/run-20260327_210301-0c555yn0/files/output.log`

**Fix history** (branch [`zhiyul/fix_wip`](https://github.com/NVIDIA-NeMo/Automodel/tree/zhiyul/fix_wip) in Automodel repo, `parallelizer.py`):
1. `AttributeError: 'FSDPNemotronHForCausalLM' object has no attribute 'model'` (jobs 10360713, 10417454) → fixed with `object.__setattr__(result, "model", inner_model)`
2. `float != c10::BFloat16` at `lora.py:239` (job 10417454) → fixed with `layer_mp_policy` override (`output_dtype=param_dtype`) for per-layer FSDP2 shards
3. Reward=0 across all steps (jobs 10453627, 10453673) → root cause unknown, handed off to automodel team

**Next step**: Automodel team to investigate. Branch `zhiyul/fix_wip` and snapshot logs available.

---

### 3. dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel
**Status**: FAIL (metric — convergence regression)
**Job**: 10402105
**Log**: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/nemo-rl-nightly/code_snapshots_v5_nightly/dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel/tests/test_suites/llm/dpo-nanov3-30B3AB-1n8g-fsdp8ep8-automodel/logs/exp_001/wandb/wandb/run-20260326_225059-0k633c08/files/output.log`
**Metric**: `train/loss` step 15 = **0.6438** (threshold < 0.6326); same for `train/preference_loss`
**Root cause**: NanoV3 FSDP8+EP8 DPO not converging fast enough. Likely a learning dynamics regression from transformers-v5 changes.
**Next step**: Compare loss curves vs main branch.

---

### 4. vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-megatrontp2.v1
**Status**: FAIL (metric — loss regression)
**Log**: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/nemo-rl-nightly/code_snapshots_v5_nightly/vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-megatrontp2.v1/tests/test_suites/vlm/vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-megatrontp2.v1/logs/exp_001/wandb/wandb/run-20260325_233919-ba2s8gvg/files/output.log`
**Metric**: `train/loss` step 200 = **0.494** (threshold < 0.1); `train/reward` = 0.943 (PASS > 0.9)
**Root cause**: Reward converged but loss is high — unusual pattern. Possible loss logging regression in Megatron VLM path.
**Next step**: Investigate what `train/loss` tracks for VLM GRPO Megatron vs main branch.

---

### 5. grpo-qwen2.5-32b-32n8g-fsdp2tp8-actckpt.v3
**Status**: FAIL (OOM)
**Job**: 10360716
**Log**: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/nemo-rl-nightly/code_snapshots_v5_nightly/grpo-qwen2.5-32b-32n8g-fsdp2tp8-actckpt.v3/tests/test_suites/llm/grpo-qwen2.5-32b-32n8g-fsdp2tp8-actckpt.v3/logs/exp_001/wandb/wandb/run-20260326_015403-fqnca7p2/files/output.log`
**Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB. 177.38 MiB free, 26.44 GiB reserved but unallocated.`
**Root cause**: Memory fragmentation during activation checkpoint recomputation on 32-node run.
**Next step**: Add `PYTORCH_ALLOC_CONF=expandable_segments:True` to `EXTRA_ENV` and resubmit.

---

## Passing Tests (51)

<details>
<summary>Show all passing tests</summary>

- distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1
- distillation-qwen3-32b-to-1.7b-base-1n8g-megatron-tp2pp2cp2-pack
- dpo-llama3.1-8b-instruct-4n8g-megatrontp2pp2-quick
- dpo-llama3.2-1b-instruct-1n8g-fsdp2tp1.v2
- dpo-mistral-nemo-instruct-2407-1n8g-fsdp2tp8-actckpt-long
- grpo-deepscaler-1.5b-8K
- grpo-deepscaler-1.5b-16K
- grpo-deepscaler-1.5b-24K
- grpo-gemma3-1b-it-1n8g-fsdp2tp1
- grpo-gspo-deepscaler-1.5b-8K
- grpo-llama3.1-8b-instruct-1n8g-megatron-fp8-rollouts.v3
- grpo-llama3.1-8b-instruct-2n8g-fsdp2tp1-noncolocated
- grpo-llama3.1-8b-instruct-2n8g-megatron-fp8-e2e
- grpo-llama3.2-1b-instruct-1n8g-fsdp2tp1.v3
- grpo-llama3.2-1b-instruct-1n8g-fsdp2tp2-temp0.8-topp0.9-topk50
- grpo-llama3.2-1b-instruct-1n8g-megatron
- grpo-llama3.2-1b-instruct-1n8g-megatron_generation
- grpo-llama3.2-1b-instruct-1n8g-megatron-temp0.8-topp0.9-topk50
- grpo-math-qwen3-30ba3b-megatron-tp4-32k
- grpo-moonlight-16ba3b-4n8g-megatron
- grpo-moonlight-16ba3b-4n8g-megatron-fp8-e2e
- grpo-moonlight-16b-automodel-1n8g-ep8
- grpo-nano-v2-12b-1n8g-megatron
- grpo-nano-v2-12b-2n8g-fsdp2tp1
- grpo-nanov3-30BA3B-2n8g-fsdp2
- grpo-nanov3-30BA3B-2n8g-megatron-lora
- grpo-qwen2.5-7b-instruct-4n8g-fsdp2tp4.v3
- grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1-sglang
- grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1.v3
- grpo-qwen3-0.6b-1n8g-sglang
- grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron
- grpo-qwen3-8B-base-1n8g-fsdp2-lora
- grpo-qwen3-8b-base-1n8g-megatron-lora
- prorlv2-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1
- sft-gpt-oss-20b-1n8g-fsdp8ep8-automodel
- sft-llama3.1-8b-1n8g-fsdp2tp1-lora
- sft-llama3.1-8b-1n8g-fsdp2tp2
- sft-llama3.1-8b-1n8g-fsdp2tp4-dynamicbatch
- sft-llama3.1-8b-1n8g-megatron
- sft-llama3.1-8b-1n8g-megatron-lora
- sft-llama3.1-8b-1n8g-megatron-seqpack
- sft-llama3.2-1b-1n8g-fsdp2tp1.v3
- sft-nanov3-30BA3B-2n8g-fsdp2
- sft-qwen2.5-32b-4n8g-fsdp2tp8sp-actckpt.v3
- sft-qwen2.5-math7b-1n8g-megatron_chunked_linear_ce_loss
- sft-qwen2.5-math7b-2n8g-megatron
- vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-dtensor2tp1.v1

</details>
