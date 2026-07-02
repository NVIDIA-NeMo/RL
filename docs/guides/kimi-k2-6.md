# Kimi K2.6

This guide covers `moonshotai/Kimi-K2.6` with Megatron Core training and vLLM rollout in NeMo RL.

## Support Status

Supported:

- Megatron Core policy training through Megatron Bridge.
- vLLM rollout with `load_format: dummy`.
- Live full-model refit from Megatron policy workers to vLLM workers.
- BF16 SFT and GRPO smoke recipes on 16 nodes with 4 GPUs per node.
- Text-only vLLM refit for Kimi K2.6, including Kimi routed-expert packed
  tensor triplets exported by Megatron Bridge.

Not covered by these recipes:

- Native INT4 training from the published checkpoint quantization metadata.
- FP8 training or FP8 rollout validation.
- MoonViT or multimodal training.
- 256K context validation.

## Weight And Cache Setup

After installing NeMo RL, make sure every training node can read the same
Hugging Face and Megatron conversion caches.

```bash
export HF_HOME=/path/to/shared/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export NRL_MEGATRON_CHECKPOINT_DIR=/path/to/shared/nemo_rl_megatron
```

Authenticate to Hugging Face and pre-download `moonshotai/Kimi-K2.6` before
launching multi-node jobs. The first Megatron run converts the Hugging Face
checkpoint into the shared `NRL_MEGATRON_CHECKPOINT_DIR`; later runs reuse that
conversion unless `policy.megatron_cfg.force_reconvert_from_hf=true`.

The Kimi recipes set:

- `policy.hf_config_overrides.architectures: [KimiK25ForConditionalGeneration]`
- Megatron policy parallelism: TP8, PP8, EP8, ETP1
- vLLM rollout parallelism: TP8, PP1, EP64
- `policy.generation.vllm_cfg.load_format: dummy`
- `policy.generation.vllm_cfg.ipc_refit_metadata_in_payload: true`
- `policy.refit_buffer_memory_ratio: 0.3`

The dummy-load path keeps vLLM's Kimi compressed-tensors model layout intact.
During refit, NeMo RL handles the text model tensors directly: exact-shape
non-expert tensors, vocab-parallel tensors, MLA projection shards, fused
non-expert projections, and packed routed-expert tensors. Vision tower and
multimodal projector tensors are intentionally not part of these text-only
recipes.

For Kimi K2.6 vLLM rollout, keep
`policy.generation.vllm_cfg.max_model_len` divisible by 128. vLLM's
FlashInfer MLA backend rejects KV-cache block counts that are not aligned this
way. When overriding smoke lengths, round the total cap up; for example,
`704` input tokens plus `512` generated tokens should use `1280`, not `1216`.

## Generation Smoke

Use the standalone generation smoke before running GRPO. It loads the Kimi
recipe, starts colocated Megatron policy and vLLM rollout workers, refits live
Megatron weights into vLLM, generates a small prompt batch, and fails if the
decoded outputs are empty or placeholder-like.

```bash
# Run from an environment with both Megatron/MCore and vLLM dependencies installed.
python tools/model_diagnostics/kimi_k2_6_generation_smoke.py \
  --config examples/configs/recipes/llm/grpo-kimi-k2.6-16n4g-tp8pp8ep8-megatron.yaml \
  --preflight-only

python tools/model_diagnostics/kimi_k2_6_generation_smoke.py \
  --config examples/configs/recipes/llm/grpo-kimi-k2.6-16n4g-tp8pp8ep8-megatron.yaml \
  policy.generation.max_new_tokens=64 \
  policy.generation.vllm_kwargs.max_num_batched_tokens=512 \
  policy.generation.vllm_kwargs.max_num_seqs=2
```

## Recipes

- GRPO: `examples/configs/recipes/llm/grpo-kimi-k2.6-16n4g-tp8pp8ep8-megatron.yaml`
- SFT: `examples/configs/recipes/llm/sft-kimi-k2.6-16n4g-tp8pp8ep8-megatron.yaml`

The GRPO recipe is configured for a 10-step OpenMath smoke. For a one-step or
repeated-refit smoke, run the same recipe with `grpo.max_num_steps=1` or
`grpo.max_num_steps=3`.

## Validation

Before marking a Kimi PR ready, complete the validation ladder below.

1. Static checks pass, including recipe/test-suite consistency.
2. The generation-smoke preflight shows the expected model, runtime, cache,
   parallelism, `load_format: dummy`, and live refit settings.
3. The generation smoke reaches `KIMI_GENERATION_SMOKE_PASS` and prints readable
   decoded samples.
4. One-step GRPO sanity finishes `1/1` with Slurm `COMPLETED` and exit code `0:0`.
5. Repeated-refit GRPO smoke finishes every configured step, at least `3/3`,
   with no traceback, Ray task failure, actor death, OOM, or NCCL failure.
6. Hard OpenMath GRPO smoke finishes `10/10` with Slurm `COMPLETED` and exit code `0:0`.
7. SFT smoke finishes all configured steps and logs a finite loss.
8. Logprob/backend consistency is validated with HF/vLLM and Megatron/vLLM
   diagnostics, or the PR documents a precise follow-up validation plan.

Healthy logs should show setup complete, generation, reward processing, logprob
computation, policy training, training results, and train JSONL output.
Generated samples must be readable. Reward/loss quality does not need to prove
convergence, but the run must not be inert: tiny plumbing smoke needs nonzero
reward or reward variance, and OpenMath validation needs a real reward signal.
