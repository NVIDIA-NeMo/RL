# Diffusion GRPO: RL for Image Diffusion Models

This guide covers the diffusion-GRPO implementation in NeMo RL — an adaptation of
[Flow-GRPO](https://arxiv.org/abs/2505.05470) that post-trains **image diffusion
(flow-matching, text-to-image) models** such as
[Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) with online RL.

> **Scope note**: this is *image* diffusion (continuous flow-matching over
> latents), not diffusion *language* models (dLLMs). The `nemo_rl/models/diffusion/`
> package and `diffusion_grpo` algorithm refer exclusively to text-to-image
> generation.

For foundational GRPO concepts (group-relative advantages, clipped policy
gradients), see the [GRPO guide](grpo.md). Diffusion GRPO mirrors
`nemo_rl.algorithms.grpo.grpo_train` in phase ordering but replaces token-level
concepts (vLLM rollouts, token log-probs, token KL) with their continuous
counterparts.

## How It Works

On-policy RL needs `log pi(action | state)`. A deterministic ODE sampler gives
no such density, so following Flow-GRPO the rollout converts the flow-matching
ODE into an equivalent **SDE**: each denoising transition becomes a Gaussian
step whose log-probability is computable in closed form
(`nemo_rl/models/diffusion/sde.py`). The training loop then applies the
standard GRPO recipe on top:

1. **Rollout**: sample `num_generations_per_prompt` images per prompt with the
   SDE sampler, recording per-step transition log-probs. An optional
   **SDE window** (`policy.algo.sde_window_size` / `sde_window_range`) makes
   only a slice of the denoising steps stochastic — the Flow-GRPO "Fast" mode —
   so the loss only needs to recompute those steps.
2. **Reward**: score decoded images with a pluggable reward environment
   (`nemo_rl/environments/image_reward_environment.py`).
3. **Advantage**: group-relative advantage with leave-one-out baseline
   (`grpo.use_leave_one_out_baseline`). `grpo.use_global_std: true` normalizes
   by the whole-batch reward std (verl-omni `global_std` semantics) because
   per-group std explodes on near-constant groups under sparse rewards.
4. **Train**: recompute transition log-probs with grad and apply the clipped
   policy-gradient loss (`nemo_rl/algorithms/loss/diffusion_grpo.py`), with
   optional Gaussian KL against the reference policy (`loss_fn.beta > 0`;
   the reference is the base model with the LoRA adapter disabled).

## Components

| Component | Path |
|---|---|
| SDE step / log-prob kernel | `nemo_rl/models/diffusion/sde.py` |
| Config schemas and protocols | `nemo_rl/models/diffusion/interfaces.py` |
| Clipped policy-gradient loss | `nemo_rl/algorithms/loss/diffusion_grpo.py` |
| Qwen-Image pipeline adapter (Diffusers) | `nemo_rl/models/diffusion/pipeline.py` |
| Ray worker (LoRA training, DP all-reduce) | `nemo_rl/models/diffusion/workers/diffusion_worker.py` |
| Controller-side policy facade | `nemo_rl/models/diffusion/policy.py` |
| Training loop | `nemo_rl/algorithms/diffusion_grpo.py` |
| Prompt dataset | `nemo_rl/data/datasets/text_to_image_prompt.py` |
| Reward environment | `nemo_rl/environments/image_reward_environment.py` |
| Entry point | `examples/run_diffusion_grpo.py` |

The worker runs data-parallel: with `cluster.gpus_per_node: N`, rollout prompts
scatter across N single-GPU workers and gradients all-reduce so every rank
applies the identical update. `policy.seed` is required for DP so all ranks
materialize bit-identical LoRA init; the training loop logs
`train/dp_checksum_spread` (must stay 0) to guard gradient sync.

## Quickstart: Qwen-Image on the OCR Task

Install the diffusion extra (Diffusers, PaddleOCR) together with the
NeMo Automodel extra, which provides the worker's model-lifecycle stack
(pipeline loading, LoRA, checkpointing):

```bash
uv sync --extra automodel --extra diffusion
```

Export the Flow-GRPO OCR prompt dataset (19,653 train / 1,018 val prompts; the
quoted text in each prompt is the OCR ground truth stored in metadata):

```bash
uv run python tools/export_ocr_prompts.py --out-dir examples/data/diffusion/ocr
```

Launch training (single node, 8 GPUs, LoRA):

```bash
uv run --frozen --extra diffusion python examples/run_diffusion_grpo.py \
    --config examples/configs/diffusion_grpo_qwen_image_ocr.yaml
```

The exemplar config mirrors the verl-omni
`run_qwen_image_ocr_lora.sh` hyperparameters. The nightly recipe
`examples/configs/recipes/diffusion/grpo-qwen-image-ocr-1n8g-dp8-lora.yaml`
runs a 60-step version with a convergence gate.

## Configuration

All defaults live on the pydantic `BaseModel` schemas
(`nemo_rl/models/diffusion/interfaces.py`,
`nemo_rl/algorithms/diffusion_grpo.py`) and are documented in the exemplar
YAML `examples/configs/diffusion_grpo_qwen_image_ocr.yaml`. Key blocks:

```yaml
grpo:
  num_prompts_per_step: 32        # x num_generations_per_prompt = samples/step
  num_generations_per_prompt: 16  # GRPO group size
  use_global_std: true            # whole-batch reward std normalization

loss_fn:
  ratio_clip_min: 1.0e-4          # tight window-mode clip (Flow-GRPO Fast mode)
  ratio_clip_max: 1.0e-4
  beta: 0.0                       # >0 adds Gaussian KL vs reference (requires LoRA)

policy:
  model_name: "Qwen/Qwen-Image"
  pipeline:
    num_inference_steps: 10       # training rollout steps (val uses grpo.val_generation)
  algo:
    noise_level: 1.2              # SDE noise scale
    sde_window_size: 2            # stochastic steps per rollout (Fast mode)
    sde_window_range: [0, 5]
  lora_cfg:
    enabled: true
```

Validation always samples with the deterministic ODE
(`grpo.val_generation.num_inference_steps`), keeps a fixed seed so
`val/reward_mean` is comparable across steps, and saves up to
`logger.num_val_samples_to_print` images per validation.

## Reward Plugins

`env.image_reward.plugins` is a weighted list; scores combine linearly.
Built-in plugins:

| Name | Reward |
|---|---|
| `dummy` | constant 0 (pipeline sanity checks) |
| `jpeg_compressibility` | negative JPEG size (classic DDPO sanity task) |
| `pickscore` | [PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) human-preference model |
| `ocr` | 1 − normalized Levenshtein distance between PaddleOCR output and the prompt's quoted target text |
| `genrm_ocr` | same OCR distance, but transcribed by a generative reward model behind an OpenAI-compatible endpoint (`GENRM_BASE_URL` env var; `model`/`temperature`/`top_p`/`max_tokens` plugin keys) |

Reward workers are Ray actors (`num_workers_per_plugin` replicas, CPU by
default); custom rewards register via
`nemo_rl.environments.image_reward_environment.register_image_reward`.

## Scope and Limitations

- **Model support**: Qwen-Image via the Diffusers pipeline adapter. Other
  flow-matching pipelines can implement the `DiffusionPipelineAdapter`
  protocol (`nemo_rl/models/diffusion/interfaces.py`).
- **Training path**: LoRA on single-GPU workers with data-parallel all-reduce
  (single- or multi-node via Ray). No FSDP/Megatron sharding; full-parameter
  training works only when the transformer fits on one GPU, and `loss_fn.beta > 0`
  requires LoRA (the reference policy is the adapter-disabled base model).
- **Rollout**: the training framework itself generates images (no separate
  inference engine, no refit step).
