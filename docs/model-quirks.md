# Model Quirks

This document outlines special cases and model-specific behaviors that require custom handling in NeMo RL. These special cases are controlled by the `ModelFlag` enum.

## Gemma-3

### vLLM Initialization

Gemma-3 models have a specific issue with vLLM dummy weight initialization due to a vLLM bug where [a `normalizer` buffer is created](https://github.com/vllm-project/vllm/blob/964472b9667508b1d4a7ed92068ff81740ae0036/vllm/model_executor/models/gemma3.py#L372) that is not present in the Hugging Face model. This causes the `normalizer` buffer to be set to dummy weights at initialization and then never updated with the correct values during model refit. As a workaround for this issue, we do not use dummy weight initialization for vLLM with Gemma-3 models and instead use the `load_format="auto"` setting to load the full weights at initialization.

**Special Handling:**
- We automatically use `load_format="auto"` for Gemma-3 models when initializing vLLM.
- This avoids issues with dummy weight initialization, where the dummy weights for this buffer would never get overwritten during refit.

### vLLM V1 runtime

NeMo-RL uses the vLLM V1 runtime for both synchronous and asynchronous inference. The V1 runtime provides improved performance and stability for inference.

**Special Handling:**
- Both sync and async inference modes use the V1 runtime by default.
- Users can override to the V0 runtime by setting the environment variable `NRL_VLLM_USE_V1=0`.
- **Important**: The async implementation always uses the V1 runtime. Users who need to use the V0 runtime must switch to synchronous inference by setting `policy.generation.vllm_cfg.async_engine=False`.

### Context Parallel with FSDP2

- NeMo-RL implemented this feature based on torch CP [implementation](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/_attention.py). And we inherit its limitations.
Whether model level support CP only depends on arguments passed to `torch.nn.functional.scaled_dot_product_attention`. Current NeMo-RL passed all ones attention mask to `model.forward`. For Gemma-3, it won't ignore attention mask as result `attn_bias` is not None which is not supported by torch CP. Please see [assertion](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/tensor/experimental/_attention.py#L262) .

- Context parallel can't be used together with sequence packing. Sequence packing requires `attn_implementation="flash_attention_2"`, this conflict with context parallel requires SDPA impl. Refer to [here](https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/modeling_utils.py#L2317) for more details.

- It's a known issue that context parallel can't be used together with sequence parallel.
Refer to [here](https://github.com/NVIDIA-NeMo/RL/issues/659) for more details.

## DeepScaleR Recipe Convergence Issues

The DeepScaleR recipe (e.g., `examples/configs/grpo-deepscaler-1.5b-8K.yaml`) has been found to experience convergence issues when CUDA graphs are enabled in vLLM.

**Special Handling:**
- CUDA graphs must be disabled by setting `enforce_eager: True` in the vLLM configuration (https://github.com/NVIDIA-NeMo/RL/pull/857 forces eager execution by default).

## vLLM Async Rollout Timeout

vLLM async generation has a configurable timeout for waiting for individual sample results. This is particularly important for longer sequences on large models.

```bash
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800  # Default: 600 (10 minutes)
```

If you encounter timeout errors, the system will suggest doubling the current timeout value.

## AutoModel Parameter Freezing (`freeze_config`)

VLM/omni recipes on the AutoModel (DTensor) backend control which sub-towers
train via a declarative `freeze_config` under
`policy.model_kwargs.automodel_kwargs`:

```yaml
automodel_kwargs:
  freeze_config:
    freeze_vision_tower: true
    freeze_audio_tower: true
    freeze_language_model: false
```

AutoModel applies this at build time (`apply_parameter_freezing`, before the
optimizer is created). Note the following sharp edges:

- **No default auto-freeze.** If `freeze_config` is omitted, *nothing* is
  frozen. Earlier versions unconditionally froze the visual encoder for
  text-only training; that implicit safety net is gone. A text-only config run
  on a vision/audio-capable checkpoint without `freeze_config` will create
  optimizer state for parameters that never receive gradients, which leads to
  a **checkpoint-resume key mismatch**.
- **Typos fail silently.** A misspelled `freeze_*` key is ignored and falls
  back to the default (unfrozen) — no error is raised. Verify the keys exactly
  match `freeze_vision_tower` / `freeze_audio_tower` / `freeze_language_model`.

Every shipped recipe sets `freeze_config`, so in-repo recipes are unaffected.
This matters only for custom configs.
