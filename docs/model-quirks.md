# Model Quirks

This document outlines special cases and model-specific behaviors that require custom handling in NeMo RL. These special cases are controlled by the `ModelFlag` enum.

## Gemma-3

### Tied Weights

Weight tying between the embedding layer (`model.embed_tokens`) and output layer (`lm_head`) is currently not respected when using the FSDP1 policy or the DTensor policy when TP > 1 (See [this issue](https://github.com/NVIDIA/NeMo-RL/issues/227)). To avoid errors when training these models, we only allow training models with tied weights using the DTensor policy with TP=1. For Llama-3 and Qwen2.5 models, weight-tying is only enabled for the smaller models (< 2B), which can typically be trained without tensor parallelism. For Gemma-3, all model sizes have weight-tying enabled, including the larger models which require tensor parallelism. To support training of these models, we specially handle the Gemma-3 models by allowing training using the DTensor policy with TP > 1.

**Special Handling:**
- We skip the tied weights check for all Gemma-3 models when using the DTensor policy, allowing training using TP > 1.
- We exclude `model.embed_tokens` and `lm_head` from the DTensor tensor parallel plan to maintain weight tying correctly.

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

NeMo-RL implemented this feature based on torch CP [implementation](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/_attention.py). And we inherit its limitations.
Whether model level support CP only depends on arguments passed to `torch.nn.functional.scaled_dot_product_attention`. Current NeMo-RL passed all ones attention mask to `model.forward`. For Gemma-3, it won't ignore attention mask as result `attn_bias` is not None which is not supported by torch CP. Please see [assertion](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributed/tensor/experimental/_attention.py#L262) .

## vLLM Async Rollout Timeout

vLLM async generation has a configurable timeout for waiting for individual sample results. This is particularly important for longer sequences on large models.

```bash
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800  # Default: 600 (10 minutes)
```

If you encounter timeout errors, the system will suggest doubling the current timeout value.
