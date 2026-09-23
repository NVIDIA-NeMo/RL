# GLM-5.3-Flash

This guide describes text-only GRPO training of GLM-5.3-Flash with the AutoModel
training backend and vLLM generation. It uses the local experiment configuration
`[exp/grpo-glm-flash.yaml](../../../../exp/grpo-glm-flash.yaml)`.

> [!IMPORTANT]
> **Status: Short-run training results available.** The supplied training curves
> include a reference run of approximately 29 steps and a full QDQ run of
> approximately 74 steps, with QDQ validation through step 70.
> They do not establish long-run convergence or a controlled QDQ comparison. This guide documents the experiment
> configuration rather than a registered nightly recipe.

## Support Status


| Model         | Training backend | Training parallelism | Generation backend   | Scope                     |
| ------------- | ---------------- | -------------------- | -------------------- | ------------------------- |
| GLM-5.3-Flash | AutoModel        | TP1 + CP1 + EP72     | vLLM with TP16 + EP1 | Text-only, colocated GRPO |




## Reference Scope

- **Model**: A local GLM-5.3-Flash checkpoint at `/apps/models/GLM-5.3-Flash`.
- **Algorithm**: GRPO with `DAPOMath17K` for training and
`DAPOMathAIME2024` for validation.
- **Training backend**: AutoModel with BF16 training, activation checkpointing,
SDPA attention, Transformer Engine linear layers, `torch_mm` experts, and the
HybridEP dispatcher.
- **Training parallelism**: TP1, CP1, and EP72.
- **Generation backend**: vLLM with FP8 precision, TP16, EP1, and eager execution.
The KV cache dtype inherits `auto` from the base configuration.
- **Sequence length**: Up to 2,048 prompt tokens and 2,048 generated tokens,
with a maximum total sequence length of 4,096.
- **Reference allocation**: 18 nodes with 8 GPUs per node, totaling 144 GPUs.
- **Deployment**: Colocated training and generation on the same GPUs.
- **Modality**: Language-only generation; the vision and audio towers are frozen
during training.
- **MTP**: Disabled with `policy.hf_config_overrides.text_config.num_mtp_modules: 0`.

The experiment YAML and its inherited
`[grpo_math_1B.yaml](../../../../examples/configs/grpo_math_1B.yaml)` are the
source of truth for these settings. The experiment file uses
`defaults: ../examples/configs/grpo_math_1B.yaml`; keep that relative path valid
if copying the configuration.

## How to Run



### 1. Prepare the Environment

Use an environment with GLM-5.3-Flash support in both AutoModel and vLLM,
Transformer Engine, and a HybridEP build that supports multi-node dispatch.
The local experiment launcher uses a custom container with HybridEP installed
under `/opt/hybridep`. A generic environment installation alone does not
establish compatibility with this experiment.

See the [installation guide](../../../about/installation.md),
[Dependency Management](../../../design-docs/dependency-management.md), and
[Cluster Setup](../../../cluster.md) for environment and multi-node Ray setup.
Make the model directory available at the same path on every node, and use a
shared Hugging Face cache for datasets:

```bash
export HF_HOME=<path-to-shared-huggingface-cache>
export WANDB_API_KEY=<your-wandb-api-key>
```

The recipe enables W&B logging. Pass `logger.wandb_enabled=false` if W&B is
not configured.

### 2. Choose the Reference Configuration


| Model         | Algorithm | Backend   | Scale | Configuration                                                |
| ------------- | --------- | --------- | ----- | ------------------------------------------------------------ |
| GLM-5.3-Flash | GRPO      | AutoModel | 18n8g | `[grpo-glm-flash.yaml](../../../../exp/grpo-glm-flash.yaml)` |


This configuration lives under `exp/`, outside the recurring recipe and nightly
test directories. Ensure the experiment file is present in your checkout.

### 3. Launch

From the repository root on the head node of the configured 18-node Ray cluster,
run:

```bash
uv run examples/run_grpo.py --config exp/grpo-glm-flash.yaml
```

To use a different shared model location, override `policy.model_name`:

```bash
uv run examples/run_grpo.py --config exp/grpo-glm-flash.yaml \
  policy.model_name=/path/to/GLM-5.3-Flash
```

The tokenizer name inherits `policy.model_name` from the base configuration.
Review EP72 and TP16 placement before changing the allocation; changing
`cluster.num_nodes` alone is insufficient. See the [GRPO guide](../../grpo.md)
for common algorithm and configuration details.

## Important Recipe Settings

- **Batch size**: 36 prompts per step with 16 generations per prompt produce
576 samples, matching `policy.train_global_batch_size: 576`. Training and
log-probability microbatches are both 1. Sequence packing and dynamic batching
are disabled.
- **Optimizer**: Transformer Engine `FusedAdam` uses a learning rate of `1e-6`,
weight decay of `0.1`, master weights, parameter remainders, and BF16 first
and second moments. The scheduler warms up over 10 steps, then remains constant.
- **GRPO loss**: Reward normalization is enabled, the leave-one-out baseline is
disabled, and reference-policy KL penalty is zero. Token-level loss uses
asymmetric ratio clipping (`0.2` / `0.28`) and truncated importance sampling
correction with an upper ratio of `2.0`.
- **Model backend**: `rms_norm: torch_fp32`, `rope_fusion: false`, and
`enable_hf_state_dict_adapter: true` are part of the experiment settings.
HybridEP shares the token dispatcher and enables FSDP optimizations.
- **Generation**: `language_model_only: true`, `enforce_eager: true`, and
`enable_flashinfer_autotune: false` are explicit overrides. GPU memory
utilization is `0.6` and `max_model_len` is 4,096.
- **Refit**: Keep the GLM-specific weight-cache invalidation in the vLLM backend.
After weight updates, the KDA merged convolution weights and DSA indexer's
cached FP32 projection weights must be rebuilt from the updated parameters.
- **Validation**: Runs every 10 steps with at most 240 validation prompts and
a validation batch size of 240; validation at startup is disabled.
- **Checkpointing**: Saves every 5 steps, retains one checkpoint with
`metric_name: null`, and includes optimizer state. The output directory is
`results/grpo-glm-flash-fix-dsa-cache`. Change it for independent experiments.



## Full FP8 QDQ

Quantize-dequantize (QDQ) exposes the BF16 training forward pass to FP8
rounding at selected operands, to more closely model FP8 generation. The
[full QDQ configuration](../../../../exp/grpo-glm-flash-full-qdq.yaml)
combines DSA Indexer QDQ with weight and activation QDQ in dense FFNs,
routed experts, and shared experts. Here, **full** means these seven options;
it does not mean that every model operation or the main MLA KV cache is
quantized.

All seven flags default to `false`. Enable them together with:

```yaml
policy:
  hf_config_overrides:
    text_config:
      indexer_fp8_fake_quant: true
      dense_fp8_weight_qdq: true
      dense_fp8_activation_qdq: true
      routed_fp8_weight_qdq: true
      routed_fp8_activation_qdq: true
      shared_fp8_weight_qdq: true
      shared_fp8_activation_qdq: true
```

### DSA Indexer QDQ

With the updated AutoModel GLM implementation, enable the following model
override to simulate vLLM's Indexer activation quantization during policy
forward passes:

```yaml
policy:
  hf_config_overrides:
    text_config:
      indexer_fp8_fake_quant: true
```

The flag defaults to `false`. It applies normalized Hadamard-128 rotation,
BF16 materialization, and per-vector E4M3FN quantize/dequantize with UE8M0
power-of-two scales to Indexer queries and completed pooled keys. Pooling and
head-weight scoring use FP32 accumulation. Incomplete tail tokens retain the
existing selection behavior. The supported layout has `index_head_dim: 128`
and `qk_rope_head_dim: 0`.

This option does not quantize projection weights, the main MLA computation,
KDA, dense FFNs, routed experts, or shared experts. Indexer top-k selection
already runs without gradients, so it does not introduce a straight-through
gradient estimator. Model parameters and checkpoint keys are unchanged.

### Dense FFN and MoE QDQ

The FFN paths use E4M3FN fake quantization without Hadamard rotation or
power-of-two scales:

| Operand | Quantization granularity | Scale calculation |
| --- | --- | --- |
| Gate, up, and down projection weights | Independent 128 × 128 blocks; experts are independent | Multiply by `448 / amax`, round to E4M3FN, then dequantize with the reciprocal multiplier; zero blocks use multiplier 1 |
| Inputs to gate/up projections and the activated input to the down projection | Groups of 128 channels per token | Divide by `max(amax, 1e-10) / 448`, round to E4M3FN, then multiply by that scale |

Quantization statistics and scaling are computed in FP32. Values are clamped
to the E4M3FN range before conversion and dequantized back to the operand's
original dtype. Weight matrix dimensions must be divisible by 128. Weights
are quantized afresh on each forward; parameter storage and checkpoint keys
remain unchanged.

An identity straight-through estimator (STE) passes the backward gradient
through the FFN quantization boundary to the original operand. Dense and
shared FFNs use ordinary PyTorch linears on the QDQ path, even when the base
backend selects Transformer Engine. Routed experts apply QDQ before both
expert GEMMs; the full configuration uses `torch_mm` with HybridEP. This
simulates operand rounding, not the accumulation behavior of actual FP8 GEMMs.

### Generation Settings and Launch

The standalone full QDQ YAML retains BF16 training, 18 nodes × 8 GPUs,
EP72 for training, TP16 for generation, and a global batch of 576. It does
not require YAML defaults inheritance. Its generation settings include:

```yaml
policy:
  generation:
    vllm_cfg:
      precision: fp8
      kv_cache_dtype: auto
      pow2_weight_scaling_factors: false
      pow2_activation_scaling_factors: false
    vllm_kwargs:
      kernel_config:
        enable_flashinfer_autotune: false
        linear_backend: cutlass
        moe_backend: triton
```

The FFN simulation uses ordinary FP32 scales to match this kernel selection;
the DSA Indexer uses its separate power-of-two scaling scheme. Main MLA KV
remains BF16 in this experiment. No additional QDQ is applied to the main
MLA computation, KDA, or Indexer projection weights.

From the head node of the prepared Ray cluster, run:

```bash
uv run --no-sync examples/run_grpo.py --config exp/grpo-glm-flash-full-qdq.yaml
```

Checkpoints go to `results/grpo-glm-flash-full-qdq`; the W&B project is
`nemorl-glm-flash-full-qdq` and the run name is `grpo-glm-flash-full-qdq`.
The YAML allows 10,000 training steps, but the supplied results below cover
only the initial short run. QDQ does not guarantee bitwise agreement with
vLLM's fused kernels or eliminate all training/generation differences.

## Training Experiments

### Reference Run Without QDQ

The supplied screenshot shows training reward, validation accuracy, mean
generated tokens per sample, generation KL error, approximate entropy, and
gradient norm for the reference experiment.

![GLM-5.3-Flash GRPO training reward, validation accuracy, response length, generation KL error, entropy, and gradient norm over approximately 29 steps](../../../assets/glm/glm-5.3-flash-grpo.png)

Validation accuracy reaches **0.32083 at step 20**. At the highlighted training
step 16, reward is **0.45313**, mean generated length is **1,597.9 tokens**, and
generation KL error is **0.010896**. Generation KL error rises later in the
displayed run, so these curves do not demonstrate resolved train/generation
parity or long-run convergence.

### Full QDQ Run

The following supplied screenshot is labeled `grpo-glm-flash-full-qdq` and
shows approximately 74 training steps, with validation every 10 steps through
step 70. Values below are approximate readings from the image rather than
an export of the underlying metrics.

![GLM-5.3-Flash full QDQ GRPO training reward, validation accuracy, response length, generation KL error, gradient norm, and entropy over approximately 74 steps](../../../assets/glm/glm-5.3-flash-qdq-grpo.png)

| Metric | Observation in the displayed full QDQ run |
| --- | --- |
| Validation accuracy | Rises from about 0.283 at step 10 to 0.467 at step 70 |
| Training reward | Noisy upward trend, ending around 0.65 |
| Mean generated length | Falls from roughly 1,500–1,600 tokens early in training to about 1,050 at the end |
| Generation KL error | Rises from about 0.012 to 0.018, with a late peak around 0.021 |
| Gradient norm | Mostly around 0.1–0.25, with a spike near 0.9 around step 64 |
| Approximate entropy | Fluctuates around 0.4–0.5 and ends near 0.42 |

The full QDQ run shows improving validation accuracy over this interval,
while generation KL error still increases. The reference and QDQ screenshots
cover different training horizons and do not establish a controlled A/B
comparison. These curves alone cannot attribute the accuracy trend to QDQ,
prove train/generation parity, or establish long-run convergence.

## Known Limitations

- This experiment covers text-only training with frozen vision and audio
towers. Multimodal training and generation are outside the documented scope.
- Training uses BF16 while generation requests FP8. The observed generation
KL error requires further evaluation; the screenshot alone does not isolate
its cause.
- MTP, sequence packing, dynamic batching, and non-colocated deployment are
not exercised by this configuration.
- Only the supplied short-run curves are reported here.
