# End-to-End NVFP4 Training with Per-Token vLLM Rollout

## Overview

NeMo RL supports end-to-end W4A4 reinforcement learning for Qwen3 MoE models
on NVIDIA Blackwell GPUs. During policy training, Transformer Engine (TE)
applies NVFP4 to routed-expert MLP computation with per-token activation
scaling. During rollout, vLLM uses NVFP4 fused-MoE kernels with the same
per-token activation granularity.

The model keeps BF16 master parameters and FP32 optimizer states. Attention,
routers, shared experts, embeddings, normalization layers, and selected
boundary layers remain in BF16. The policy backward pass currently uses TE's
dequantized path.

After every policy update, NeMo RL quantizes the updated routed-expert weights,
packs them for transfer, and refits the colocated vLLM engines before the next
rollout. This keeps the training and rollout paths on the same NVFP4 weight and
activation-scaling contract.

## Performance and GPU Memory

The following comparison uses Qwen3-30B-A3B-Base with DAPO on 8 nodes with
4 GB200 GPUs per node, a global batch size of 512, and a 20K response limit.
Training uses TP=2, EP=8, and PP=1; each vLLM engine uses TP=1.
Each value is the median over the 502 logged steps shared by both runs between
steps 200 and 800. These are representative runs rather than a controlled
precision-only A/B; their configurations differ beyond numerical precision.

| Metric | BF16 | NVFP4 W4A4 | Change |
|---|---:|---:|---:|
| Rollout generation throughput | 457.9 tokens/s/GPU | 799.4 tokens/s/GPU | 1.75x |
| Generation time | 319.2 s | 144.7 s | 2.21x faster |
| Observed step time | 390.9 s | 224.8 s | 42.5% lower |
| Token-normalized end-to-end throughput | 376.1 tokens/s/GPU | 506.2 tokens/s/GPU | 1.35x |
| Weight transfer and update | 1.76 s | 17.93 s | 16.17 s higher |

The NVFP4 run generated shorter responses in this interval, so observed step
time alone overstates the speedup: its median response length was approximately
6,219 tokens, compared with 7,730 tokens for BF16. Generation throughput and
token-normalized end-to-end throughput control for this difference. Policy
training time is currently similar because the backward path remains
dequantized; the main gain comes from W4A4 rollout.

NVFP4 stores each quantized weight using an E2M1 value, one E4M3 scale per
16-value block, and one FP32 global scale per tensor. Excluding the small
per-tensor scale overhead, this is approximately 4.5 bits per quantized weight,
compared with 16 bits for BF16.

Both runs set `gpu_memory_utilization=0.5`. The vLLM logs report the following
rollout memory values for each TP=1 engine:

| Rollout memory | BF16 | NVFP4 W4A4 | Change |
|---|---:|---:|---:|
| Model weights | 56.88 GiB | 18.07 GiB | 68.2% lower |
| Available KV cache | 30.37 GiB | 79.70 GiB | 2.62x |
| KV cache token capacity | 331,744 | 870,560 | 2.62x |

These values describe the vLLM rollout model and KV cache, not the peak memory
of the complete colocated RL process. Process-wide GPU samples also include TE
training workspaces, quantization buffers, and allocator caching, and did not
show a lower end-to-end peak in this comparison. The current result therefore
demonstrates lower rollout weight memory and greater KV-cache headroom, rather
than lower peak memory for the full training process.

The current performance result includes quantization and weight refit after
each policy update. Refit is still a visible part of the step and remains an
optimization target. Training-quality curves will be added after the ongoing
long-run stability validation is complete.

## Quick Start

Start with the complete Qwen3-30B-A3B example:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-30ba3b-4n4g-megatron-te-nvfp4-pertoken-quick.yaml
```

The feature-specific policy configuration is:

```yaml
policy:
  generation:
    colocated:
      enabled: true
    nvfp4_pertoken_rollout:
      enabled: true
      additional_ignore:
        - "*.layers.0.mlp.experts*"
        - "*.layers.1.mlp.experts*"
        - "*.layers.44.mlp.experts*"
        - "*.layers.45.mlp.experts*"
        - "*.layers.46.mlp.experts*"
        - "*.layers.47.mlp.experts*"
    vllm_cfg:
      precision: bfloat16
      kv_cache_dtype: auto
      expert_parallel_size: 1

  megatron_cfg:
    moe_router_dtype: fp32
    fp4_cfg:
      enabled: true
      fp4: e2m1
      fp4_recipe: nvfp4
      fp4_param: false
    first_last_layers_bf16: true
    num_layers_at_start_in_bf16: 2
    num_layers_at_end_in_bf16: 4
    te_precision_config_file: examples/configs/te_precision/attn_bf16_mlp_nvfp4.yaml
    env_vars:
      NVTE_NVFP4_ROW_SCALED_ACTIVATION: "1"
      NVTE_NVFP4_DISABLE_RHT: "1"
      NVTE_NVFP4_DISABLE_2D_QUANTIZATION: "1"
      NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING: "1"
      NVTE_BACKWARD_OVERRIDE: dequantized
```

`additional_ignore` keeps complete routed-expert layers in BF16 during rollout.
It must cover the same boundary layers selected by
`first_last_layers_bf16`, `num_layers_at_start_in_bf16`, and
`num_layers_at_end_in_bf16`. Only complete expert-layer patterns in the form
`*.layers.<index>.mlp.experts*` are accepted.

## How It Works

```mermaid
flowchart LR
    A[TE per-token NVFP4 policy training] --> B[Megatron-Bridge HF weight export]
    B --> C[Quantize and pack routed experts]
    C --> D[Colocated weight refit]
    D --> E[vLLM per-token NVFP4 rollout]
    E --> A
```

### Per-token NVFP4 policy training

`policy.megatron_cfg.fp4_cfg` enables TE NVFP4, while
`te_precision_config_file` selects which modules use it. The provided precision
recipe applies NVFP4 to MLP linears and keeps attention linears in BF16.

`fp4_param=false` keeps persistent model parameters in BF16. NVFP4 is used for
the selected forward computations, while the optimizer continues to update the
BF16 model through FP32 master parameters. FP8 and FP4 cannot be enabled
together.

During policy training, per-token activation scaling gives each token its own
activation range. This reduces the effect of outlier tokens and matches the
activation granularity used by the rollout kernel.

### Weight refit

Megatron-Bridge exports the updated policy weights with Hugging Face parameter
names and reconstructs tensors across the training TP, PP, and EP topology.
NeMo RL then processes each routed-expert layer:

1. collect the gate, up, and down projection weights;
2. fuse gate and up projections into W13;
3. quantize weights in 16-value NVFP4 blocks;
4. emit packed weights, block scales, and global scales; and
5. fuse each quantized layer into six transport tensors.

Layer fusion keeps the refit stream compact: large MoE models transfer six
tensors per quantized expert layer instead of separate tensors for every expert
and projection. Ignored BF16 expert layers use two fused tensors per layer.

The current path quantizes after Megatron-Bridge reconstructs the expert
weights. Each vLLM engine receives the stream and its native loaders select the
TP and PP shards owned by the local rank.

### Per-token vLLM rollout

NeMo RL configures vLLM with the `nvfp4_pertoken` quantization mode. During
refit, vLLM loads the packed expert weights, prepares them for the FlashInfer
fused-MoE runtime, and rebuilds the affected kernels. Rollout activations are
quantized dynamically for each token; no static activation calibration is
required.

The reload path preserves the parameter storage used by CUDA graphs and
synchronizes the device before refit buffers are reused. A failed or incomplete
refit stops the rollout instead of continuing with stale weights.

## Supported Configuration

| Setting | Current support |
|---|---|
| Hardware | NVIDIA Blackwell; validated on GB200 |
| Model | `Qwen3MoeForCausalLM` with an MoE block in every decoder layer |
| Quantized modules | Routed-expert MLP projections |
| Policy parameters | BF16 with `fp4_param=false` |
| Rollout | Colocated vLLM, synchronous or asynchronous |
| vLLM tensor parallelism | Supported; TP=1 and TP=2 are validated |
| vLLM pipeline parallelism | Supported; PP=1 and PP=2 are validated; PP>1 requires the asynchronous engine |
| vLLM expert parallelism | EP=1 |
| KV cache | `auto` |
| Refit transport | Default colocated CUDA IPC/ZMQ path (`refit_transport: null`) |

Other generation quantization settings, including `generation.quant_cfg` and
`generation.real_quant`, must be unset when this mode is enabled. Incompatible
settings are reported during setup.

Dense MLP layers, hybrid dense/MoE layouts, and vLLM expert parallelism are not
yet supported by this path.

## Roadmap

- Reduce refit latency and data movement, including quantization before the
  expert-parallel gather.
- Add native NVFP4 backward computation.
- Complete longer stability and model-quality studies and publish the training
  curves.
