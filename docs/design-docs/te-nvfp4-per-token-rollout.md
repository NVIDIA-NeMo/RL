# End-to-End W4A4 NVFP4 with Per-Token vLLM Rollout

## Overview

NeMo RL provides an end-to-end W4A4 NVFP4 path in which selected routed-expert
MLP computations use per-token activation scaling during both policy training
and vLLM rollout on NVIDIA Blackwell GPUs. Transformer Engine (TE) performs the
training computation, and vLLM uses NVFP4 fused-MoE rollout kernels with the
same activation-scaling granularity.

The feature is designed around a reusable training-to-rollout contract: policy
workers retain BF16 source weights, refit transports BF16, and rollout workers
own quantization into their serving representation. The current implementation
is intentionally guarded to the Qwen3 all-MoE layout that has been validated
end to end. See [Validated Configuration](#validated-configuration) and
[Current Limitations](#current-limitations) for the release boundary.

The model keeps BF16 master parameters and FP32 optimizer states. Attention,
routers, shared experts, embeddings, normalization layers, and selected
boundary layers remain in BF16. The policy backward pass currently uses TE's
dequantized path.

After every policy update, NeMo RL exports the updated BF16 weights to each
colocated vLLM engine. The rollout workers quantize routed-expert weights while
vLLM's native `reload_weights` API consumes the checkpoint-format stream before
the next rollout. Training remains unaware of the rollout representation.

## How It Works

```mermaid
flowchart LR
    A[TE per-token NVFP4 policy training] --> B[Megatron-Bridge HF weight export]
    B --> C[BF16 IPC stream]
    C --> D[vLLM native reload_weights]
    D --> E[Quantize routed experts at load time]
    E --> F[Restore kernel-format storage]
    F --> G[vLLM per-token NVFP4 rollout]
    G --> A
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
names and reconstructs tensors across the training TP, PP, and EP topology,
sending plain BF16. This is the same transport representation used by a
BF16-only run.

Each vLLM engine receives the BF16 stream over CUDA IPC. NeMo RL converts each
transport batch into owned checkpoint-format tensors and supplies one lazy
iterator to vLLM's native `reload_weights` API. Routed-expert projections are
quantized at load time, mirroring the fp8/mxfp8 real-quant rollout path. Gate
and up projections are quantized together under one shared per-expert global
scale because vLLM's fused-MoE loader keeps one gate/up scale per expert. Down
projections are quantized independently. Ignored BF16 expert layers
(`additional_ignore`) pass through unchanged.

The IPC allocation is acknowledged only after quantization and passthrough
copies no longer depend on it. The final COMPLETE message is acknowledged only
after `reload_weights` consumes the entire stream, performs layerwise
post-processing, and completes the final device fence. If preparation or load
fails, NeMo RL drains and acknowledges the remaining IPC messages so the sender
cannot deadlock, and the rollout worker raises instead of serving stale
weights.

### Per-token vLLM rollout

NeMo RL configures vLLM with the `nvfp4_pertoken` quantization mode. During
refit, vLLM loads the packed expert weights, prepares them for the FlashInfer
fused-MoE runtime, and rebuilds the affected kernels. Rollout activations are
quantized dynamically for each token; no static activation calibration is
required.

The native reload path restores checkpoint-shaped parameters layer by layer,
loads and processes them, then copies the resulting kernel-format tensors back
into the storage used by CUDA graphs. A failed or incomplete refit stops the
rollout instead of continuing with stale weights.

## Configuration

The complete validated Qwen3-30B-A3B configuration is:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml
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

## Validated Configuration

The shipped recipe captures the training and rollout settings used for the
end-to-end validation:

| Area | Validated setting |
|---|---|
| Model | Qwen3-30B-A3B-Base (`Qwen3MoeForCausalLM`) |
| Model layout | A routed-expert MoE block in every decoder layer |
| Hardware | 8 nodes with 4 NVIDIA GB200 GPUs per node |
| Algorithm and data | GRPO with DAPOMath17K training and DAPOMathAIME2024 validation |
| Sequence lengths | 2,048-token prompt and up to 20,480-token response |
| Global batch | 32 prompts with 16 generations each, for 512 sequences per step |
| Training parallelism | Megatron TP=2, EP=8, PP=1 |
| Rollout parallelism | Colocated vLLM TP=1, EP=1, PP=1 |
| Policy state | BF16 parameters, FP32 optimizer states, `fp4_param=false` |
| Refit | BF16 CUDA IPC stream into native vLLM `reload_weights` |

The validation recipe is a long-running configuration with periodic synchronous
checkpoints. Short smoke coverage remains available in
`grpo-qwen3-30ba3b-4n4g-megatron-te-nvfp4-pertoken-quick.yaml`.

## Current Limitations

The end-to-end contract is intended to support additional W4A4 per-token MoE
models, but this release has the following enforced or validated boundaries:

| Area | Current limitation |
|---|---|
| Model layout | Runtime validation accepts only `Qwen3MoeForCausalLM` with `decoder_sparse_step=1` and no `mlp_only_layers` |
| Quantized modules | Routed-expert MLP projections only; dense MLPs, attention, routers, and shared experts remain BF16 |
| Hardware and kernel | NVIDIA Blackwell with the FlashInfer TRT-LLM NVFP4 fused-MoE backend; GB200 is validated |
| Training precision | BF16 persistent parameters with `fp4_param=false`; backward computation uses TE's dequantized path |
| Rollout placement | Colocated vLLM rollout only; standalone evaluation is not supported because the dummy-loaded engine requires a refit first |
| vLLM parallelism | EP must be 1; the shipped end-to-end recipe uses TP=1 and PP=1; PP>1 requires the asynchronous engine |
| Cache and decoding | `kv_cache_dtype=auto`; speculative decoding is not supported |
| Refit transport | Default colocated CUDA IPC/ZMQ path (`refit_transport: null`) |
| Configuration | `generation.quant_cfg`, `generation.real_quant`, and explicit vLLM quantization/load-format overrides are mutually exclusive with this mode |
| Layer exclusions | `additional_ignore` can exclude only complete routed-expert layers and must match the BF16 boundary-layer selection |

Extending the feature to another architecture requires validating its parameter
names, MoE fusion and scale domains, vLLM kernel backend, and cold-load versus
warm-refit equivalence before relaxing the model-layout guard.

## Performance and GPU Memory

The following comparison uses Qwen3-30B-A3B-Base with DAPO on 8 nodes with
4 GB200 GPUs per node, a global batch size of 512, and a 20K response limit.
Training uses TP=2, EP=8, and PP=1; each vLLM engine uses TP=1. The generation
and step metrics are medians over the 502 logged steps shared by both runs
between steps 200 and 800. These are representative runs rather than a
controlled precision-only A/B; their configurations differ beyond numerical
precision. The final NVFP4 refit value is the mean and median from the 90-step
native `reload_weights` validation run; both are 18.42 seconds.

| Metric | BF16 | NVFP4 W4A4 | Change |
|---|---:|---:|---:|
| Rollout generation throughput | 457.9 tokens/s/GPU | 799.4 tokens/s/GPU | 1.75x |
| Generation time | 319.2 s | 144.7 s | 2.21x faster |
| Observed step time | 390.9 s | 224.8 s | 42.5% lower |
| Token-normalized end-to-end throughput | 376.1 tokens/s/GPU | 506.2 tokens/s/GPU | 1.35x |
| Weight transfer and update | 1.76 s | 18.42 s | 16.66 s higher |

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

Profiling the vLLM-side path showed that native weight loading, rather than IPC
transfer or NVFP4 arithmetic, dominates refit time:

| Refit phase | Representative time | Approximate share |
|---|---:|---:|
| vLLM load and layer processing | 13.13 s | 75% |
| NVFP4 quantization | 3.80 s | 22% |
| IPC wait and other work | 0.66 s | 4% |
| Finalization after all layers load | approximately 0 s | approximately 0% |

The load phase is Python-call-bound. Qwen3-30B-A3B emits 64,512 per-expert
checkpoint names per refit across 42 quantized layers and 128 experts, and each
name passes through vLLM's mapping, loader, and layerwise-reload bookkeeping.

A research-only prototype coalesced those outputs into eight full fused-expert
parameters per layer. In paired 90-step runs, the final per-expert path measured
18.42 seconds mean and median refit time; the prototype measured 6.31 seconds
mean and 6.38 seconds median, a 2.92x improvement. Both runs completed
successfully. The prototype is not part of this feature because complete
fused-parameter loading depends on vLLM's internal expert layout and should be
implemented upstream. The proposal and reproducible evidence are tracked in
[vLLM issue #53687](https://github.com/vllm-project/vllm/issues/53687), in
coordination with vLLM's streaming quantization-unit RFC
[#53192](https://github.com/vllm-project/vllm/issues/53192).

Refit remains a visible part of the step and an optimization target.
Training-quality curves will be added after the ongoing long-run stability
validation is complete.

## Roadmap

- Reduce refit latency through the native layer-fused MoE parameter loader
  proposed in [vLLM issue #53687](https://github.com/vllm-project/vllm/issues/53687)
  and quantization before the expert-parallel gather.
- Validate additional MoE architectures and layouts, then relax the Qwen3-only
  runtime guard where their naming, fusion, and scale contracts are compatible.
- Add native NVFP4 backward computation.
- Complete longer stability and model-quality studies and publish the training
  curves.
