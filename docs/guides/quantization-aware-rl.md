# Quantization-Aware RL (QARL)

Quantization-Aware RL (QARL) integrates [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) into the NeMo RL training loop, enabling quantization-aware training and generation for both GRPO and on-policy distillation workflows. QARL automatically quantizes a standard model at initialization, maintains quantizer state (amax values) throughout training, and transfers quantized state to vLLM during weight refit. By default, vLLM generation uses fake-quantized modules. Real-quant rollout instead exports the policy through ModelOpt's canonical Hugging Face schema and uses vLLM's native ModelOpt loader.

## Overview

In a standard NeMo RL loop, model weights are trained in full precision and refitted into vLLM for generation. QARL applies quantization-aware modules so that both the policy forward pass and rollout generation exercise quantized weights and, depending on the recipe, quantized activations. The policy backward pass remains in full precision, using the straight-through estimator to propagate gradients through the quantization nodes.

There are two vLLM rollout modes:

- **Fake-quant rollout**: vLLM receives folded full-precision weights and runs fake-quantized layers. This is the default when `policy.generation.quant_cfg` is set.
- **Real-quant rollout**: vLLM receives canonical packed weights and format-owned sidecars during every refit. Enable this with `policy.generation.real_quant: true`; `policy.quant_cfg` is the sole source of the rollout quantization format. ModelOpt owns packing and scale semantics, Megatron-Bridge owns distributed topology and Hugging Face naming, and vLLM selects the native runtime method.

See [Verified Configurations](#verified-configurations) for the workflow + recipe combinations that have been empirically validated, and [Supported Quantization Formats](#supported-quantization-formats) for the full set of available formats. Results are recipe- and model-specific: the generic W4A4 `NVFP4_DEFAULT_CFG` has known GRPO convergence issues, while the routed-expert Qwen3 W4A4 real-quant recipe below completed the documented single-seed campaign.

## Verified Configurations

The following table records prior recipe campaigns and the validation level reached. Real-quant results that predate the native vLLM loader migration require architecture-specific revalidation on the native path.

| Workflow | Quantization | Recipe | Status | Example Config |
|---|---|---|---|---|
| QA-Distillation | W4A4 | `NVFP4_DEFAULT_CFG` (NVFP4 weights + NVFP4 activations) | ✅ Converges | `examples/modelopt/qa_distillation_math_megatron.yaml` |
| QA-GRPO | W4A16 | `examples/modelopt/quant_configs/nvfp4_a16.yaml` (NVFP4 weights, native-dtype activations) | ✅ Converges | `examples/modelopt/qa_grpo_llama8b_megatron.v2.yaml` |
| QA-GRPO | W4A4 | `NVFP4_DEFAULT_CFG` | ⚠️ Known convergence issue | `examples/modelopt/qa_grpo_math_megatron.yaml` |
| QA-Distillation | W4A4 | `examples/modelopt/quant_configs/nano3_nvfp4_default.yaml` | ✅ Converges | `examples/modelopt/qa_distillation_nano3_megatron.yaml` |
| QA-GRPO | W4A16 | `NVFP4_MLP_WEIGHT_ONLY_CFG` | ✅ Smoke tested on MoE | `examples/modelopt/qa_grpo_qwen3_30ba3b_megatron.yaml` |
| QA-GRPO real quantization rollout | W4A16 | `W4A16_NVFP4_CFG` with `policy.generation.real_quant: true` | ✅ Native-loader 10-step functional run | `examples/modelopt/qa_grpo_qwen2_5_0p5b_megatron_real_quant.yaml` |
| QA-GRPO real quantization rollout | W4A16 | `examples/modelopt/quant_configs/nvfp4_a16_mlp_only.yaml` with `policy.generation.real_quant: true` | Historical convergence; native-loader revalidation pending | `examples/configs/recipes/llm/grpo-qwen3-8b-base-dapo-2n8g-long-megatron-qa-nvfp4-w4a16.yaml` |
| QA-GRPO real quantization rollout | W4A16 | `examples/modelopt/quant_configs/nano3_nvfp4_weightonly.yaml` with `policy.generation.real_quant: true` | Historical hybrid MoE/Mamba convergence; native-loader revalidation pending | `examples/configs/recipes/llm/grpo-nanov3-30ba3b-4n4g-megatron-qa-nvfp4-w4a16-real.yaml` |
| QA-GRPO real quantization rollout | W4A4 | `examples/modelopt/quant_configs/nvfp4_experts.yaml` with `policy.generation.real_quant: true` | Historical 300-step Qwen3-30B-A3B run; native-loader revalidation pending | `examples/configs/recipes/llm/grpo-qwen3-30ba3b-4n4g-megatron-qa-nvfp4-w4a4-real.yaml` |

The `nvfp4_a16.yaml` custom YAML enables NVFP4 e2m1 weight quantization (with dynamic e4m3 micro-block scales) and leaves activations unquantized; weights are still exercised through both Megatron training and vLLM generation. The `nvfp4_a16_mlp_only.yaml` recipe restricts W4A16 to MLP weights for real-quant rollout. The Nano3 `nano3_nvfp4_weightonly.yaml` recipe applies the same W4A16 weight-only format to supported MLP/MoE weights while its authored quantizer selection keeps sensitive paths in BF16.

## Simulated KV-Cache Quantization

QARL can apply ModelOpt fake quantization to the attention K/V tensors in both
the Megatron policy and the vLLM rollout model. Use the same quantization recipe
for both workers so calibrated K/V amax values from the policy can be
transferred to matching rollout quantizers during refit. The only supported
exception is rollout-only FP8 K/V fake quantization with
`use_constant_amax: true`; vLLM computes the constant amax locally, so no
policy-side K/V quantization or amax transfer is required.

```yaml
policy:
  quant_cfg: /absolute/path/to/examples/modelopt/quant_configs/kv_cache_fp8.yaml

  generation:
    backend: vllm
    quant_cfg: /absolute/path/to/examples/modelopt/quant_configs/kv_cache_fp8.yaml
```

The KV-only examples are:

- `kv_cache_fp8.yaml`: FP8 E4M3 K/V fake quantization with constant amax.
- `kv_cache_nvfp4.yaml`: calibrated NVFP4 K/V fake quantization with dynamic
  per-block scales and a calibrated global amax.

NVFP4 K/V quantization has shown training-quality degradation in current
experiments. It is intended for studying QARL-based recovery of PTQ accuracy
and alternative recipes, including mixed formats and quantizing K/V in only
selected layers.

These recipes enable only the attention K/V quantizers, isolating K/V fake
quantization from weight and other activation quantization. To combine K/V
quantization with other formats, define all required quantizers in a single
recipe.

This path does not enable vLLM's native FP8 KV-cache storage. Set neither
`policy.generation.real_quant` nor a native vLLM KV-cache dtype for these
simulated recipes.

## ModelOpt Layer Spec Toggle

For QARL configs, try setting `policy.disable_modelopt_layer_spec=true` first.
This keeps ModelOpt quantization enabled while using the standard Megatron layer
specs instead of ModelOpt's custom layer specs. This is usually faster and works
for most models, but it is not guaranteed for every architecture or recipe. If
you encounter errors with the standard Megatron layer specs, leave it unset or
set it to `false` to exercise ModelOpt's Megatron layer-spec path.

## Quantization-Aware GRPO (QA-GRPO)

### Configuration

The QA-GRPO config extends the standard Megatron GRPO config by adding quantization parameters. See [Verified Configurations](#verified-configurations) for the status of W4A4 vs W4A16 on GRPO.

```yaml
# examples/modelopt/qa_grpo_llama8b_megatron.v2.yaml
defaults: "../configs/grpo_math_8B_megatron.yaml"

policy:
  quant_cfg: "examples/modelopt/quant_configs/nvfp4_a16.yaml"
  quant_calib_data: "cnn_dailymail"
  quant_calib_size: 512
  quant_batch_size: 1
  quant_sequence_length: 2048

  generation:
    quant_cfg: "examples/modelopt/quant_configs/nvfp4_a16.yaml"
```

### Running QA-GRPO

**Single node (8 GPUs):**

```bash
uv run examples/run_grpo.py \
  --config examples/modelopt/qa_grpo_llama8b_megatron.v2.yaml \
  policy.model_name=meta-llama/Llama-3.1-8B-Instruct
```

**Via Slurm:**

```bash
COMMAND="uv run examples/run_grpo.py \
  --config examples/modelopt/qa_grpo_llama8b_megatron.v2.yaml \
  policy.model_name=meta-llama/Llama-3.1-8B-Instruct \
  checkpointing.checkpoint_dir=results/qa_grpo" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=1 \
    --account=YOUR_ACCOUNT \
    --job-name=qa-grpo \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

## Real-Quant ModelOpt Rollout

Real-quant rollout exercises deployment-style vLLM kernels during RL, not only the fake-quant training path. With `policy.generation.real_quant: true`, the Megatron policy worker exports canonical ModelOpt tensors through Megatron-Bridge, and vLLM loads them through its native ModelOpt checkpoint path. The policy training worker remains a QAT model.

NVFP4 W4A4 and W4A16 are supported recipe examples of this general path; see the validation matrix above for their current status. The policy recipe must select only formats and tensors supported by the target model and pinned vLLM loader; no separate generation-side recipe or ignore list is applied.

### Minimal Configuration

Start from a Megatron GRPO config, set the ModelOpt weight-only recipe on the policy, and enable real quantization for generation:

```yaml
policy:
  quant_cfg: examples/modelopt/quant_configs/nvfp4_a16_mlp_only.yaml

  generation:
    backend: vllm
    real_quant: true
```

For routed-expert Qwen3 MoE W4A4, use the block-16 E2M1 policy recipe:

```yaml
policy:
  quant_cfg: examples/modelopt/quant_configs/nvfp4_experts.yaml

  generation:
    backend: vllm
    real_quant: true
```

ModelOpt derives the canonical configuration from `policy.quant_cfg`, and vLLM selects the corresponding native method. NeMo RL does not infer W4A4, W4A16, or another format from recipe names. Unsupported formats or fused-layer combinations fail during setup or native loading.

For Nano3 W4A16 real-quant rollout, use the Nano3 weight-only policy recipe. The recipe itself selects supported tensors and leaves sensitive paths in BF16:

```yaml
policy:
  quant_cfg: examples/modelopt/quant_configs/nano3_nvfp4_weightonly.yaml

  generation:
    backend: vllm
    real_quant: true
    vllm_cfg:
      gpu_memory_utilization: 0.35
      enable_prefix_caching: false
```

The ready-to-run 2-node DAPO long-context recipe is:

```text
examples/configs/recipes/llm/grpo-qwen3-8b-base-dapo-2n8g-long-megatron-qa-nvfp4-w4a16.yaml
```

The ready-to-run Nano3 4-node x 4-GPU smoke recipe is:

```text
examples/configs/recipes/llm/grpo-nanov3-30ba3b-4n4g-megatron-qa-nvfp4-w4a16-real.yaml
```

The Qwen3-30B-A3B W4A4 real-quant recipe is:

```text
examples/configs/recipes/llm/grpo-qwen3-30ba3b-4n4g-megatron-qa-nvfp4-w4a4-real.yaml
```

This recipe contains the 300-step, 256-example-validation campaign settings.
The GB200 nightly driver overrides it to a two-step, 32-example smoke test.
Before running this MoE recipe, use jointly compatible ModelOpt and embedded
Megatron-Bridge revisions. ModelOpt must provide
`modelopt.torch.export.quantized_weight_export`; Megatron-Bridge must import the
functional APIs from that module and provide
`AutoBridge.export_hf_weight_groups_modelopt`. Older or independently selected
revisions are unsupported.

For a BF16 baseline, copy the recipe, remove `policy.quant_cfg`,
`policy.generation.quant_cfg`, and `policy.generation.real_quant`, and use distinct
checkpoint and log directories.

### Running the Example

From the repository root inside the NeMo RL container:

```bash
uv run --extra mcore --extra modelopt --extra vllm \
  examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-8b-base-dapo-2n8g-long-megatron-qa-nvfp4-w4a16.yaml
```

For Nano3:

```bash
uv run --extra mcore --extra modelopt --extra vllm \
  examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-nanov3-30ba3b-4n4g-megatron-qa-nvfp4-w4a16-real.yaml
```

For Qwen3 MoE W4A4:

```bash
uv run --extra mcore --extra modelopt --extra vllm \
  examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-30ba3b-4n4g-megatron-qa-nvfp4-w4a4-real.yaml
```

For Slurm, wrap the same command in `ray.sub` as shown in [Running QA-GRPO](#running-qa-grpo). Keep each quantized and BF16 comparison arm separate and use distinct checkpoint directories.

### Checkpoints and Fresh Starts

For a clean first-step comparison, use a new, empty `checkpointing.checkpoint_dir`; NeMo RL resumes automatically from the highest `step_*` directory it finds. The Megatron policy path also uses a converted startup checkpoint, so move aside both the training checkpoint and the converted Megatron checkpoint before launching:

```bash
# Training checkpoint: matches `checkpointing.checkpoint_dir` in your config.
mv checkpoints/grpo-qwen3-8b-base-dapo-2n-long-w4a16 checkpoints/grpo-qwen3-8b-base-dapo-2n-long-w4a16.old

# Converted Megatron checkpoint: under `$NRL_MEGATRON_CHECKPOINT_DIR` if set,
# else `$HF_HOME/nemo_rl` or `~/.cache/huggingface/nemo_rl`. The subdirectory is
# named after the HF model; see "Megatron Checkpoint Directory" below.
MEGATRON_CKPT_ROOT="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/nemo_rl}"
mv "$MEGATRON_CKPT_ROOT/<hf-model-subdir>" "$MEGATRON_CKPT_ROOT/<hf-model-subdir>.old"
```

If `NRL_MEGATRON_CHECKPOINT_DIR` is set, move aside the subdirectory used by the run. On first startup, the log should show that iteration 0 was saved or loaded from a freshly generated conversion checkpoint.

For long runs on queues with short wall times, enable periodic checkpointing and submit dependency jobs with `afterany` so the next job can resume from the checkpoint written by the previous job.

### Log Checks

A healthy W4A16 real-rollout run should include these lines or equivalent vLLM logs:

```text
quantization=modelopt
Detected ModelOpt NVFP4 checkpoint
Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM
MegatronQuantPolicyWorker[rank=0]: Packed ... groups of tensors
```

A W4A4 run should show the same ModelOpt checkpoint/refit signals and an activation-quantizing native method. Backend selection and validation are owned by vLLM rather than a NeMo RL format-specific check.

It should not include:

```text
Using rollout logprobs
negative scales
CUDA error: invalid argument
```

For an initial sanity check, compare the first `Generation KL Error` with the BF16 baseline. They should be close on step 1. A substantially larger W4A16 first-step KL usually means the rollout model does not match the policy model, the run reused a stale checkpoint, or the real-quant export/refit path did not load the expected tensors.

### Troubleshooting

| Symptom | Likely Cause | Action |
|---|---|---|
| vLLM does not log `quantization=modelopt` | `policy.generation.real_quant` is not set or generation is not using vLLM | Check the YAML under `policy.generation` |
| `Using rollout logprobs` appears | The run is bypassing policy/reference logprob computation | Do not use rollout logprobs for real-quant validation |
| First-step W4A16 `Generation KL Error` is much higher than BF16 | Stale resume state, a stale converted Megatron checkpoint on the Megatron path, or a refit/export mismatch | Use a fresh training checkpoint directory; on Megatron also move aside the converted startup checkpoint; confirm packed tensors are streamed |
| `negative scales` warning appears | Invalid or stale NVFP4 scale tensors reached vLLM | Use a fresh checkpoint directory and verify `policy.quant_cfg` resolves to a supported NVFP4 recipe |
| Nano3 first-step KL is high while dense W4A16 is healthy | The policy recipe quantized a Nano3-sensitive path or the exported payload does not match the native vLLM layout | Use `nano3_nvfp4_weightonly.yaml` as `policy.quant_cfg` and inspect the first refit |
| CUDA invalid argument during refit or generation | vLLM consumed malformed packed tensors or stale IPC state | Restart from a fresh job and inspect the first real-quant refit logs |

## Quantization-Aware Distillation (On-Policy QAD)

QAD combines on-policy distillation with quantization. The student model is quantized while the teacher remains in full precision, allowing the student to recover accuracy lost from quantization through knowledge distillation.

### Configuration

```yaml
# examples/modelopt/qa_distillation_math_megatron.yaml
defaults: "../configs/distillation_math_megatron.yaml"

policy:
    quant_cfg: "NVFP4_DEFAULT_CFG"
    quant_calib_data: "cnn_dailymail"
    quant_calib_size: 512
    quant_batch_size: 1
    quant_sequence_length: 2048

    generation:
        quant_cfg: "NVFP4_DEFAULT_CFG"
```

### Running QAD

```bash
uv run examples/run_distillation.py \
  --config examples/modelopt/qa_distillation_math_megatron.yaml \
  policy.model_name=Qwen/Qwen3-1.7B \
  teacher.model_name=Qwen/Qwen3-1.7B
```

## Quantization Parameters

These parameters are added under the `policy` section:

| Parameter | Description |
|---|---|
| `quant_cfg` | Quantization config. Accepts: a built-in ModelOpt config name (e.g. `"NVFP4_DEFAULT_CFG"`), a built-in ModelOpt PTQ recipe name (e.g. `"general/ptq/nvfp4_default-fp8_kv"`, suffix optional), or the path to a custom YAML recipe (e.g. `"examples/modelopt/quant_configs/nvfp4_a16.yaml"`). Use absolute paths for user-authored recipes in Ray/container workers. See `examples/modelopt/quant_configs/` for an example and `modelopt_recipes/general/ptq/` in Model-Optimizer for the canonical YAML format. |
| `quant_calib_data` | Dataset name used for calibration. See the [ModelOpt PTQ examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq) for supported datasets. |
| `quant_calib_size` | Number of samples for the calibration pass |
| `quant_batch_size` | Batch size during calibration |
| `quant_sequence_length` | Sequence length for calibration data |

For fake-quant rollout, `policy.generation.quant_cfg` should match `policy.quant_cfg`. For real-quant rollout, leave `policy.generation.quant_cfg` unset or `null`; the canonical vLLM configuration is generated exclusively from `policy.quant_cfg` after policy initialization.

Generation-specific parameters are added under `policy.generation`:

| Parameter | Description |
|---|---|
| `quant_cfg` | ModelOpt recipe used only for simulated/fake-quant vLLM generation. Set it to `null` for real-quant rollout. |
| `real_quant` | When `true`, vLLM uses native ModelOpt real-quant kernels and receives packed weights and deployment scales during refit. `policy.quant_cfg` selects the format; unsupported formats fail during setup. When unset or `false`, vLLM uses fake-quantized generation. |
| `real_quant_export_cpu_offload` | Boolean required for real-quant rollout. When `true`, packed export tensors are copied to CPU before refit. Set to `false` only for colocated CUDA-IPC refit without an explicit `refit_transport`; this avoids the CPU round trip at the cost of transient GPU memory. |

## Megatron Checkpoint Directory

On first run, the HF model is automatically converted to a Megatron checkpoint. By default, this checkpoint is saved under `$HF_HOME/nemo_rl` (or `~/.cache/huggingface/nemo_rl` if `HF_HOME` is not set). To control where the converted checkpoint is stored — for example, to keep it alongside your experiment outputs — set the `NRL_MEGATRON_CHECKPOINT_DIR` environment variable:

```bash
export NRL_MEGATRON_CHECKPOINT_DIR="/path/to/your/megatron/checkpoints"
```

## Differences from FP8 Training

QARL (via ModelOpt) and NeMo RL's built-in [FP8 training](../fp8.md) (via TransformerEngine) serve different purposes:

- **TransformerEngine FP8** focuses on **speeding up pre-training and fine-tuning** using real quantization. It replaces linear layers with FP8-native implementations that compute directly in reduced precision for throughput gains.

- **ModelOpt QARL** focuses on **recovering accuracy under quantization** using quantization-aware training. The policy forward pass uses quantized weights and, depending on the recipe, quantized activations while the backward pass uses full-precision gradients, so the model learns to be robust to quantization error. vLLM generation can use fake quantization or native real-quant ModelOpt formats supported by both ModelOpt export and the pinned vLLM.

## Supported Quantization Formats

- **Weight quantization**: per-tensor, per-channel, and block-wise formats are supported by fake-quant rollout. Real-quant rollout accepts only formats supported by both ModelOpt functional export and the pinned vLLM. Static NVFP4 preserves calibrated per-block and global amax state through distributed reconstruction.
- **Input (activation) quantization**: fake-quant rollout supports the existing ModelOpt recipes. Real-quant activation support is limited to canonical ModelOpt formats accepted by the pinned vLLM; required activation sidecars are emitted by ModelOpt and transported without format-specific NeMo RL handling.

## Exporting Megatron Checkpoints

After quantization-aware training, a Megatron checkpoint contains BF16 weights alongside quantization metadata (amax values, scales). To export it to a fully quantized HuggingFace format (with real low-precision weights), use the Megatron-Bridge export tool. The exported checkpoint is ready for deployment with inference engines like vLLM or TensorRT-LLM.

From within the NeMo RL container:

```bash
cd /opt/nemo-rl

PYTHONPATH=$PWD/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-} \
uv run --extra mcore --extra modelopt \
  torchrun --nproc_per_node <pipeline-parallel-size> \
  examples/modelopt/export_quantized_to_hf.py \
  --hf-model-id <hf-model-name-or-path> \
  --megatron-load-path <path-to-megatron-checkpoint>/policy/weights \
  --export-dir <output-hf-directory> \
  --tp 1 --pp <pipeline-parallel-size>
```

- `examples/modelopt/export_quantized_to_hf.py` is a thin wrapper around `Megatron-Bridge/examples/quantization/export.py`. All CLI flags pass through to the upstream script unchanged.
- `--hf-model-id` should point to the original (pre-training) HuggingFace model so that the exporter knows the model architecture and tokenizer.
- The `PYTHONPATH` prefix exposes Megatron-LM's `megatron.training` to the bridge script.
- **`--tp 1` is required**: modelopt currently does not support TP>1 at export time. Training at TP>1 is fine; the bridge re-shards on load via `mp_overrides`.
- **`--pp` can be >1** for large models that don't fit on one GPU. `--nproc_per_node` must equal `--pp` (since `--tp` is fixed at 1).

## Limitations

- **Generation**: Currently only vLLM is supported for generation.
- **DTensor backend**: Quantization support for the DTensor policy worker is not yet implemented.
- **Real-quant rollout**: The native-loader path has end-to-end functional coverage for dense W4A16. W4A4, fused-MoE, and hybrid MoE/Mamba paths require architecture-specific revalidation; fused MoE currently requires all experts local to each vLLM rank. The policy recipe must leave unsupported or sensitive paths in BF16.
- **Real-quant refit transport**: NIXL, custom checkpoint engines, `nccl_reshard`, and remote sparse-delta transports are not supported because they bypass vLLM's native layerwise reload lifecycle.
- **Router Replay (R3)**: R3 is supported on the Megatron policy path.
- **Real-quant activation formats**: Runtime support is the intersection of formats emitted by ModelOpt and complete fused-layer combinations accepted by the pinned vLLM. The architecture supports mixed formats without downstream format branches, but only entries in the validation table above should be treated as empirically validated.
- **Model support**: Dense Transformer, MoE (Mixture of Experts), and hybrid MoE/Mamba models are supported on the Megatron policy + vLLM generation path when Megatron-Bridge and ModelOpt support the model architecture and quantization recipe.
