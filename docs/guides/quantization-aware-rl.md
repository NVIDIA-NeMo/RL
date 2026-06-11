# Quantization-Aware RL (QARL)

Quantization-Aware RL (QARL) integrates [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) into the NeMo RL training loop, enabling quantization-aware training and generation for both GRPO and on-policy distillation workflows. QARL automatically quantizes a standard model at initialization, maintains quantizer state (amax values) throughout training, and transfers quantized state to vLLM during weight refit. By default, vLLM generation uses fake-quantized modules. For NVFP4 W4A16 rollout experiments, NeMo RL can instead stream packed real-quant ModelOpt NVFP4 weights into vLLM.

## Overview

In a standard NeMo RL loop, model weights are trained in full precision and refitted into vLLM for generation. QARL applies quantization-aware modules so that both the policy forward pass and rollout generation exercise quantized weights and, depending on the recipe, quantized activations. The policy backward pass remains in full precision, using the straight-through estimator to propagate gradients through the quantization nodes.

There are two vLLM rollout modes:

- **Fake-quant rollout**: vLLM receives folded full-precision weights and runs fake-quantized layers. This is the default when `policy.generation.quant_cfg` is set.
- **Real-quant rollout**: vLLM is initialized with ModelOpt NVFP4 kernels and receives packed NVFP4 weights plus scale tensors during every refit. Enable this with `policy.generation.real_quant: true`.

See [Verified Configurations](#verified-configurations) for the workflow + recipe combinations that have been empirically validated, and [Supported Quantization Formats](#supported-quantization-formats) for the full set of available formats. W4A4 (`NVFP4_DEFAULT_CFG`) converges for on-policy distillation but has been observed to have convergence issues on GRPO; W4A16 (NVFP4 weights, native-dtype activations) works for GRPO.

## Verified Configurations

The following workflow + quantization recipe combinations have been validated end-to-end (Megatron training + NVFP4-quantized vLLM generation + held-out validation):

| Workflow | Quantization | Recipe | Status | Example Config |
|---|---|---|---|---|
| QA-Distillation | W4A4 | `NVFP4_DEFAULT_CFG` (NVFP4 weights + NVFP4 activations) | ✅ Converges | `examples/modelopt/qa_distillation_math_megatron.yaml` |
| QA-GRPO | W4A16 | `examples/modelopt/quant_configs/nvfp4_a16.yaml` (NVFP4 weights, native-dtype activations) | ✅ Converges | `examples/modelopt/qa_grpo_llama8b_megatron.v2.yaml` |
| QA-GRPO | W4A8 | `examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml` (NVFP4 weights, FP8 input activations) | ✅ Converges | `examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron-qa-nvfp4-w4a8-fake.yaml` |
| QA-GRPO | W4A4 | `NVFP4_DEFAULT_CFG` | ⚠️ Known convergence issue | `examples/modelopt/qa_grpo_math_megatron.yaml` |
| QA-Distillation | W4A4 | `examples/modelopt/quant_configs/nano3_nvfp4_default.yaml` | ✅ Converges | `examples/modelopt/qa_distillation_nano3_megatron.yaml` |
| QA-GRPO | W4A16 | `NVFP4_MLP_WEIGHT_ONLY_CFG` | ✅ Smoke tested on MoE | `examples/modelopt/qa_grpo_qwen3_30ba3b_megatron.yaml` |
| QA-GRPO real quantization rollout | W4A16 | `examples/modelopt/quant_configs/nvfp4_a16.yaml` with `policy.generation.real_quant: true` | ✅ Converges | `examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron-qa-nvfp4-w4a16.yaml` |

The `nvfp4_a16.yaml` custom YAML enables NVFP4 e2m1 weight quantization (with dynamic e4m3 micro-block scales) and leaves activations unquantized; weights are still exercised through both Megatron training and vLLM generation. The `nvfp4_w4a8_fp8.yaml` recipe uses the same NVFP4 weight format and enables FP8 e4m3 input activation fake quantization.

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

## Real-Quant NVFP4 Rollout (W4A16)

Real-quant rollout is intended for checking the deployment-style vLLM path during RL, not only the fake-quant training path. With `policy.generation.real_quant: true`, the Megatron policy worker exports ModelOpt QAT weights as packed NVFP4 tensors during refit, and the vLLM worker loads them into ModelOpt NVFP4 layers. This exercises vLLM's real FP4 kernel path during rollout while the policy training worker remains a QAT model.

This path is validated for W4A16.

### Minimal Configuration

Start from a Megatron GRPO config and add the ModelOpt weight-only recipe to both the policy and generation sections:

```yaml
policy:
  quant_cfg: examples/modelopt/quant_configs/nvfp4_a16.yaml

  generation:
    backend: vllm
    quant_cfg: examples/modelopt/quant_configs/nvfp4_a16.yaml
    real_quant: true
```

The ready-to-run 1-node DAPO smoke recipe is:

```text
examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron-qa-nvfp4-w4a16.yaml
```

Use the matching BF16 recipe as the baseline:

```text
examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron.yaml
```

### Running the Example

From the repository root inside the NeMo RL container:

```bash
uv run --extra mcore --extra modelopt --extra vllm \
  examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron-qa-nvfp4-w4a16.yaml
```

For a BF16 comparison run:

```bash
uv run --extra mcore --extra vllm \
  examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron.yaml
```

For Slurm, wrap the same command in `ray.sub` as shown in [Running QA-GRPO](#running-qa-grpo). Keep the W4A16 and BF16 runs separate and use distinct checkpoint directories.

### Checkpoints and Fresh Starts

Real-quant rollout is sensitive to stale Megatron conversion checkpoints. For a clean first-step comparison, move aside or remove both the training checkpoint and the converted Megatron checkpoint before launching:

```bash
mv checkpoints/grpo-qwen2.5-0.5b-dapo-1n8g-w4a16 checkpoints/grpo-qwen2.5-0.5b-dapo-1n8g-w4a16.old
mv /path/to/megatron_ckpt/dapo8b_2n_long_w4a16 /path/to/megatron_ckpt/dapo8b_2n_long_w4a16.old
```

If `NRL_MEGATRON_CHECKPOINT_DIR` is set, clear the subdirectory used by the run. On first startup, the log should show that iteration 0 was saved or loaded from a freshly generated conversion checkpoint.

For long runs on queues with short wall times, enable periodic checkpointing and submit dependency jobs with `afterany` so the next job can resume from the checkpoint written by the previous job.

### Log Checks

A healthy W4A16 real-rollout run should include these lines or equivalent vLLM logs:

```text
quantization=modelopt_fp4
Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM
MegatronQuantPolicyWorker[rank=0]: Packed ... groups of tensors
```

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
| vLLM does not log `quantization=modelopt_fp4` | `policy.generation.real_quant` is not set or generation is not using vLLM | Check the YAML under `policy.generation` |
| `Using rollout logprobs` appears | The run is bypassing policy/reference logprob computation | Do not use rollout logprobs for real-quant validation |
| First-step W4A16 `Generation KL Error` is much higher than BF16 | Stale converted Megatron checkpoint or refit/export mismatch | Clear checkpoints and rerun; confirm packed tensors are streamed |
| `negative scales` warning appears | Invalid or stale NVFP4 scale tensors reached vLLM | Clear checkpoints and verify `nvfp4_a16.yaml` is used for both policy and generation |
| CUDA invalid argument during refit or generation | vLLM consumed malformed packed tensors or stale IPC state | Restart from a fresh job and inspect the first real-quant refit logs |

## Fake-Quant NVFP4 Rollout (W4A8)

W4A8 rollout is supported through the fake-quant vLLM path. The policy and vLLM workers both use a ModelOpt recipe with NVFP4 weights and FP8 input activations, while refit transfers folded weights and quantizer amax state instead of packed deployment tensors.

```yaml
policy:
  quant_cfg: examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml

  generation:
    backend: vllm
    quant_cfg: examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml
```

The ready-to-run 1-node DAPO smoke recipe is:

```text
examples/configs/recipes/llm/grpo-qwen2.5-0.5b-dapo-1n8g-megatron-qa-nvfp4-w4a8-fake.yaml
```

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
| `quant_cfg` | Quantization config. Accepts: a built-in ModelOpt config name (e.g. `"NVFP4_DEFAULT_CFG"`), a built-in ModelOpt PTQ recipe name (e.g. `"general/ptq/nvfp4_default-fp8_kv"`, suffix optional), or the path to a custom YAML recipe (e.g. `"examples/modelopt/quant_configs/nvfp4_a16.yaml"`). See `examples/modelopt/quant_configs/` for an example and `modelopt_recipes/general/ptq/` in Model-Optimizer for the canonical YAML format. |
| `quant_calib_data` | Dataset name used for calibration. See the [ModelOpt PTQ examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq) for supported datasets. |
| `quant_calib_size` | Number of samples for the calibration pass |
| `quant_batch_size` | Batch size during calibration |
| `quant_sequence_length` | Sequence length for calibration data |

The `policy.generation.quant_cfg` should match `policy.quant_cfg` to ensure consistent quantization between training and generation.

Generation-specific parameters are added under `policy.generation`:

| Parameter | Description |
|---|---|
| `quant_cfg` | Quantization config used by the vLLM generation worker. For QARL, this should normally match `policy.quant_cfg`. |
| `real_quant` | When `true`, vLLM uses ModelOpt NVFP4 real kernels and receives packed quantized weights during refit. When unset or `false`, vLLM uses fake-quantized generation. |
| `real_quant_ignore` | Optional list of vLLM parameter name patterns that should stay in native dtype during real-quant rollout. If omitted, NeMo RL uses the default ModelOpt NVFP4 ignore set for sensitive layers such as attention and output heads. |

## Megatron Checkpoint Directory

On first run, the HF model is automatically converted to a Megatron checkpoint. By default, this checkpoint is saved under `$HF_HOME/nemo_rl` (or `~/.cache/huggingface/nemo_rl` if `HF_HOME` is not set). To control where the converted checkpoint is stored — for example, to keep it alongside your experiment outputs — set the `NRL_MEGATRON_CHECKPOINT_DIR` environment variable:

```bash
export NRL_MEGATRON_CHECKPOINT_DIR="/path/to/your/megatron/checkpoints"
```

## Differences from FP8 Training

QARL (via ModelOpt) and NeMo RL's built-in [FP8 training](../fp8.md) (via TransformerEngine) serve different purposes:

- **TransformerEngine FP8** focuses on **speeding up pre-training and fine-tuning** using real quantization. It replaces linear layers with FP8-native implementations that compute directly in reduced precision for throughput gains.

- **ModelOpt QARL** focuses on **recovering accuracy under quantization** using quantization-aware training. The policy forward pass uses quantized weights and, depending on the recipe, quantized activations while the backward pass uses full-precision gradients, so the model learns to be robust to quantization error. vLLM generation can run fake-quantized layers for W4A8/W4A16 recipes.
  W4A16 experiments can also use real ModelOpt NVFP4 kernels.

## Supported Quantization Formats

- **Weight quantization**: per-tensor, per-channel, and block-wise formats are all supported. In fake-quant rollout, weights are pre-folded on the policy (Megatron) side before transfer to vLLM. In W4A16 real-quant rollout, weights are packed as NVFP4 tensors and streamed with their scale tensors.
- **Input (activation) quantization**: only per-tensor is supported. The input quantizer amax is synced to vLLM as a per-tensor scalar.

## Exporting Checkpoints

After quantization-aware training, the Megatron checkpoint contains BF16 weights alongside quantization metadata (amax values, scales). To export a trained checkpoint to a fully quantized HuggingFace format (with real low-precision weights), use the Megatron-Bridge export tool. The exported checkpoint is ready for deployment with inference engines like vLLM or TensorRT-LLM.

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
- **Real-quant rollout**: W4A16 real rollout is supported for dense vLLM ModelOpt NVFP4 layers.
- **W4A8 rollout**: W4A8 is supported through fake-quant rollout.
- **Input quantization**: Only per-tensor input (activation) quantization is supported.
- **Model support**: Dense Transformer, MoE (Mixture of Experts), and hybrid MoE/Mamba models are supported on the Megatron policy + vLLM generation path when Megatron-Bridge and ModelOpt support the model architecture and quantization recipe. MoE/Mamba support is currently covered by smoke-tested example configs rather than broad convergence guarantees.
