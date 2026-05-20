# Quantization-Aware RL (QARL)

Quantization-Aware RL (QARL) integrates [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) into the NeMo RL training loop, enabling quantization-aware training and generation for both GRPO and on-policy distillation workflows. QARL automatically quantizes a standard model at initialization, maintains quantizer state (amax values) throughout training, and transfers them to vLLM during weight refit for fake-quantized generation. The goal is to reduce quantization error through quantization-aware training.

## Overview

In a standard NeMo RL loop, model weights are trained in full precision and refitted into vLLM for generation. QARL applies fake quantization so that both the policy forward pass (training) and the rollout forward pass (vLLM generation) use quantized weights and activations. The policy backward pass remains in full precision, using the straight-through estimator to propagate gradients through the quantization nodes.

See [Verified Configurations](#verified-configurations) for the workflow + recipe combinations that have been empirically validated, and [Supported Quantization Formats](#supported-quantization-formats) for the full set of available formats. W4A4 (`NVFP4_DEFAULT_CFG`) converges for on-policy distillation but has been observed to have convergence issues on GRPO; W4A16 (NVFP4 weights, native-dtype activations) works for GRPO.

## Verified Configurations

The following workflow + quantization recipe combinations have been validated end-to-end (Megatron training + NVFP4-quantized vLLM generation + held-out validation):

| Workflow | Quantization | Recipe | Status | Example Config |
|---|---|---|---|---|
| QA-Distillation | W4A4 | `NVFP4_DEFAULT_CFG` (NVFP4 weights + NVFP4 activations) | ✅ Converges | `examples/modelopt/qa_distillation_math_megatron.yaml` |
| QA-GRPO | W4A16 | `examples/modelopt/quant_configs/nvfp4_a16.yaml` (NVFP4 weights, native-dtype activations) | ✅ Converges | `examples/modelopt/qa_grpo_llama8b_megatron.yaml` |
| QA-GRPO | W4A4 | `NVFP4_DEFAULT_CFG` | ⚠️ Known convergence issue | `examples/modelopt/qa_grpo_math_megatron.yaml` |

The `nvfp4_a16.yaml` custom YAML enables NVFP4 e2m1 weight quantization (with dynamic e4m3 micro-block scales) and leaves activations unquantized; weights are still exercised through both Megatron training and vLLM generation.

## Quantization-Aware GRPO (QA-GRPO)

### Configuration

The QA-GRPO config extends the standard Megatron GRPO config by adding quantization parameters. See [Verified Configurations](#verified-configurations) for the status of W4A4 vs W4A16 on GRPO.

```yaml
# examples/modelopt/qa_grpo_llama8b_megatron.yaml
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
  --config examples/modelopt/qa_grpo_llama8b_megatron.yaml \
  policy.model_name=meta-llama/Llama-3.1-8B-Instruct
```

**Via Slurm:**

```bash
COMMAND="uv run examples/run_grpo.py \
  --config examples/modelopt/qa_grpo_llama8b_megatron.yaml \
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

## Megatron Checkpoint Directory

On first run, the HF model is automatically converted to a Megatron checkpoint. By default, this checkpoint is saved under `$HF_HOME/nemo_rl` (or `~/.cache/huggingface/nemo_rl` if `HF_HOME` is not set). To control where the converted checkpoint is stored — for example, to keep it alongside your experiment outputs — set the `NRL_MEGATRON_CHECKPOINT_DIR` environment variable:

```bash
export NRL_MEGATRON_CHECKPOINT_DIR="/path/to/your/megatron/checkpoints"
```

## Differences from FP8 Training

QARL (via ModelOpt) and NeMo RL's built-in [FP8 training](../fp8.md) (via TransformerEngine) serve different purposes:

- **TransformerEngine FP8** focuses on **speeding up pre-training and fine-tuning** using real quantization. It replaces linear layers with FP8-native implementations that compute directly in reduced precision for throughput gains.

- **ModelOpt QARL** focuses on **recovering accuracy under quantization** using fake quantization (quantization-aware training). The forward pass uses quantized weights and activations while the backward pass uses full-precision gradients, so the model learns to be robust to quantization error. Both training and vLLM generation use fake-quantized forward passes for consistency.

## Supported Quantization Formats

- **Weight quantization**: per-tensor, per-channel, and block-wise formats are all supported. Weights are pre-folded on the policy (Megatron) side before transfer to vLLM.
- **Input (activation) quantization**: only per-tensor is supported. The input quantizer amax is synced to vLLM as a per-tensor scalar.

## Exporting Checkpoints

After quantization-aware training, the Megatron checkpoint contains BF16 weights alongside quantization metadata (amax values, scales). To export a trained checkpoint to a fully quantized HuggingFace format (with real low-precision weights), use the Megatron-Bridge export tool. The exported checkpoint is ready for deployment with inference engines like vLLM or TensorRT-LLM.

From within the NeMo RL container:

```bash
cd /opt/nemo-rl

PYTHONPATH=$PWD/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-} \
uv run --extra mcore --extra modelopt \
  torchrun --nproc_per_node <pipeline-parallel-size> \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/examples/quantization/export.py \
  --hf-model-id <hf-model-name-or-path> \
  --megatron-load-path <path-to-megatron-checkpoint>/policy/weights \
  --export-dir <output-hf-directory> \
  --tp 1 --pp <pipeline-parallel-size>
```

- `--hf-model-id` should point to the original (pre-training) HuggingFace model so that the exporter knows the model architecture and tokenizer.
- The `PYTHONPATH` prefix exposes Megatron-LM's `megatron.training` to the bridge script.
- **`--tp 1` is required**: modelopt currently does not support TP>1 at export time. Training at TP>1 is fine; the bridge re-shards on load via `mp_overrides`.
- **`--pp` can be >1** for large models that don't fit on one GPU. `--nproc_per_node` must equal `--pp` (since `--tp` is fixed at 1).

## Limitations

- **Generation**: Currently only vLLM is supported for generation.
- **DTensor backend**: Quantization support for the DTensor policy worker is not yet implemented.
- **Input quantization**: Only per-tensor input (activation) quantization is supported.
- **Model support**: MoE (Mixture of Experts) and Mamba models are currently not supported.