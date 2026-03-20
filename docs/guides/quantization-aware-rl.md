# Quantization-Aware RL (QARL)

Quantization-Aware RL (QARL) integrates [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer) into the NeMo RL training loop, enabling quantization-aware training and generation for both GRPO and on-policy distillation workflows. QARL automatically quantizes a standard model at initialization, maintains quantizer state (amax values) throughout training, and transfers them to vLLM during weight refit for fake-quantized generation. The goal is to reduce quantization error through quantization-aware training.

## Overview

In a standard NeMo RL loop, model weights are trained in full precision and refitted into vLLM for generation. QARL applies fake quantization so that both the policy forward pass (training) and the rollout forward pass (vLLM generation) use quantized weights and activations. The policy backward pass remains in full precision, using the straight-through estimator to propagate gradients through the quantization nodes.

See [Supported Quantization Formats](#supported-quantization-formats) for details on which formats are available. The examples below use NVFP4 (`NVFP4_DEFAULT_CFG`), which offers strong compression while maintaining accuracy through quantization-aware training.

## Quantization-Aware GRPO (QA-GRPO)

### Configuration

The QA-GRPO config extends the standard Megatron GRPO config by adding quantization parameters:

```yaml
# examples/modelopt/qa_grpo_math_megatron.yaml
defaults: "../configs/grpo_math_1B_megatron.yaml"

policy:
  quant_cfg: "NVFP4_DEFAULT_CFG"
  quant_calib_data: "cnn_dailymail"
  quant_calib_size: 512
  quant_batch_size: 1
  quant_sequence_length: 2048

  generation:
    quant_cfg: "NVFP4_DEFAULT_CFG"
```

### Running QA-GRPO

**Single node (8 GPUs):**

```bash
uv run examples/run_grpo.py \
  --config examples/modelopt/qa_grpo_math_megatron.yaml \
  policy.model_name=Qwen/Qwen3-1.7B
```

**Via Slurm:**

```bash
COMMAND="uv run examples/run_grpo.py \
  --config examples/modelopt/qa_grpo_math_megatron.yaml \
  policy.model_name=Qwen/Qwen3-1.7B \
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
| `quant_cfg` | ModelOpt quantization config name (e.g. `"NVFP4_DEFAULT_CFG"`) or a path to a custom config file with variable name (e.g. `"path/to/config.py:MY_CONFIG"`). See `examples/modelopt/quant_configs/` for examples. |
| `quant_calib_data` | Dataset name used for calibration. See the [ModelOpt PTQ examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_ptq) for supported datasets. |
| `quant_calib_size` | Number of samples for the calibration pass |
| `quant_batch_size` | Batch size during calibration |
| `quant_sequence_length` | Sequence length for calibration data |

The `policy.generation.quant_cfg` should match `policy.quant_cfg` to ensure consistent quantization between training and generation.

## Differences from FP8 Training

QARL (via ModelOpt) and NeMo RL's built-in [FP8 training](../fp8.md) (via TransformerEngine) serve different purposes:

- **TransformerEngine FP8** focuses on **speeding up pre-training and fine-tuning** using real quantization. It replaces linear layers with FP8-native implementations that compute directly in reduced precision for throughput gains.

- **ModelOpt QARL** focuses on **recovering accuracy under quantization** using fake quantization (quantization-aware training). The forward pass uses quantized weights and activations while the backward pass uses full-precision gradients, so the model learns to be robust to quantization error. Both training and vLLM generation use fake-quantized forward passes for consistency.

## Supported Quantization Formats

- **Weight quantization**: per-tensor, per-channel, and block-wise formats are all supported. Weights are pre-folded on the policy (Megatron) side before transfer to vLLM.
- **Input (activation) quantization**: only per-tensor is supported. The input quantizer amax is synced to vLLM as a per-tensor scalar.

## Limitations

- **Generation**: Currently only vLLM is supported for generation.
- **DTensor backend**: Quantization support for the DTensor policy worker is not yet implemented.
- **Input quantization**: Only per-tensor input (activation) quantization is supported.
- **Model support**: MoE (Mixture of Experts) and Mamba models are currently not supported.