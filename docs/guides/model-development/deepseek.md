# Use DeepSeek Models

## Overview

This guide covers using DeepSeek models in NeMo RL. Two DeepSeek models are relevant:

- **DeepSeek V3**: The full-scale MoE model with Multi-Head Latent Attention. Requires checkpoint conversion from FP8 to BF16 format before use.
- **DeepSeek-R1-Distill-Qwen-1.5B**: A distilled reasoning model based on Qwen architecture. Used for Chain-of-Thought reasoning tasks and ready to use from Hugging Face.

## Convert DeepSeek V3 Checkpoint

DeepSeek V3 releases in FP8 format. Convert it to BF16 for use with NeMo RL.

Adapted from [NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html).

```bash
# clone DeepSeek V3 weights from HF  (This can take hours)
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3 DeepSeek-V3-FP8

# clone DeepSeek-V3 code
git clone https://github.com/deepseek-ai/DeepSeek-V3.git

# transformers (since v4.23.0) (checks for tensor format in the metadata)[https://github.com/huggingface/transformers/blob/9ae22fe3c1b81f99a764d382054b6ebe2b025bd4/src/transformers/modeling_utils.py#L388]
cd DeepSeek-V3/inference
sed -i '88{s/new_safetensor_file/new_safetensor_file, metadata={"format": "pt"}/}' fp8_cast_bf16.py

# convert weights
uv run python fp8_cast_bf16.py --input-fp8-hf-path ../../DeepSeek-V3-FP8 --output-bf16-hf-path ../../DeepSeek-V3-BF16

# copy other files
cd ../..
cp DeepSeek-V3-FP8/{tokenizer_config.json,tokenizer.json,modeling_deepseek.py,configuration_deepseek.py} DeepSeek-V3-BF16/

# copy config.json, remove `quantization_config`, and set num_nextn_predict_layers to 0 (we currently do not support mtp):
jq 'del(.quantization_config) | .num_nextn_predict_layers=0' DeepSeek-V3-FP8/config.json > DeepSeek-V3-BF16/config.json
```

## Use DeepSeek-R1-Distill-Qwen

DeepSeek-R1-Distill-Qwen-1.5B is available from Hugging Face. Use it directly in NeMo RL configurations.

### Basic Configuration

```yaml
policy:
  model_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  train_global_batch_size: 64
  train_micro_batch_size: 1
  max_total_sequence_length: 8192
  
  dtensor_cfg:
    cpu_offload: true
    sequence_parallel: true
    activation_checkpointing: true
  
  generation:
    vllm_kwargs:
      compilation_config:
        use_inductor: false  # Important for this model
```

### Key Configuration Notes

- **`use_inductor: false`**: This model requires disabling Torch Inductor in vLLM for stable generation
- **DTensor**: Works well with CPU offload and sequence parallelism for memory efficiency
- **Context length**: Supports progressive training at 8K, 16K, and 24K sequence lengths

## Training Examples

For a complete example of training DeepSeek-R1-Distill-Qwen on the DeepScaleR dataset with GRPO:

- See [GRPO on DeepScaler](../../learning-resources/examples/grpo-deepscaler.md) for multi-stage training
- Reference config: `examples/configs/recipes/llm/grpo-deepscaler-1.5b-8K.yaml`

## Next Steps

- [Add New Models](add-new-models.md) - Learn about model integration and validation
- [Model Quirks](model-quirks.md) - Understand model-specific behaviors
- [Training Algorithms](../training-algorithms/index.md) - Apply GRPO and other algorithms
