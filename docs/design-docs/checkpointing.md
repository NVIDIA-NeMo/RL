# Exporting Checkpoints to Hugging Face Format

NeMo RL provides two checkpoint formats for Hugging Face models: Torch distributed and Hugging Face format. Torch distributed is used by default for efficiency, and Hugging Face format is provided for compatibility with Hugging Face's `AutoModel.from_pretrained` API. Note that Hugging Face format checkpoints save only the model weights, ignoring the optimizer states. It is recommended to use Torch distributed format to save intermediate checkpoints and to save a Hugging Face checkpoint only at the end of training. 

## Converting Torch Distributed Checkpoints to Hugging Face Format

A checkpoint converter is provided to convert a Torch distributed checkpoint to Hugging Face format after training:

```sh
uv run examples/converters/convert_dcp_to_hf.py --config=<YAML CONFIG USED DURING TRAINING> <ANY CONFIG OVERRIDES USED DURING TRAINING> --dcp-ckpt-path=<PATH TO DIST CHECKPOINT TO CONVERT> --hf-ckpt-path=<WHERE TO SAVE HF CHECKPOINT>
```

Usually Hugging Face checkpoints keep the weights and tokenizer together (which we also recommend for provenance). You can copy it afterwards. Here's an end-to-end example:

```sh
# Change to your appropriate checkpoint directory
CKPT_DIR=results/sft/step_10

uv run examples/converters/convert_dcp_to_hf.py --config=$CKPT_DIR/config.yaml --dcp-ckpt-path=$CKPT_DIR/policy/weights --hf-ckpt-path=${CKPT_DIR}-hf
rsync -ahP $CKPT_DIR/policy/tokenizer ${CKPT_DIR}-hf/
```

## Converting Megatron Checkpoints to Hugging Face Format

For models that were originally trained using the Megatron-LM backend, a separate converter is available to convert Megatron checkpoints to Hugging Face format. This script requires Megatron-Core, so make sure to launch the conversion with the `mcore` extra. For example,

```sh
CKPT_DIR=results/sft/step_10

uv run --extra mcore examples/converters/convert_megatron_to_hf.py --config=$CKPT_DIR/config.yaml --megatron-ckpt-path=$CKPT_DIR/policy/weights/iter_0000000/ --hf-ckpt-path=<path_to_save_hf_ckpt>
```

## Merging Megatron LoRA Adapter Checkpoints to Hugging Face Format

When training with [LoRA (Low-Rank Adaptation)](../guides/sft.md#lora-configuration) on the Megatron backend, the resulting checkpoint contains only the adapter weights alongside the base model configuration. To produce a standalone Hugging Face checkpoint suitable for inference or evaluation, use the LoRA merger script. It loads the base model, applies the LoRA adapter weights on top, and saves the merged result in Hugging Face format.

This script requires Megatron-Core, so make sure to launch with the `mcore` extra:

```sh
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt <path_to_base_megatron_checkpoint>/iter_0000000 \
    --adapter-ckpt <path_to_lora_adapter_checkpoint>/iter_0000000 \
    --hf-model-name <huggingface_model_name> \
    --hf-ckpt-path <output_path_for_merged_hf_model>
```

### Arguments

| Argument | Description |
|---|---|
| `--base-ckpt` | Path to the base model's Megatron checkpoint directory (the `iter_XXXXXXX` folder). |
| `--adapter-ckpt` | Path to the LoRA adapter's Megatron checkpoint directory (must contain a `run_config.yaml` with a `peft` section). |
| `--hf-model-name` | HuggingFace model identifier used to resolve the model architecture and tokenizer (e.g. `Qwen/Qwen2.5-7B`). |
| `--hf-ckpt-path` | Output directory for the merged HuggingFace checkpoint. Must not already exist. |

### Example

```sh
# Merge a LoRA adapter trained on Qwen2.5-7B back into a full HF checkpoint
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt ~/.cache/huggingface/nemo_rl/Qwen/Qwen2.5-7B/iter_0000000 \
    --adapter-ckpt results/sft_lora/step_100/policy/weights/iter_0000000 \
    --hf-model-name Qwen/Qwen2.5-7B \
    --hf-ckpt-path results/sft_lora/merged_hf
```

The merged checkpoint can then be used directly with `AutoModelForCausalLM.from_pretrained` or passed to the [evaluation pipeline](../guides/eval.md).
