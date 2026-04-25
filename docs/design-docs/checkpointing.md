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

For models that were originally trained using the Megatron-LM backend, a separate converter is available to convert Megatron checkpoints to Hugging Face format. This script requires Megatron-Core, so make sure to launch the conversion with the `mcore` extra. 

Use `--hf-model-name` argument to override the model name mentioned in `config.yaml`. This is useful for models like GPT-OSS whose base checkpoint precision(mxfp4) is different from supported export precision(bfloat16) in Megatron-Bridge, [Ref](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/gpt_oss).

For example,
```sh
CKPT_DIR=results/sft/step_10

uv run --extra mcore examples/converters/convert_megatron_to_hf.py \
  --config=$CKPT_DIR/config.yaml \
  --hf-model-name <repo>/model_name \
  --megatron-ckpt-path=$CKPT_DIR/policy/weights/iter_0000000/ \
  --hf-ckpt-path=<path_to_save_hf_ckpt>
```

## Converting Megatron LoRA Adapter Checkpoints to Hugging Face Format

When training with [LoRA (Low-Rank Adaptation)](../guides/sft.md#lora-configuration) on the Megatron backend, the resulting checkpoint contains only the adapter weights alongside the base model configuration. The `convert_lora_to_hf.py` script supports two export modes:

- **Merged**: fold the LoRA adapter into the base model and export a single standalone HuggingFace checkpoint.
- **Adapter-only**: export only the LoRA adapter weights in [HuggingFace PEFT](https://huggingface.co/docs/peft) format, keeping the base model separate.

This script requires Megatron-Core, so make sure to launch with the `mcore` extra.

### Option A — Merged checkpoint

Loads the base model, applies the LoRA adapter weights on top, and saves the merged result in HuggingFace format. The output can be used directly with `AutoModelForCausalLM.from_pretrained` or passed to the [evaluation pipeline](../guides/eval.md).

**Example:**

```sh
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt <path_to_base_megatron_checkpoint>/iter_0000000 \
    --adapter-ckpt <path_to_lora_adapter_checkpoint>/iter_0000000 \
    --hf-model-name <huggingface_model_name> \
    --hf-ckpt-path <output_path_for_merged_hf_model>
```

### Option B — Adapter-only (PEFT format)

Exports only the LoRA adapter weights in HuggingFace PEFT format without merging into the base model. This is useful when you want to serve the base model and adapter separately (e.g. with vLLM's LoRA support).

Although the output is adapter-only, the converter still needs `--base-ckpt` to reconstruct the Megatron model, apply the LoRA modules, and load the adapter weights before exporting them to PEFT format.

**Example:**

```sh
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt <path_to_base_megatron_checkpoint>/iter_0000000 \
    --adapter-only \
    --adapter-ckpt <path_to_lora_adapter_checkpoint>/iter_0000000 \
    --hf-model-name <huggingface_model_name> \
    --hf-ckpt-path <output_path_for_hf_adapter>
```

### Arguments

| Argument | Description |
|---|---|
| `--base-ckpt` | Path to the base model's Megatron checkpoint directory (the `iter_XXXXXXX` folder). Required for both merged and adapter-only export. |
| `--adapter-ckpt` | Path to the LoRA adapter's Megatron checkpoint directory (must contain a `run_config.yaml` with a `peft` section). |
| `--hf-model-name` | HuggingFace model identifier used to resolve the model architecture and tokenizer (e.g. `Qwen/Qwen2.5-7B`). |
| `--hf-ckpt-path` | Output directory for the exported HuggingFace checkpoint or adapter. Must not already exist. |
| `--adapter-only` | Export only the LoRA adapter in HuggingFace PEFT format without merging into the base model. |
