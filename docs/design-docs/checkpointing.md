# Checkpointing with Hugging Face Models 

NeMo RL provides two checkpoint formats for Hugging Face models: Torch distributed and Hugging Face format. Torch distributed is used by default for efficiency, and Hugging Face format is provided for compatibility with Hugging Face's `AutoModel.from_pretrained` API. Note that Hugging Face format checkpoints save only the model weights, ignoring the optimizer states. It is recommended to use Torch distributed format to save intermediate checkpoints and to save a Hugging Face checkpoint only at the end of training. 

A checkpoint converter is provided to convert a Torch distributed checkpoint checkpoint to Hugging Face format after training:

```sh
uv run examples/convert_dcp_to_hf.py --config=<YAML CONFIG USED DURING TRAINING> <ANY CONFIG OVERRIDES USED DURING TRAINING> --dcp-ckpt-path=<PATH TO DIST CHECKPOINT TO CONVERT> --hf-ckpt-path=<WHERE TO SAVE HF CHECKPOINT>
```

Usually Hugging Face checkpoints keep the weights and tokenizer together (which we also recommend for provenance). You can copy it afterwards. Here's an end-to-end example:

```sh
# Change to your appropriate checkpoint directory
CKPT_DIR=results/sft/step_10

uv run examples/convert_dcp_to_hf.py --config=$CKPT_DIR/config.yaml --dcp-ckpt-path=$CKPT_DIR/policy/weights --hf-ckpt-path=${CKPT_DIR}-hf
rsync -ahP $CKPT_DIR/policy/tokenizer ${CKPT_DIR}-hf/
```
