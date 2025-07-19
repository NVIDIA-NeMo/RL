# DeepSeek-V3

## Create BF16 Hugging Face checkpoint

(adapted from https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html)

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
python fp8_cast_bf16.py --input-fp8-hf-path ../../DeepSeek-V3-FP8 --output-bf16-hf-path ../../DeepSeek-V3-BF16

# copy other files
cd ../..
cp DeepSeek-V3-FP8/{tokenizer_config.json,tokenizer.json,modeling_deepseek.py,configuration_deepseek.py} DeepSeek-V3-BF16/

# copy config.json, remove `quantization_config`, and set num_nextn_predict_layers to 0 (we currently do not support mtp):
jq 'del(.quantization_config) | .num_nextn_predict_layers=0' DeepSeek-V3-FP8/config.json > DeepSeek-V3-BF16/config.json
```