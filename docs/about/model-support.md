# Model Support

## Broad coverage for 🤗Hugging Face models via [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)

NeMo-RL support 🤗Hugging Face models from the following classes
- LLMs ([AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM))
- VLMs ([AutoModelForImageTextToText](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForImageTextToText))

for model sizes under 70B at up to 32k sequence length.

## Optimal acceleration for top models via [NeMo Megatron-bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) provides acceleration [recipes](https://github.com/NVIDIA-NeMo/RL/tree/main/examples/configs/recipes) for the below models. Users can also leverage the on-line checkpoint conversion (i.e the "bridge") by directly inputting a 🤗Hugging Face checkpoint. 

**LLMs**:

- **Qwen**: Qwen2.5-1.5B/7B/32B, Qwen3-1.5B/8B/32B, Qwen3-30B-A3B, Qwen3-235B-A22B
- **Llama**: Llama 3.1/3.3-8B, Llama 3.1/3.3-70B, Llama 3.2-1B
- **Deepseek**: Deepseek-V3/R1-671B
- **Mistral**: Mistral-NeMo-12B
- **Moonlight-16B-A3B**
- **Gemma**: Gemma-3-1B/27B
- **GPT-OSS**: GPT-OSS-20B/120B
- **NeMotron**: Llama-Nemotron-Super-49B, Nemotron-nano-v2-12B, Nemotron-Nano-v3-30A3B

**VLMs**:

- **Qwen**: Qwen2.5VL-3B

In addition, please refer to our [performance page](https://docs.nvidia.com/nemo/rl/latest/about/performance-summary.html) for benchmarks and full reproducible yaml recipe configs.
