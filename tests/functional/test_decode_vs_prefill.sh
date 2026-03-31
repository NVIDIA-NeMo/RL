uv run --extra vllm python tools/model_diagnostics/2.long_generation_decode_vs_prefill.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --prompts arc \
    --max-tokens 8192 \
    --num-batches 4 \
    --tensor-parallel-size 2 \
