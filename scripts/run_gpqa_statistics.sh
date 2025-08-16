#!/bin/bash

uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-*-Base_high.txt \
    > logs/Qwen3-Base_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-sft-scp-16k-0.2x_step_*_high.txt \
    > logs/Qwen3-14B-Base-sft-scp-16k-0.2x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-sft-scp-16k-3x_step_*_high.txt \
    > logs/Qwen3-14B-Base-sft-scp-16k-3x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-scp-16k-128x32-1e-6-3x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-scp-16k-128x32-1e-6-3x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.025x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.025x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.05x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.05x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.125x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.125x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.25x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.25x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.5x-7.5x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-0.5x-7.5x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-1x-7x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-1x-7x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-2x-6x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-2x-6x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-4x-4x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-4x-4x_high_statistics.txt &
uv run python scripts/compute_statistics.py --tokenizer Qwen/Qwen3-14B logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-8x-1x_step_*_high.txt \
    > logs/Qwen3-14B-Base-grpo-from-sft-scp-16k-128x32-8x-1x_high_statistics.txt &