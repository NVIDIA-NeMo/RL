#!/usr/bin/env python3
"""Run one real CUDA Graph-enabled generation with vLLM 0.24."""

from __future__ import annotations

import json
import os
import time

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ.get("ENGINE_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    started_at = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
    )
    initialized_at = time.perf_counter()
    outputs = llm.generate(
        ["Return only the number that is the sum of 20 and 22."],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    finished_at = time.perf_counter()
    output = outputs[0].outputs[0]
    assert output.token_ids, "vLLM returned no generated tokens"

    print(
        json.dumps(
            {
                "cuda_graph_enabled": True,
                "generated_text": output.text,
                "generated_tokens": len(output.token_ids),
                "generation_seconds": finished_at - initialized_at,
                "initialization_seconds": initialized_at - started_at,
                "model": model,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
