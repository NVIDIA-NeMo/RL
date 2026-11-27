# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare CPU input construction with full-history capture splice and TITO.

This benchmark excludes history lookup, storage, native message parsing, GPU
inference and sandbox execution. It checks token equality before timing and
alternates measurement order. Report its results as component timings only.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.tito import Qwen3IncrementalTokenizer, bridge_suffix


def benchmark(
    tokenizer: Any, *, history_repeats: int, iterations: int, enable_thinking: bool
) -> dict[str, Any]:
    """Measure equivalent input construction for a fixed synthetic SWE trace."""
    adapter = Qwen3IncrementalTokenizer()
    adapter.qualify_template(tokenizer.get_chat_template())
    history = [
        {
            "role": "user",
            "content": "Fix the failing test.\n"
            + "def check(x): assert x == 1\n" * history_repeats,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pytest -q"}',
                    },
                }
            ],
        },
    ]
    suffix = [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "1 failed, 2 passed. Inspect check().",
        }
    ]

    def render(messages: list[dict[str, Any]], *, generation: bool) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=enable_thinking,
        )

    prefix = tokenizer.encode(
        render(history, generation=False).removesuffix("\n"), add_special_tokens=False
    )
    adapter.validate_prefix(tokenizer, prefix)

    def full_splice() -> list[int]:
        complete = tokenizer.encode(
            render(history + suffix, generation=True), add_special_tokens=False
        )
        anchor = tokenizer.encode(
            render(history, generation=False), add_special_tokens=False
        )
        return splice_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=prefix,
            template_prefix_token_ids=anchor,
            template_token_ids=complete,
        ).token_ids

    def incremental() -> list[int]:
        adapter.qualify_template(tokenizer.get_chat_template())
        adapter.validate_prefix(tokenizer, prefix)
        return prefix + adapter.encode_suffix(
            tokenizer, suffix, enable_thinking=enable_thinking
        )

    def bridge() -> list[int]:
        complete = tokenizer.encode(
            render(history + suffix, generation=True), add_special_tokens=False
        )
        return prefix + bridge_suffix(
            parent_token_ids=prefix,
            rendered_token_ids=complete,
            turn_close_token_ids={tokenizer.eos_token_id},
        )

    expected = full_splice()
    if incremental() != expected:
        raise ValueError("Incremental tokens differ from the existing capture splice")
    if bridge() != expected:
        raise ValueError("Bridged tokens differ from the existing capture splice")
    operations = {
        "full_splice": full_splice,
        "bridge": bridge,
        "incremental": incremental,
    }
    for _ in range(3):
        for operation in operations.values():
            operation()
    elapsed: dict[str, list[float]] = {name: [] for name in operations}
    for repeat in range(iterations):
        order = list(operations) if repeat % 2 == 0 else list(reversed(operations))
        for name in order:
            started = time.perf_counter()
            actual = operations[name]()
            elapsed[name].append(time.perf_counter() - started)
            if actual != expected:
                raise ValueError(f"{name} changed tokens during measurement")
    measurements = {
        name: {
            "median_ms": statistics.median(samples) * 1000,
            "min_ms": min(samples) * 1000,
            "max_ms": max(samples) * 1000,
            "samples_ms": [sample * 1000 for sample in samples],
        }
        for name, samples in elapsed.items()
    }
    return {
        "prefix_tokens": len(prefix),
        "suffix_tokens": len(expected) - len(prefix),
        "iterations": iterations,
        "token_equal": True,
        "measurements": measurements,
        "component_speedup": statistics.median(elapsed["full_splice"])
        / statistics.median(elapsed["incremental"]),
        "bridge_speedup": statistics.median(elapsed["full_splice"])
        / statistics.median(elapsed["bridge"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--history-repeats", type=int, nargs="+", default=[64, 512, 2048]
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or any(count < 1 for count in args.history_repeats):
        parser.error("iterations and history-repeats must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = [
        benchmark(
            tokenizer,
            history_repeats=count,
            iterations=args.iterations,
            enable_thinking=args.enable_thinking,
        )
        for count in args.history_repeats
    ]
    report = {
        "scope": "CPU input construction only; not rollout throughput",
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        print(
            f"prefix={case['prefix_tokens']} suffix={case['suffix_tokens']} "
            f"component_speedup={case['component_speedup']:.2f}x "
            f"bridge_speedup={case['bridge_speedup']:.2f}x token_equal=True"
        )


if __name__ == "__main__":
    main()
