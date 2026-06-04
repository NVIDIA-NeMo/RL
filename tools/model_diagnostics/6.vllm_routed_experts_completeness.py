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
"""Check whether vLLM returns routed experts for every generated route.

This is a standalone diagnostic for MoE models and vLLM builds that support
``enable_return_routed_experts``. It is intended to reproduce rare omissions
seen with prefix caching plus chunked prefill.
"""

import argparse
import json
from typing import Any


def _parse_extra_kwarg(raw: str) -> tuple[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"Expected --llm-kwarg KEY=VALUE, got {raw!r}"
        )
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _route_count(route_tensor: Any) -> int:
    if route_tensor is None:
        return 0
    return len(route_tensor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-repeat", type=int, default=128)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--llm-kwarg",
        action="append",
        default=[],
        type=_parse_extra_kwarg,
        help="Extra vLLM LLM kwarg as KEY=VALUE. VALUE may be JSON.",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_return_routed_experts": True,
    }
    if args.enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    for key, value in args.llm_kwarg:
        llm_kwargs[key] = value

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
    )

    shared_prefix = "Solve the following math problem carefully. " * args.prompt_repeat
    prompts = [
        f"{shared_prefix}\nProblem {idx}: What is {idx} plus {idx + 1}?"
        for idx in range(args.num_prompts)
    ]
    outputs = llm.generate(prompts, sampling_params)

    failures = []
    for idx, request_output in enumerate(outputs):
        completion_output = request_output.outputs[0]
        prompt_routes = _route_count(
            getattr(request_output, "prompt_routed_experts", None)
        )
        completion_routes = _route_count(
            getattr(completion_output, "routed_experts", None)
        )
        actual_routes = prompt_routes + completion_routes
        valid_length = len(request_output.prompt_token_ids) + len(
            completion_output.token_ids
        )
        expected_routes = max(valid_length - 1, 0)
        max_allowed_routes = expected_routes + 1

        if actual_routes < expected_routes or actual_routes > max_allowed_routes:
            failures.append(
                {
                    "sample": idx,
                    "prompt_routes": prompt_routes,
                    "completion_routes": completion_routes,
                    "actual_routes": actual_routes,
                    "expected_routes": expected_routes,
                    "max_allowed_routes": max_allowed_routes,
                    "prompt_tokens": len(request_output.prompt_token_ids),
                    "completion_tokens": len(completion_output.token_ids),
                    "num_cached_tokens": getattr(
                        request_output, "num_cached_tokens", None
                    ),
                }
            )

    summary = {
        "model": args.model,
        "num_outputs": len(outputs),
        "num_failures": len(failures),
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "failures": failures[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
