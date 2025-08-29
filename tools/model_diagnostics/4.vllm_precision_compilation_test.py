# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import os
from contextlib import contextmanager

import numpy as np
import torch
from vllm import LLM, SamplingParams


@contextmanager
def environment(env_vars):
    """Context manager to temporarily set environment variables.

    Args:
        env_vars (dict): Dictionary of environment variable names and values to set

    Example:
        with environment({"CUDA_VISIBLE_DEVICES": "0"}):
            # Code here runs with CUDA_VISIBLE_DEVICES=0
            pass
        # Environment variables are restored here
    """
    # Store original values
    original_values = {}
    for key in env_vars:
        if key in os.environ:
            original_values[key] = os.environ[key]
        else:
            original_values[key] = None

    # Set new values
    for key, value in env_vars.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = str(value)

    try:
        yield
    finally:
        # Restore original values
        for key, value in original_values.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value


def extract_logprobs(logprobs):
    output = []
    for lp in logprobs:
        if lp is not None:
            output.append(list(lp.values())[0].logprob)
    return output


def pad_logprobs_list(logprobs_list):
    """Pad a list of logprobs lists into a numpy array.

    Args:
        logprobs_list (list): List of lists, where each inner list contains logprobs

    Returns:
        np.ndarray: Padded numpy array with shape (num_sequences, max_length)
    """
    if not logprobs_list:
        return np.array([])

    max_length = max(len(lp) for lp in logprobs_list)
    padded_array = np.full((len(logprobs_list), max_length), np.nan, dtype=np.float32)

    for i, lp in enumerate(logprobs_list):
        padded_array[i, : len(lp)] = lp

    return padded_array


def assert_logprobs_close(actual, expected, test_name, atol=1e-3, rtol=1e-3):
    """Assert that two logprobs arrays are close to each other.

    Args:
        actual: The actual logprobs array
        expected: The expected logprobs array
        test_name (str): Name of the test for error messages
        atol (float): Absolute tolerance
        rtol (float): Relative tolerance
    """
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        print(
            f"{test_name}: PASSED - Arrays are close within tolerance (atol={atol}, rtol={rtol})"
        )
    except AssertionError as e:
        print("=" * 100)
        print(f"{test_name}: FAILED - Arrays are different")
        print(f"  Detailed error: {e}")
        print("=" * 100)


def get_logprobs(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    prompt_lps = []
    generation_lps = []

    # Collect all logprobs
    for output in outputs:
        prompt_logprobs = extract_logprobs(output.prompt_logprobs)
        generation_logprobs = extract_logprobs(output.outputs[0].logprobs)
        prompt_lps.append(prompt_logprobs)
        generation_lps.append(generation_logprobs)

    # Use common padding function
    padded_prompt_lps = pad_logprobs_list(prompt_lps)
    padded_generation_lps = pad_logprobs_list(generation_lps)

    return padded_prompt_lps, padded_generation_lps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    args = parser.parse_args()
    seed = 0

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=8192,
        prompt_logprobs=0,
        logprobs=0,
        seed=seed,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "<｜begin▁of▁sentence｜><｜User｜>Think step-by-step to solve the following problem. Output your answer inside of \\\\boxed{} tags.:\n$A B C D$ is a rectangle with $A B=20$ and $B C=3$. A circle with radius 5, centered at the midpoint of $D C$, meets the rectangle at four points: $W, X, Y$, and $Z$. Find the area of quadrilateral $W X Y Z$.\n\nLet's think step-by-step<｜Assistant｜><think>\n",
    ]

    common_llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    }

    eager_prompt_lps, eager_generation_lps = get_logprobs(
        LLM(enforce_eager=True, **common_llm_kwargs),  # eager mode for ground truth lps
        prompts,
        sampling_params,
    )

    torch.cuda.empty_cache()

    cuda_graph_prompt_lps, cuda_graph_generation_lps = get_logprobs(
        LLM(enforce_eager=False, **common_llm_kwargs),  # cuda graph mode
        prompts,
        sampling_params,
    )

    assert_logprobs_close(
        cuda_graph_prompt_lps,
        eager_prompt_lps,
        "Eager and cuda graph mode lps (prompt lps)",
    )
    assert_logprobs_close(
        cuda_graph_generation_lps,
        eager_generation_lps,
        "Eager and cuda graph mode lps (generation lps)",
    )

    torch.cuda.empty_cache()

    with environment(env_vars={"TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1"}):
        cuda_graph_prompt_lps_w_flag, cuda_graph_generation_lps_w_flag = get_logprobs(
            LLM(enforce_eager=False, **common_llm_kwargs),
            prompts,
            sampling_params,
        )

    assert_logprobs_close(
        cuda_graph_prompt_lps_w_flag,
        eager_prompt_lps,
        "Eager and cuda graph mode lps with torch inductor precision flag (prompt lps)",
    )
    assert_logprobs_close(
        cuda_graph_generation_lps_w_flag,
        eager_generation_lps,
        "Eager and cuda graph mode lps with torch inductor precision flag (generation lps)",
    )

    torch.cuda.empty_cache()

    (
        cuda_graph_prompt_lps_w_inductor_disabled,
        cuda_graph_generation_lps_w_inductor_disabled,
    ) = get_logprobs(
        LLM(
            enforce_eager=False,
            compilation_config={"use_inductor": False},
            **common_llm_kwargs,
        ),
        prompts,
        sampling_params,
    )

    assert_logprobs_close(
        cuda_graph_prompt_lps_w_inductor_disabled,
        eager_prompt_lps,
        "Eager and cuda graph mode lps with use_inductor disabled (prompt lps)",
    )
    assert_logprobs_close(
        cuda_graph_generation_lps_w_inductor_disabled,
        eager_generation_lps,
        "Eager and cuda graph mode lps with use_inductor disabled (generation lps)",
    )


if __name__ == "__main__":
    main()
