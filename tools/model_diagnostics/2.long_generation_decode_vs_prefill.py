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

import torch
from vllm import LLM, SamplingParams


def extract_logprobs(logprobs):
    output = []
    for lp in logprobs:
        if lp is not None:
            output.append(list(lp.values())[0].logprob)
    return output


def calculate_error(a, b):
    return torch.exp(torch.abs(a - b)).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, nargs="?", default="nvidia/Nemotron-H-8B-Base-8K"
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

    # Examples as of 0.9.1
    # model="meta-llama/Meta-Llama-3-8B", # pass
    # model="nvidia/Nemotron-H-8B-Base-8K", # fail
    # model="ibm-ai-platform/Bamba-9B-v1", # pass
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        seed=seed,
    )

    num_batches = 2

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts * num_batches, sampling_params)

    for i, output in enumerate(outputs):
        sequence = output.prompt_token_ids + list(output.outputs[0].token_ids)
        prompt_logprobs = extract_logprobs(output.prompt_logprobs)
        logprobs = extract_logprobs(output.outputs[0].logprobs)
        decode_lp = prompt_logprobs + logprobs
        decode_lp = torch.tensor(decode_lp)

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1, prompt_logprobs=0
        )
        score = llm.generate({"prompt_token_ids": sequence}, sampling_params)

        prefill_lp = extract_logprobs(score[0].prompt_logprobs)
        prefill_lp = torch.tensor(prefill_lp)

        lp_error = calculate_error(decode_lp, prefill_lp)
        max_abs_error = torch.abs(decode_lp - prefill_lp).max().item()
        print(
            f"Processed sequence length {len(sequence)} with lp error {lp_error} and max abs error {max_abs_error}"
        )
        assert lp_error < 1.05, f"lp error is higher than expected (1.0636): {lp_error}"

    print(f"[{args.model}] ALL GOOD!")


if __name__ == "__main__":
    main()
