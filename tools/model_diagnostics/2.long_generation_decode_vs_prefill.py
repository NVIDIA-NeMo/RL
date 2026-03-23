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

# Checks that vLLM decode-time logprobs match a subsequent full-sequence
# prefill pass on the same tokens.  Mamba-MoE hybrid models (e.g. Nemotron-H,
# Nano v3) are susceptible to a chunked-prefill bug where the Mamba SSM state
# is corrupted at chunk boundaries, causing divergence at the first decode
# token.  Use --prompts arc to reproduce that failure mode with the validated
# long ARC-AGI prompts (~3 000 tokens) from the TME investigation.

import argparse

import torch
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# ARC-AGI prompts validated against Nano v3 TME divergence investigation.
# PROMPT_BROKEN tokenises to ~3 053 tokens and triggers the chunked-prefill
# Mamba-state corruption bug (TME > 300 in affected vLLM builds).
# PROMPT_OK is shorter and stays healthy (per-seq TME ≈ 1.007).
# Source: tme-reproduce/vllm_decode_debug.py
# ---------------------------------------------------------------------------

_ARC_SYSTEM = (
    "Find the common rule that maps an input grid to an output grid, "
    "given the examples below.\n"
    "After reasoning you must provide only the output and nothing else.\n"
    "Output format: \\boxed{solution} where solution is an array of rows "
    "separated by newlines, values by spaces."
)

_PROMPT_BROKEN = """\
Please solve this ARC-AGI problem:

Train Example 1:

Input:
3 4 8 9 3 8 4 2 9 6 0 3 9 7 3 9 9 8 1
9 1 1 1 1 1 1 1 1 7 0 1 2 2 2 2 2 2 2
4 1 1 1 1 1 1 1 1 3 0 9 2 2 2 2 2 2 3
4 1 1 1 1 1 1 1 1 9 0 9 2 2 2 2 2 2 4
6 1 1 1 1 1 1 1 1 1 0 7 2 2 2 2 2 2 5
3 1 1 1 1 1 1 1 1 3 0 5 2 2 2 2 2 2 2
8 1 1 1 1 1 1 1 1 5 0 8 2 2 2 2 2 2 3
6 1 1 1 1 1 1 1 1 2 0 9 2 2 2 2 2 2 4
1 1 1 1 1 1 1 1 1 8 0 2 2 2 2 2 2 2 5
2 1 3 5 1 5 8 9 1 7 0 6 4 4 3 9 1 4 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
8 1 8 2 5 5 8 1 9 4 0 2 9 9 1 1 2 1 9
9 8 8 8 8 8 8 8 8 3 0 7 4 4 4 4 4 4 5
4 8 8 8 8 8 8 8 8 2 0 8 4 4 4 4 4 4 8
6 8 8 8 8 8 8 8 8 8 0 6 4 4 4 4 4 4 9
1 8 8 8 8 8 8 8 8 3 0 6 4 4 4 4 4 4 1
9 8 8 8 8 8 8 8 8 9 0 9 4 4 4 4 4 4 7
1 8 8 8 8 8 8 8 8 1 0 8 4 4 4 4 4 4 9
7 2 4 1 5 3 2 4 1 4 0 4 3 5 6 6 5 2 8

Output:
1 0 2
0 0 0
8 0 4

Train Example 2:

Input:
4 7 4 1 3 2 5 1 1 5 9 4 9 9 9 7 7 1 7
9 2 2 2 2 2 7 1 9 0 0 0 0 0 0 0 0 0 4
8 2 2 2 2 2 1 1 5 0 0 0 0 0 0 0 0 0 2
5 2 2 2 2 2 2 1 7 0 0 0 0 0 0 0 0 0 3
4 2 2 2 2 2 9 1 4 0 0 0 0 0 0 0 0 0 2
7 2 2 2 2 2 5 1 4 0 0 0 0 0 0 0 0 0 2
7 2 2 2 2 2 7 1 9 0 0 0 0 0 0 0 0 0 1
7 2 2 2 2 2 6 1 4 0 0 0 0 0 0 0 0 0 5
8 2 2 2 2 2 4 1 6 0 0 0 0 0 0 0 0 0 4
7 1 1 1 1 6 4 1 7 1 7 7 9 6 5 6 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
6 4 2 7 4 5 6 1 5 5 6 6 4 6 9 2 7 9 4
4 4 4 4 4 4 4 1 6 8 8 8 8 8 8 8 8 8 4
2 4 4 4 4 4 5 1 9 8 8 8 8 8 8 8 8 8 8
2 4 4 4 4 4 5 1 7 8 8 8 8 8 8 8 8 8 8
9 4 4 4 4 4 6 1 9 8 8 8 8 8 8 8 8 8 7
9 4 4 4 4 4 1 1 9 8 8 8 8 8 8 8 8 8 8
3 4 4 4 4 4 4 1 9 8 8 8 8 8 8 8 8 8 8
7 5 7 8 8 9 1 1 5 9 8 9 7 8 5 8 9 6 8

Output:
2 1 0
1 1 1
4 1 8

Test Input:
6 8 1 5 2 1 8 9 3 1 3 5 3 6 2 7 7 6 6
7 3 3 3 3 6 8 8 2 2 2 2 2 2 2 2 2 2 3
6 3 3 3 3 9 8 8 2 2 2 2 2 2 2 2 2 2 1
8 3 3 3 3 4 8 8 2 2 2 2 2 2 2 2 2 2 6
2 3 3 3 3 8 8 5 2 2 2 2 2 2 2 2 2 2 1
3 2 2 3 4 2 8 6 4 4 3 2 6 3 3 6 9 3 2
8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
1 3 8 4 3 4 8 7 9 4 6 1 6 5 8 6 8 8 9
1 4 4 4 4 5 8 5 1 1 1 1 1 1 1 1 1 1 9
8 4 4 4 4 8 8 8 1 1 1 1 1 1 1 1 1 1 1
3 4 4 4 4 9 8 4 1 1 1 1 1 1 1 1 1 1 8
3 4 4 4 4 3 8 2 1 1 1 1 1 1 1 1 1 1 7
6 4 4 4 4 3 8 9 1 1 1 1 1 1 1 1 1 1 2
5 4 4 4 4 7 8 2 1 1 1 1 1 1 1 1 1 1 5
1 4 4 4 4 1 8 2 1 1 1 1 1 1 1 1 1 1 1
9 4 4 4 4 4 8 4 1 1 1 1 1 1 1 1 1 1 8
4 4 4 4 4 6 8 5 1 1 1 1 1 1 1 1 1 1 7
7 4 4 4 4 8 8 7 1 1 1 1 1 1 1 1 1 1 3
7 1 4 5 2 8 8 1 9 7 3 6 6 4 8 8 6 7 1

"""

_PROMPT_OK = """\
Please solve this ARC-AGI problem:

Train Example 1:

Input:
0 0 0 4 0 6 0 0 5 0 7 0 0 0 0 6
7 3 7 0 1 0 9 0 0 4 8 7 9 0 0 0
0 0 0 3 0 0 6 1 2 0 2 3 7 0 0 0
0 6 4 0 5 0 0 9 9 0 4 0 1 0 0 3
1 0 0 1 9 9 3 8 9 0 7 1 5 5 5 0
0 2 4 0 6 0 0 0 9 0 0 4 0 5 7 7
5 5 0 9 4 0 0 8 0 0 9 0 0 6 4 0
2 8 0 2 7 2 0 0 0 4 0 0 0 1 0 0
0 0 1 0 3 1 0 0 1 0 0 6 4 0 1 0
0 0 0 7 0 7 0 0 0 0 0 4 0 2 1 1
0 5 0 0 5 0 0 0 0 0 2 6 0 7 0 0
7 0 3 6 5 0 3 0 0 0 1 7 0 9 4 0
6 0 0 0 2 0 9 1 4 0 0 8 0 5 0 4
2 0 0 9 1 0 0 0 2 3 0 0 0 0 6 0
7 5 0 0 3 3 2 0 9 0 0 5 2 0 8 0
2 4 8 0 0 5 3 0 9 3 9 0 4 5 0 0

Output:
0 0 0 4 0 6 2 2 5 0 7 0 0 0 0 6
7 3 7 0 1 0 9 2 2 4 8 7 9 0 0 0
0 0 0 3 0 0 6 1 2 2 2 3 7 0 0 0
0 6 4 0 5 0 0 9 9 2 4 0 1 0 0 3
1 2 2 1 9 9 3 8 9 2 7 1 5 5 5 0
2 2 4 0 6 2 2 2 9 2 2 4 2 5 7 7
5 5 2 9 4 2 2 8 2 2 9 2 2 6 4 0
2 8 2 2 7 2 2 2 2 4 2 2 2 1 0 0
2 2 1 2 3 1 2 2 1 2 2 6 4 2 1 0
2 2 2 7 0 7 2 2 2 2 2 4 2 2 1 1
2 5 2 2 5 2 2 2 2 2 2 6 2 7 0 0
7 2 3 6 5 2 3 2 2 2 1 7 2 9 4 0
6 2 2 2 2 2 9 1 4 2 2 8 2 5 0 4
2 2 2 9 1 2 2 2 2 3 2 2 2 2 6 0
7 5 2 2 3 3 2 2 9 2 2 5 2 2 8 0
2 4 8 2 2 5 3 2 9 3 9 0 4 5 0 0

Train Example 2:

Input:
0 0 9 0 5 7 6 6 6 0 9 2 0 0 0
8 0 1 1 5 0 0 0 8 5 0 0 0 6 0
4 5 9 0 2 0 7 4 0 0 0 4 0 4 0
7 0 1 0 1 0 0 2 7 2 7 5 2 1 9
3 3 3 8 0 0 7 1 0 1 0 2 8 0 0
3 7 0 1 0 9 0 0 1 0 4 0 0 3 7
0 0 1 3 4 8 0 1 0 0 6 0 7 0 8
0 4 0 0 9 0 0 0 2 0 9 0 0 2 0
8 8 2 0 9 0 0 7 0 4 7 0 0 0 0
0 0 0 0 0 4 0 0 0 5 0 1 0 9 4
0 0 1 2 1 5 0 3 0 2 0 6 0 0 4
3 9 0 0 1 6 2 0 5 0 0 7 1 0 0
0 0 0 0 5 7 8 2 8 8 5 0 0 0 6
0 6 0 1 0 3 5 5 0 0 0 0 5 1 0
0 5 4 4 0 7 4 0 0 0 4 0 0 0 0

Output:
0 0 9 0 5 7 6 6 6 0 9 2 2 2 2
8 0 1 1 5 2 2 2 8 5 2 2 2 6 2
4 5 9 2 2 2 7 4 2 2 2 4 2 4 2
7 0 1 2 1 2 2 2 7 2 7 5 2 1 9
3 3 3 8 2 2 7 1 0 1 2 2 8 0 0
3 7 0 1 2 9 2 2 1 2 4 2 2 3 7
0 0 1 3 4 8 2 1 2 2 6 2 7 2 8
0 4 2 2 9 2 2 2 2 2 9 2 2 2 2
8 8 2 2 9 2 2 7 2 4 7 2 2 2 2
2 2 2 2 2 4 2 2 2 5 2 1 2 9 4
2 2 1 2 1 5 2 3 2 2 2 6 2 2 4
3 9 2 2 1 6 2 2 5 2 2 7 1 2 2
2 2 2 2 5 7 8 2 8 8 5 2 2 2 6
2 6 2 1 0 3 5 5 2 2 2 2 5 1 2
2 5 4 4 0 7 4 2 2 2 4 2 2 2 2

Train Example 3:

Input:
3 1 1 0 1 0 0 7 2 2 2 0 0 7 0 6
2 2 8 8 1 0 8 4 6 8 1 6 0 4 9 4
4 0 1 4 4 6 0 5 0 0 0 6 6 4 6 4
0 9 7 0 0 0 0 8 0 2 0 4 0 1 0 4
4 0 9 4 0 3 0 0 0 5 0 3 8 0 8 7
0 0 0 0 0 0 6 8 0 7 0 7 0 1 6 3
7 7 0 0 8 2 0 0 9 6 0 0 0 8 1 6
3 0 3 4 0 2 0 0 0 5 0 3 8 0 8 0
0 7 0 0 6 7 7 6 5 4 8 5 0 3 0 0
3 6 0 0 0 4 0 7 5 0 0 3 2 0 0 0
7 6 8 1 8 0 7 5 1 2 4 5 9 4 3 3
4 7 4 7 8 9 3 8 0 9 0 0 9 2 0 0
5 0 0 0 7 2 4 0 8 8 0 9 0 9 2 1
0 1 0 0 0 0 2 0 0 0 7 0 1 1 6 7
0 6 0 4 8 9 2 0 6 5 2 4 3 0 9 3
0 5 0 2 4 7 0 5 5 4 0 5 0 0 8 0

Output:
3 1 1 0 1 0 0 7 2 2 2 2 2 7 0 6
2 2 8 8 1 0 8 4 6 8 1 6 2 4 9 4
4 2 1 4 4 6 2 5 2 2 2 6 6 4 6 4
0 9 7 2 2 2 2 8 2 2 2 4 0 1 0 4
4 2 9 4 2 3 2 2 2 5 2 3 8 0 8 7
2 2 2 2 2 2 6 8 2 7 2 7 2 1 6 3
7 7 2 2 8 2 2 2 9 6 2 2 2 8 1 6
3 0 3 4 2 2 2 2 2 5 2 3 8 0 8 2
0 7 0 0 6 7 7 6 5 4 8 5 2 3 2 2
3 6 0 0 0 4 0 7 5 2 2 3 2 2 2 2
7 6 8 1 8 0 7 5 1 2 4 5 9 4 3 3
4 7 4 7 8 9 3 8 0 9 0 0 9 2 2 2
5 2 2 2 7 2 4 2 8 8 0 9 0 9 2 1
0 1 2 2 2 2 2 2 2 2 7 0 1 1 6 7
0 6 2 4 8 9 2 2 6 5 2 4 3 0 9 3
0 5 2 2 4 7 2 5 5 4 2 5 0 0 8 0


Test Input:
5 0 0 0 0 9 4 0 5 0 9 0 5
2 0 0 7 0 0 0 0 0 0 0 5 6
5 0 1 5 8 0 0 6 3 0 4 0 4
0 0 4 7 0 7 0 2 0 0 0 1 2
8 7 0 2 0 8 5 0 0 1 6 3 9
0 4 0 0 0 0 0 0 3 2 4 0 0
5 0 0 9 0 0 0 3 6 3 0 4 1
0 0 0 4 3 6 0 0 3 0 0 6 0
5 5 0 0 0 0 6 0 0 6 0 0 0
4 0 0 4 0 7 6 0 0 0 0 5 0
1 0 0 7 0 3 5 5 4 0 6 3 8
0 0 0 0 0 0 0 3 9 0 0 0 4
0 0 0 3 6 0 6 0 0 0 0 0 8

"""


def _arc_prompts():
    """Return the two validated ARC-AGI prompts as plain strings.

    BROKEN is ~3 053 tokens and triggers the chunked-prefill Mamba-state
    corruption bug (TME > 300 in affected vLLM builds).  OK is shorter and
    stays healthy (per-seq TME ≈ 1.007).
    """
    broken = _ARC_SYSTEM + "\n\n" + _PROMPT_BROKEN
    ok = _ARC_SYSTEM + "\n\n" + _PROMPT_OK
    return [broken, ok]


def extract_logprobs(logprobs):
    output = []
    for lp in logprobs:
        if lp is not None:
            output.append(list(lp.values())[0].logprob)
    return output


def calculate_error(a, b) -> float:
    return torch.exp(torch.abs(a - b)).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, nargs="?", default="nvidia/Nemotron-H-8B-Base-8K"
    )
    parser.add_argument(
        "--prompts",
        choices=["short", "arc"],
        default="short",
        help=(
            "short: four brief prompts (original default, fast); "
            "arc: two long ARC-AGI prompts (~3 000 tokens) that reproduce "
            "the chunked-prefill Mamba-state corruption bug"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per sequence (default 128)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (default 1)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=2,
        help=(
            "How many times to repeat the prompt list (default 2). "
            "Use --num-batches 16 with --prompts arc for a 32-sequence investigation run."
        ),
    )
    args = parser.parse_args()

    seed = 0

    gen_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        prompt_logprobs=0,
        logprobs=0,
        seed=seed,
    )

    # Examples as of 0.9.1
    # model="meta-llama/Meta-Llama-3-8B",        # pass
    # model="nvidia/Nemotron-H-8B-Base-8K",       # fail
    # model="ibm-ai-platform/Bamba-9B-v1",        # pass

    # Examples < 0.17.0
    # model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",   # fail in arc prompts

    # Examples >= 0.17.0
    # model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",   # pass
    llm_kwargs = dict(
        model=args.model,
        enforce_eager=False,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=seed,
        gpu_memory_utilization=0.8,
    )
    llm = LLM(**llm_kwargs)

    if args.prompts == "arc":
        prompts = _arc_prompts()
    else:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

    # ── Step 1: generate (decode-time logprobs) ──────────────────────────────
    outputs = llm.generate(prompts * args.num_batches, gen_params)

    # Collect all full sequences and decode logprobs before scoring.
    sequences = []
    decode_lps = []
    for output in outputs:
        sequence = output.prompt_token_ids + list(output.outputs[0].token_ids)
        prompt_logprobs = extract_logprobs(output.prompt_logprobs)
        logprobs = extract_logprobs(output.outputs[0].logprobs)
        sequences.append(sequence)
        decode_lps.append(torch.tensor(prompt_logprobs + logprobs))

    # ── Step 2: score all sequences in one batched prefill pass ──────────────
    score_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    score_outputs = llm.generate(
        [{"prompt_token_ids": seq} for seq in sequences],
        score_params,
    )

    # ── Step 3: compare and assert ───────────────────────────────────────────
    prefill_lps = [
        torch.tensor(extract_logprobs(score.prompt_logprobs)) for score in score_outputs
    ]

    all_decode = torch.cat(decode_lps)
    all_prefill = torch.cat(prefill_lps)

    lp_error = calculate_error(all_decode, all_prefill)
    max_abs_error = torch.abs(all_decode - all_prefill).max().item()
    print(
        f"Processed {len(sequences)} sequences ({len(all_decode)} tokens total) "
        f"with lp error {lp_error} and max abs error {max_abs_error}"
    )
    assert lp_error < 1.05, f"lp error exceeds threshold 1.05: {lp_error}"

    print(f"[{args.model}] ALL GOOD!")


if __name__ == "__main__":
    main()
