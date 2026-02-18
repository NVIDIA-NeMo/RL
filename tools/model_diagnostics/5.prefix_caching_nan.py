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
"""Prefix caching NaN reproducer.

Tests that prefix caching doesn't produce NaN logprobs when prior generation
is rolled back into the prompt (the standard RL / multi-turn pattern).

Known failure: vLLM >= 0.14 may return token_id=0 (<unk>) with logprob=nan
for every token after the first on the second request.

Usage:
    python 5.prefix_caching_nan.py
    python 5.prefix_caching_nan.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import math

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TP = 2
MAX_TOKENS = 2048
MAX_MODEL_LEN = 32768
COUNT_UP_TO = 3000

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=MODEL)
parser.add_argument("--tp", type=int, default=TP)
args = parser.parse_args()

import vllm
from vllm import LLM, SamplingParams

print(f"vLLM version: {vllm.__version__}")

numbers = " ".join(str(i) for i in range(1, COUNT_UP_TO + 1))
prompt = (
    "You are a counting assistant. Output ONLY numbers separated by spaces.\n\n"
    f"User: Continue counting: {numbers} "
)

llm = LLM(
    model=args.model,
    tensor_parallel_size=args.tp,
    enable_prefix_caching=True,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1)

# Iteration 1: initial generation (builds the prefix cache)
print(f"\nIteration 1 — prompt length: {len(prompt)} chars")
out1 = llm.generate([prompt], sampling_params)[0].outputs[0]
print(f"  tokens: {len(out1.token_ids)}, finish_reason: {out1.finish_reason}")
print(f"  text (first 100): {out1.text[:100]!r}")

# Iteration 2: extend prompt with prior output (triggers prefix cache reuse)
prompt += out1.text
print(f"\nIteration 2 — prompt length: {len(prompt)} chars")
out2 = llm.generate([prompt], sampling_params)[0].outputs[0]
print(f"  tokens: {len(out2.token_ids)}, finish_reason: {out2.finish_reason}")
print(f"  text (first 100): {out2.text[:100]!r}")

# Check for NaN logprobs
nan_count = 0
if out2.logprobs:
    for step in out2.logprobs:
        if step is None:
            continue
        for _tid, lp_obj in step.items():
            lp = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
            if isinstance(lp, float) and math.isnan(lp):
                nan_count += 1
            break

if nan_count > 0:
    print(f"\n  Sample logprobs from iteration 2:")
    for idx in [0, 1, 2, len(out2.logprobs) - 1]:
        if idx < len(out2.logprobs) and out2.logprobs[idx] is not None:
            for tid, lp_obj in out2.logprobs[idx].items():
                lp = lp_obj.logprob if hasattr(lp_obj, "logprob") else lp_obj
                decoded = lp_obj.decoded_token if hasattr(lp_obj, "decoded_token") else "?"
                print(f"    token[{idx}] id={tid}: logprob={lp} decoded={decoded!r}")
                break

assert nan_count == 0, (
    f"FAIL: {nan_count}/{len(out2.token_ids)} logprobs are NaN on iteration 2 "
    f"(prefix caching is broken in vLLM {vllm.__version__})"
)
print(f"\n[{args.model}] ALL GOOD!")
