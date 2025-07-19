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

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, nargs="?", default="ibm-ai-platform/Bamba-9B-v1")
args = parser.parse_args()


from vllm import LLM, SamplingParams

llm = LLM(
    # Examples as of 0.9.0
    # model="facebook/opt-125m", # ok: 20
    # model="meta-llama/Llama-3.1-8B-Instruct", # ok: 20
    # model="meta-llama/Meta-Llama-3-8B", # ok: 20
    # model="nvidia/Nemotron-H-8B-Base-8K", # not good: 21
    # model="ibm-ai-platform/Bamba-9B-v1",  # not good: 21
    model=args.model,
    max_model_len=20,  # total tokens = prompt + generated
    trust_remote_code=True,
)
prompt = "Hello, this is a test prompt."
sampling_params = SamplingParams(max_tokens=20)
output = llm.generate([prompt], sampling_params)[0]

num_prompt_tokens = len(output.prompt_token_ids)
num_generated_tokens = len(output.outputs[0].token_ids)
num_total_tokens = num_prompt_tokens + num_generated_tokens

print(f"Prompt tokens: {num_prompt_tokens}")
print(f"Generated tokens: {num_generated_tokens}")
print(f"Total tokens: {num_total_tokens}")

assert num_total_tokens <= llm.llm_engine.model_config.max_model_len, (
    f"num_total_tokens={num_total_tokens} > max_model_len={llm.llm_engine.model_config.max_model_len} for model={args.model}, which should not happen."
)
print(f"[{args.model}] ALL GOOD!")
