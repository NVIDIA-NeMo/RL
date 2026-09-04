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

"""Prompt construction helpers for length-scheduling experiments."""


def make_request_unique_prompts(
    prompt_lengths: list[int], common_token_id: int = 1000
) -> list[dict[str, list[int]]]:
    """Build fixed-length prompts whose first token is unique per request.

    A request-specific first token makes every prompt diverge before the first
    KV-cache block, preventing automatic prefix caching from sharing prompt KV
    state between requests while keeping the remaining token content fixed.
    """
    prompts = []
    for request_index, prompt_length in enumerate(prompt_lengths):
        if prompt_length < 1:
            raise ValueError("prompt lengths must be positive")
        prompt_token_ids = [common_token_id] * prompt_length
        prompt_token_ids[0] = common_token_id + request_index
        prompts.append({"prompt_token_ids": prompt_token_ids})
    return prompts
