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

"""Per-request response limits using the standard training-context cap."""


def _context_capped_max_new_tokens(
    *, configured_max_new_tokens: int, input_length: int, max_model_len: int
) -> int:
    """Cap generation so the training prompt and response fit the context."""
    remaining_context = max_model_len - input_length
    if remaining_context <= 0:
        raise ValueError(
            "Cannot generate from an input whose training length exhausts the "
            f"model context: input_length={input_length}, max_model_len={max_model_len}."
        )
    return min(configured_max_new_tokens, remaining_context)


def per_request_output_limits(
    output_caps: list[int | None],
    input_lengths: list[int],
    *,
    configured_max_new_tokens: int,
    max_model_len: int,
) -> list[int]:
    """Bound explicit row caps; null rows preserve the configured generation limit."""
    if len(output_caps) != len(input_lengths):
        raise ValueError("max_output_tokens must have one value per generation request")
    if any(
        cap is not None and (type(cap) is not int or cap < 1) for cap in output_caps
    ):
        raise ValueError("max_output_tokens must contain positive integers or None")
    return [
        configured_max_new_tokens
        if cap is None
        else _context_capped_max_new_tokens(
            configured_max_new_tokens=min(configured_max_new_tokens, cap),
            input_length=length,
            max_model_len=max_model_len,
        )
        for cap, length in zip(output_caps, input_lengths)
    ]
