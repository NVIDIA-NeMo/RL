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

"""Restore cases every serving path that resumes a cut call must reproduce.

Each case is one uninterrupted call, as token pieces, and how many of its
generated tokens a checkpoint cut saved. A serving path passes when the reply
it builds for the resumed call (text, the token IDs its parsers see, and usage)
equals the reply for the uninterrupted call. The native vLLM path runs them in
test_prefix_restore_cases.py; the Dynamo frontend reuses this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RestoreCase:
    name: str
    prompt: tuple[str, ...]
    # Every token the uninterrupted call generated, in order.
    output: tuple[str, ...]
    # How many of them the cut saved; the rest is the resumed call's tail.
    saved: int
    finish_reason: Literal["stop", "length"]
    # The last output token is a stop token (EOS or a stop_token_ids entry).
    stop_token_terminated: bool = False
    stop: tuple[str, ...] = ()
    stop_reason: str | int | None = None
    include_stop_str_in_output: bool = False
    min_tokens: int = 0


_PROMPT = ("<s>", " Q", ":")
_EOS = "</s>"

RESTORE_CASES = (
    RestoreCase(
        name="plain_text",
        prompt=_PROMPT,
        output=(" Hello", " world", "!", _EOS),
        saved=2,
        finish_reason="stop",
        stop_token_terminated=True,
    ),
    # Tool-call and reasoning parsers must see the saved tokens, not only the
    # tail, or the call's JSON or thinking block is cut in half.
    RestoreCase(
        name="tool_call_starts_in_the_saved_part",
        prompt=_PROMPT,
        output=("<tool_call>", "{", '"name"', ":", '"f"', "}", "</tool_call>", _EOS),
        saved=3,
        finish_reason="stop",
        stop_token_terminated=True,
    ),
    RestoreCase(
        name="reasoning_ends_in_the_saved_part",
        prompt=_PROMPT,
        output=("<think>", " x", "</think>", " Hello", _EOS),
        saved=3,
        finish_reason="stop",
        stop_token_terminated=True,
    ),
    RestoreCase(
        name="min_tokens_still_owed_after_the_cut",
        prompt=_PROMPT,
        output=(" Hello", " world", "!", _EOS),
        saved=1,
        finish_reason="stop",
        stop_token_terminated=True,
        min_tokens=3,
    ),
    # Finished at the cut: the resumed call is answered without decoding.
    RestoreCase(
        name="stop_string_in_the_saved_part",
        prompt=_PROMPT,
        output=(" answer", ":", " 4", "</", "answer", ">"),
        saved=6,
        finish_reason="stop",
        stop=("</answer>",),
        stop_reason="</answer>",
    ),
    RestoreCase(
        name="stop_string_kept_in_the_output",
        prompt=_PROMPT,
        output=(" answer", ":", " 4", "</", "answer", ">"),
        saved=6,
        finish_reason="stop",
        stop=("</answer>",),
        stop_reason="</answer>",
        include_stop_str_in_output=True,
    ),
    RestoreCase(
        name="stop_token_in_the_saved_part",
        prompt=_PROMPT,
        output=(" Hello", " world", _EOS),
        saved=3,
        finish_reason="stop",
        stop_token_terminated=True,
    ),
    RestoreCase(
        name="output_limit_reached_in_the_saved_part",
        prompt=_PROMPT,
        output=(" Hello", " world", "!"),
        saved=3,
        finish_reason="length",
    ),
)


def case_pieces() -> list[str]:
    """Every piece the cases use, for a word-level test vocabulary."""
    return sorted(
        {piece for case in RESTORE_CASES for piece in case.prompt + case.output}
    )
