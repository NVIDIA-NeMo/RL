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
"""Token-budget helpers shared by the privileged value's prefix builders."""

from typing import Any

TRUNCATION_MARKER = "\n... [truncated]"
# A part that had content but got no budget is marked rather than omitted, so
# "this exists but was cut" and "this prompt has none" do not look identical.
OMITTED_MARKER = "... [omitted: token budget]"


def cap_tokens(text: str, max_tokens: int, tokenizer: Any) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_tokens``, deterministically.

    Returns the kept text and whether anything was cut. The flag, not a token
    count, decides truncation: a field cut down to the bare marker can count as
    many tokens as the original.

    Truncation is a pure function of the instance, never of the rollout, so
    sibling rollouts of one task share an identical block.
    """
    if not text:
        return "", False
    # Char pre-cut at 20 chars/token, above any text's ratio, so a multi-MB field
    # is never fully tokenized.
    head = text[: max_tokens * 20]
    ids = tokenizer.encode(head, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return head, len(head) < len(text)
    # Reserve room for the marker so the result respects max_tokens.
    marker_len = len(tokenizer.encode(TRUNCATION_MARKER, add_special_tokens=False))
    keep = max(max_tokens - marker_len, 0)
    return tokenizer.decode(ids[:keep]) + TRUNCATION_MARKER, True


def count_tokens(text: str, tokenizer: Any, hint: int) -> int:
    """Token count, with a char pre-cut so a multi-MB field is never fully encoded."""
    if not text:
        return 0
    return len(tokenizer.encode(text[: max(hint, 1) * 20], add_special_tokens=False))
