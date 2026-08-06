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
"""Strict, vLLM-free configuration for per-token NVFP4 rollout."""

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

NVFP4_QUANT_PATTERNS = ["*.experts.*"]
NVFP4_PERTOKEN_ZMQ_TIMEOUT_MS = 600_000
DEFAULT_NVFP4_IGNORE = [
    "*lm_head*",
    "*mlp.gate",
    "*mlp.gate.*",
    "*mlp.shared_expert*",
    "*self_attn*",
    "*embed_tokens*",
    "*input_layernorm*",
    "*post_attention_layernorm*",
    "*norm*",
]

_FULL_EXPERT_LAYER_IGNORE_RE = re.compile(r"^\*\.layers\.\d+\.mlp\.experts\*$")


def _require_full_expert_layer_ignores(value: list[str]) -> list[str]:
    invalid = [
        pattern
        for pattern in value
        if _FULL_EXPERT_LAYER_IGNORE_RE.fullmatch(pattern) is None
    ]
    if invalid:
        raise ValueError(
            "additional_ignore may only exclude complete expert layers using "
            "'*.layers.<index>.mlp.experts*'; invalid patterns: "
            f"{invalid}"
        )
    return value


class NvFp4PerTokenRolloutConfig(BaseModel, extra="forbid"):
    """User configuration for the constrained per-token NVFP4 rollout."""

    enabled: bool = False
    additional_ignore: Annotated[
        list[str], AfterValidator(_require_full_expert_layer_ignores)
    ] = Field(default_factory=list)

    @property
    def quant_patterns(self) -> list[str]:
        """Return the fixed expert-only producer selection."""
        return list(NVFP4_QUANT_PATTERNS)

    def resolved_ignore(self) -> list[str]:
        return [*DEFAULT_NVFP4_IGNORE, *self.additional_ignore]
