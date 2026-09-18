# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Per-rollout assistant-token accounting, independent of reward shaping."""

from dataclasses import dataclass, field


@dataclass
class OutputTokenBudget:
    total: int | None
    used: int = field(init=False, default=0)
    calls: int = field(init=False, default=0)
    allowance: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.total is not None and (type(self.total) is not int or self.total <= 0):
            raise ValueError("Rollout output budget must be a positive integer")

    @property
    def exhausted(self) -> bool:
        return self.total is not None and self.used >= self.total

    def limit(self, per_call: int | None) -> int | None:
        if per_call is not None and (type(per_call) is not int or per_call <= 0):
            raise ValueError("Per-call output budget must be a positive integer")
        if self.total is None:
            return per_call
        if self.exhausted:
            raise RuntimeError(
                "Cannot issue a model call after exhausting the rollout budget"
            )
        remaining = self.total - self.used
        self.allowance = min(per_call, remaining) if per_call is not None else remaining
        return self.allowance

    def consume(self, output_tokens: int | None) -> None:
        if self.total is None:
            return
        if type(output_tokens) is not int or output_tokens < 0:
            raise RuntimeError("Missing or invalid assistant output-token usage")
        if self.allowance is None or output_tokens > self.allowance:
            raise RuntimeError("Model returned more output tokens than requested")
        self.used += output_tokens
        self.calls += 1
        self.allowance = None
