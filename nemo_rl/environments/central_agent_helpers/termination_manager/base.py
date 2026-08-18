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

"""When the agent loop stops."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TurnState:
    """Everything the termination check sees about the turn that just ended."""

    response: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    steering_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_stats: dict[str, Any] = field(default_factory=dict)
    has_output_message: bool = False


class BaseTerminationManager:
    def is_termination_criteria_met(self, turn: TurnState) -> bool:
        """Called once per turn, after tool results and steering are appended."""
        raise NotImplementedError

    def termination_stats(self) -> dict[str, Any]:
        raise NotImplementedError


class TerminationManager(BaseTerminationManager):
    """Default criteria: truncation, natural stop, malformed budget, turn budget.

    ``stop_on_no_tool_calls`` reproduces ``simple_agent``: a response with an
    assistant message and no tool calls ends the rollout. It does not fire when
    steering injected messages, since steering exists to continue past an
    apparent final answer.
    """

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        cfg = cfg or {}
        self._max_turns = cfg.get("max_turns")
        self._max_malformed = cfg.get("max_malformed_tool_calls")
        self._stop_on_no_tool_calls = cfg.get("stop_on_no_tool_calls", True)
        self._stop_on_incomplete_details = cfg.get("stop_on_incomplete_details", True)
        self._turns = 0
        self._reason: Optional[str] = None

    def is_termination_criteria_met(self, turn: TurnState) -> bool:
        self._turns += 1
        malformed = turn.tool_stats.get("tool_calls_malformed", 0)

        if self._stop_on_incomplete_details and turn.response.get("incomplete_details"):
            self._reason = "incomplete_details"
        elif (
            self._stop_on_no_tool_calls
            and not turn.tool_calls
            and turn.has_output_message
            and not turn.steering_messages
        ):
            self._reason = "no_tool_calls"
        elif self._max_malformed is not None and malformed >= self._max_malformed:
            self._reason = "max_malformed_tool_calls"
        elif self._max_turns and self._turns >= self._max_turns:
            self._reason = "max_turns"
        else:
            self._reason = None
        return self._reason is not None

    def termination_stats(self) -> dict[str, Any]:
        return {
            "central_agent_termination_reason": self._reason,
            "central_agent_turns": self._turns,
        }
