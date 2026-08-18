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

from typing import Any, Optional

from nemo_rl.environments.central_agent_helpers.tool_call_manager.base import (
    BaseToolCallManager,
    CallTool,
    ParallelToolCallManager,
    SerialToolCallManager,
    function_call_output,
)

TOOL_CALL_MANAGERS = {
    "serial": SerialToolCallManager,
    "parallel": ParallelToolCallManager,
}


def build_tool_call_manager(
    cfg: Optional[dict[str, Any]], call_tool: CallTool
) -> BaseToolCallManager:
    mode = (cfg or {}).get("mode", "serial")
    if mode not in TOOL_CALL_MANAGERS:
        raise ValueError(
            f"Unknown central_agent.tool_calls.mode '{mode}'. "
            f"Available: {sorted(TOOL_CALL_MANAGERS)}"
        )
    return TOOL_CALL_MANAGERS[mode](cfg, call_tool)


__all__ = [
    "BaseToolCallManager",
    "ParallelToolCallManager",
    "SerialToolCallManager",
    "TOOL_CALL_MANAGERS",
    "build_tool_call_manager",
    "function_call_output",
]
