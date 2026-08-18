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

from nemo_rl.environments.central_agent_helpers.termination_manager.base import (
    BaseTerminationManager,
    TerminationManager,
    TurnState,
)

TERMINATION_MANAGERS = {"default": TerminationManager}


def build_termination_manager(
    cfg: Optional[dict[str, Any]],
) -> BaseTerminationManager:
    termination_type = (cfg or {}).get("type", "default")
    if termination_type not in TERMINATION_MANAGERS:
        raise ValueError(
            f"Unknown central_agent.termination.type '{termination_type}'. "
            f"Available: {sorted(TERMINATION_MANAGERS)}"
        )
    return TERMINATION_MANAGERS[termination_type](cfg)


__all__ = [
    "BaseTerminationManager",
    "TERMINATION_MANAGERS",
    "TerminationManager",
    "TurnState",
    "build_termination_manager",
]
