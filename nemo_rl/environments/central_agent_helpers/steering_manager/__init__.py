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

from nemo_rl.environments.central_agent_helpers.steering_manager.base import (
    BaseSteeringMessageManager,
    NoOpSteeringMessageManager,
    PostSteering,
    SteeringMessageManager,
)

STEERING_MANAGERS = {
    "noop": NoOpSteeringMessageManager,
    "resources_server": SteeringMessageManager,
}


def build_steering_manager(
    cfg: Optional[dict[str, Any]], post: Optional[PostSteering]
) -> BaseSteeringMessageManager:
    cfg = cfg or {}
    # `enabled` is the switch; `type` picks which enabled implementation.
    steering_type = (
        "noop" if not cfg.get("enabled") else cfg.get("type", "resources_server")
    )
    if steering_type not in STEERING_MANAGERS:
        raise ValueError(
            f"Unknown central_agent.steering_message.type '{steering_type}'. "
            f"Available: {sorted(STEERING_MANAGERS)}"
        )
    if steering_type == "noop":
        return NoOpSteeringMessageManager(cfg, post)
    assert post is not None, "steering requires a bound post callable"
    return STEERING_MANAGERS[steering_type](cfg, post)


__all__ = [
    "BaseSteeringMessageManager",
    "NoOpSteeringMessageManager",
    "STEERING_MANAGERS",
    "SteeringMessageManager",
    "build_steering_manager",
]
