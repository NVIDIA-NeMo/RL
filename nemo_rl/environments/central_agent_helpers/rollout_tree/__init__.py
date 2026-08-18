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

from typing import Any, Callable, Optional

from nemo_rl.environments.central_agent_helpers.rollout_tree.base import BaseRolloutTree
from nemo_rl.environments.central_agent_helpers.rollout_tree.linear import (
    LinearRolloutTree,
)

ROLLOUT_TREES = {"linear": LinearRolloutTree}


def build_rollout_tree(
    cfg: Optional[dict[str, Any]],
    prompt: list[dict[str, Any]],
    model_client: Optional[Callable[..., Any]] = None,
) -> BaseRolloutTree:
    tree_type = (cfg or {}).get("type", "linear")
    if tree_type not in ROLLOUT_TREES:
        raise ValueError(
            f"Unknown central_agent.rollout_tree.type '{tree_type}'. "
            f"Available: {sorted(ROLLOUT_TREES)}"
        )
    return ROLLOUT_TREES[tree_type](prompt, model_client, cfg)


__all__ = ["BaseRolloutTree", "LinearRolloutTree", "build_rollout_tree"]
