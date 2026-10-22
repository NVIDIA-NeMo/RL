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

"""Inject a research worker through the native GRPO policy factory."""

from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

from logra.actor_environments import WORKER
from nemo_rl.models.policy.lm_policy import Policy

if TYPE_CHECKING:
    from logra.config import LoGRAConfig
    from logra.cua import CuaConfig


def make_policy_factory(
    logra_config: "LoGRAConfig", cua_config: "CuaConfig | None" = None
) -> Callable[..., Policy]:
    def factory(*, config, **kwargs):
        config = deepcopy(config)
        config["logra"] = logra_config.model_dump()
        if cua_config is not None:
            config["cua"] = cua_config.model_dump()
        return Policy(config=config, worker_extension_cls_fqn=WORKER, **kwargs)

    return factory
