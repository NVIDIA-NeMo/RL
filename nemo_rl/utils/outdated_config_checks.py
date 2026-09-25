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
"""Rejects config the code no longer accepts, pointing at the migration to apply."""

from typing import Any, Iterator


def _train_backend_configs(config: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yields (dotted path, config) for every block that selects a training backend.

    The policy, the value model, each teacher and the reward-model environment. A check
    for any backend key can iterate these rather than re-deriving the locations.
    """
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    if not isinstance(config, dict):
        return

    # policy and value model blocks
    blocks = [(section, config.get(section)) for section in ("policy", "value")]

    # teacher blocks
    teachers = config.get("teachers")
    if isinstance(teachers, (list, tuple)):
        blocks += [(f"teachers.{i}", t) for i, t in enumerate(teachers)]

    # reward-model environment block
    env = config.get("env")
    if isinstance(env, dict):
        blocks.append(("env.reward_model", env.get("reward_model")))

    for path, block in blocks:
        if isinstance(block, dict):
            yield path, block


def reject_outdated_dtensor_v2_key(config: Any) -> None:
    """Fail when a config still carries the removed dtensor_cfg._v2 key.

    Args:
        config: The resolved MasterConfig, or any mapping holding one.
    """
    for path, backend_config in _train_backend_configs(config):
        dtensor_cfg = backend_config.get("dtensor_cfg")
        if not isinstance(dtensor_cfg, dict) or "_v2" not in dtensor_cfg:
            continue
        config_path = f"{path}.dtensor_cfg"
        raise ValueError(
            f"DTensor v1 ({config_path}._v2=false) and the _v2 key itself have been "
            f"removed. DTensor is always the Automodel backend now, which is what "
            f"_v2=true selected, so delete the key."
        )


def check_outdated_config(config: Any) -> None:
    """Fail fast on config the code no longer accepts, naming the migration to apply.

    Call this from every entrypoint right after the MasterConfig is built, so a stale
    config fails before any cluster or worker is created. Add a check here whenever a key
    is removed or the shape it accepts changes.

    Args:
        config: The resolved MasterConfig, or any mapping holding one.
    """
    reject_outdated_dtensor_v2_key(config)
