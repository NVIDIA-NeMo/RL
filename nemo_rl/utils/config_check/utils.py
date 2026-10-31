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

from typing import Any, Iterator

from pydantic import BaseModel


def train_backend_configs(
    config: BaseModel | dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yields (dotted path, config) for every block that selects a training backend.

    The policy, the value model, the teachers and the reward-model environment. A check
    for any backend key can iterate these rather than re-deriving the locations.

    Distillation keeps its single teacher under "teacher", while the multi-teacher
    algorithms use a "teachers" list, so both spellings are visited.
    """
    if isinstance(config, BaseModel):
        config = dict(config)

    # policy, value model and single-teacher (distillation) blocks
    blocks = [
        (section, config.get(section)) for section in ("policy", "value", "teacher")
    ]

    # multi-teacher blocks
    teachers = config.get("teachers")
    if isinstance(teachers, (list, tuple)):
        blocks += [(f"teachers.{i}", t) for i, t in enumerate(teachers)]

    # reward-model environment block
    env = config.get("env")
    if isinstance(env, dict):
        blocks.append(("env.reward_model", env.get("reward_model")))

    for path, block in blocks:
        if isinstance(block, BaseModel):
            block = dict(block)
        if isinstance(block, dict):
            yield path, block
