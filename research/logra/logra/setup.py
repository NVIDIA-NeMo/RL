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

"""Small initialization helpers; model construction stays in NeMo RL."""

import importlib

import torch


def build_scheduler(optimizer, config):
    """Rebuild the configured schedule for the two optimizer groups."""

    def create(spec):
        module, name = spec["name"].rsplit(".", 1)
        return getattr(importlib.import_module(module), name)(
            optimizer, **spec["kwargs"]
        )

    if config is None:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    if isinstance(config, dict):
        return create(config)
    schedulers = [create(spec) for spec in config if "name" in spec]
    milestones = next(spec["milestones"] for spec in config if "milestones" in spec)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
