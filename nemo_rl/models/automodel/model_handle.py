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

"""The model parts an Automodel policy worker owns."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn


@dataclass
class ModelRuntime:
    """Local model parts and their per-part optimizer state.

    Without pipeline parallelism there is one part, the whole model, with stage
    ID 0. With pipeline parallelism each rank owns one part per local pipeline
    stage (several under ``interleaved_1f1b``), and ``stage_ids`` holds their
    global stage indices in the same order, for example ``[0, 2]`` on PP rank 0
    of an interleaved pipeline with two ranks.
    """

    parts: list[nn.Module]
    stage_ids: list[int]
    # One optimizer and one LR scheduler per part, in ``parts`` order. Empty
    # when the worker was built without an optimizer.
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[Any]
    # Automodel's ``AutoPipeline`` when pipeline parallelism is enabled.
    pipeline: Optional[Any]

    @property
    def pp_enabled(self) -> bool:
        """Whether the parts are pipeline stages driven by ``pipeline``."""
        return self.pipeline is not None

    @property
    def has_first_stage(self) -> bool:
        """Whether this rank runs the stage that consumes input token IDs."""
        return not self.pp_enabled or self.pipeline.info.has_first_stage

    @property
    def has_last_stage(self) -> bool:
        """Whether this rank runs the stage that produces logits."""
        return not self.pp_enabled or self.pipeline.info.has_last_stage
