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

"""Blocking checkpoint hot-load through the optional cwtrain client."""

from typing import Any

from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

from nemo_rl.models.generation.coreweave.config import CoreWeaveConfig


class CoreWeaveHotLoadWeightSynchronizer(HotLoadWeightSynchronizer):
    """Commit the served identity only after all replicas accept the checkpoint."""

    def __init__(self, policy: Any, generation: Any) -> None:
        settings = CoreWeaveConfig(**generation.cfg["coreweave_cfg"])
        settings.validate_training()
        super().__init__(
            policy,
            generation,
            generation.cfg,
            hf_checkpoint=policy.cfg["model_name"],
        )

    def sync_weights(
        self, *, timer: Any = None, kv_scales: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Block sampling from a failed update, including after an earlier success."""
        self._stale = True
        return super().sync_weights(timer=timer, kv_scales=kv_scales)
