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

"""Build a ``RolloutRestorer`` and dispatch mode over a test's controller."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nemo_rl.algorithms.rollout_checkpointing import (
    CheckpointToRestore,
    NoRolloutCheckpointing,
    PromptDispatch,
    RolloutRestorer,
)
from nemo_rl.algorithms.single_controller import _ControllerRolloutHost
from nemo_rl.experience.environment_checkpoint import NoEnvironmentCheckpoint


def restorer_for(
    controller: Any,
    checkpoint: CheckpointToRestore = CheckpointToRestore(),
) -> RolloutRestorer:
    """The restorer ``controller`` builds, from the fields the test set on it.

    Tests build bare controllers with only the fields they need, so a missing
    field is passed as ``None``. A test that sets ``_async_cfg`` but leaves
    ``async_rl`` off its ``_master_config`` gets the two merged, as in a real
    config.
    """
    master_config = getattr(controller, "_master_config", None)
    async_cfg = getattr(controller, "_async_cfg", None)
    if async_cfg is not None and getattr(master_config, "async_rl", None) is None:
        master_config = SimpleNamespace(
            **(vars(master_config) if master_config is not None else {}),
            async_rl=async_cfg,
        )
    return RolloutRestorer(
        host=_ControllerRolloutHost(controller),
        master_config=master_config,  # type: ignore[arg-type]
        dp_client=getattr(controller, "_dp_client", None),
        partition_id=getattr(controller, "_partition_id", "rollout_data"),
        buffer=getattr(controller, "_buffer", None),  # type: ignore[arg-type]
        sampler=getattr(controller, "_sampler", None),  # type: ignore[arg-type]
        dataloader=getattr(controller, "_dataloader", None),
        rollout_manager=getattr(controller, "_rollout_manager", None),  # type: ignore[arg-type]
        data_plane_barrier=getattr(controller, "_data_plane_checkpoint_barrier", None),  # type: ignore[arg-type]
        environment_checkpoint=getattr(
            controller, "_environment_checkpoint", NoEnvironmentCheckpoint()
        ),
        checkpoint=checkpoint,
    )


def dispatch_for(controller: Any, *, records_prompt_groups: bool) -> PromptDispatch:
    """The dispatch mode over ``controller``'s barrier and rollout manager.

    Ledger dispatch reads both, so set them on the controller first.
    """
    return PromptDispatch(
        data_plane_barrier=getattr(controller, "_data_plane_checkpoint_barrier", None),  # type: ignore[arg-type]
        rollout_manager=getattr(controller, "_rollout_manager", None),  # type: ignore[arg-type]
        records_prompt_groups=records_prompt_groups,
    )


def use_dispatch(
    controller: Any, *, records_prompt_groups: bool
) -> NoRolloutCheckpointing:
    """Give ``controller`` a ``NoRolloutCheckpointing`` with this dispatch mode.

    Ledger dispatch reads the controller's barrier and rollout manager, so set
    them first.
    """
    component = NoRolloutCheckpointing(
        restorer=restorer_for(controller),
        dispatch=dispatch_for(controller, records_prompt_groups=records_prompt_groups),
    )
    controller._rollout_checkpointing = component
    return component
