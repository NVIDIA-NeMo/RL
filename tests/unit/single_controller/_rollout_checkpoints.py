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

"""Rollout checkpointing for controllers that tests build without __init__."""

import asyncio
from types import SimpleNamespace
from typing import Any

from nemo_rl.algorithms.rollout_checkpointing.state import (
    ResumeSource,
    RolloutRuntimeState,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.environments.checkpointing.interfaces import NoEnvironmentCheckpoint

# Parts a bare controller may lack, and what stands in for each.
_BARE_CONTROLLER_PARTS = {
    "_dp_client": lambda: None,
    "_partition_id": lambda: "",
    "_buffer": lambda: None,
    "_dataloader": lambda: None,
    "_sampler": lambda: None,
    "_data_plane_checkpoint_barrier": lambda: None,
    "_environment_checkpoint": NoEnvironmentCheckpoint,
    "_rollout_runtime": RolloutRuntimeState,
    "_checkpointer": lambda: None,
    "_timeout": lambda: None,
    "_teacher_coordinator": lambda: None,
    "_rollout_recovery_enabled": lambda: False,
    "_algo_cfg": lambda: None,
    "_buffer_capacity": lambda: asyncio.Semaphore(0),
    "_train_steps": lambda: 0,
    "_trainer_version": lambda: 0,
    "_current_epoch": lambda: 0,
}


def attach_rollout_checkpointing(controller: Any) -> None:
    """Build what ``SingleControllerActor.__init__`` builds, from ``controller``'s parts.

    Call it again after a test swaps one of those parts, such as the
    environment checkpoint; parts a bare controller lacks are left empty, and
    state the previous snapshotter and restorer held carries over.
    """
    snapshots = getattr(controller, "_rollout_snapshots", None)
    restorer = getattr(controller, "_rollout_restorer", None)
    for name, stand_in in _BARE_CONTROLLER_PARTS.items():
        if not hasattr(controller, name):
            setattr(controller, name, stand_in())
    if not hasattr(controller, "_master_config"):
        controller._master_config = SimpleNamespace()
    if not hasattr(controller._master_config, "async_rl"):
        # Bare controllers carry their async config on the controller.
        controller._master_config.async_rl = getattr(controller, "_async_cfg", None)
    rollout_manager = getattr(controller, "_rollout_manager", None)
    if rollout_manager is None:
        # A stand-in without a ledger only needs the commit window and lock.
        controller._rollout_manager = SimpleNamespace(recovery_ledger=None)
    elif not hasattr(rollout_manager, "recovery_ledger"):
        rollout_manager.recovery_ledger = None
    build = SingleControllerActor.__ray_metadata__.modified_class._build_rollout_checkpointing
    controller._rollout_snapshots, controller._rollout_restorer = build(
        controller,
        bootstrap_identity=snapshots._bootstrap_identity if snapshots else None,
        # A test sets what the run resumed from on the controller; otherwise
        # keep what the restorer was built with.
        resume=getattr(
            controller, "_resume", restorer._resume if restorer else ResumeSource()
        ),
    )
    if restorer is not None:
        controller._rollout_restorer._groups_to_regenerate = (
            restorer._groups_to_regenerate
        )
        controller._rollout_restorer._requires_recovery_dispatch = (
            restorer._requires_recovery_dispatch
        )
