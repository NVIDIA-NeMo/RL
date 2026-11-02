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

from nemo_rl.algorithms.rollout_checkpointing.restore import RolloutRestorer
from nemo_rl.algorithms.rollout_checkpointing.snapshots import RolloutSnapshotter
from nemo_rl.algorithms.rollout_checkpointing.state import (
    ResumeSource,
    RolloutParts,
    RolloutRuntimeState,
    TrainingProgress,
)
from nemo_rl.environments.checkpointing.interfaces import NoEnvironmentCheckpoint


def attach_rollout_checkpointing(controller: Any) -> None:
    """Build what ``SingleControllerActor.__init__`` builds, from ``controller``'s parts.

    Call it again after a test swaps one of those parts, such as the
    environment checkpoint; parts a bare controller lacks are left empty, and
    state the previous snapshotter and restorer held carries over.
    """
    rollout_manager = getattr(controller, "_rollout_manager", None)
    snapshots = getattr(controller, "_rollout_snapshots", None)
    restorer = getattr(controller, "_rollout_restorer", None)
    master_config = getattr(controller, "_master_config", SimpleNamespace())
    if not hasattr(master_config, "async_rl"):
        # Bare controllers carry their async config on the controller.
        master_config = SimpleNamespace(
            **vars(master_config), async_rl=getattr(controller, "_async_cfg", None)
        )
    if not hasattr(controller, "_rollout_runtime"):
        controller._rollout_runtime = RolloutRuntimeState()
    parts = RolloutParts(
        master_config=master_config,
        dp_client=getattr(controller, "_dp_client", None),
        partition_id=getattr(controller, "_partition_id", ""),
        buffer=getattr(controller, "_buffer", None),
        dataloader=getattr(controller, "_dataloader", None),
        sampler=getattr(controller, "_sampler", None),
        # A stand-in without a ledger only needs the commit window and lock.
        rollout_manager=rollout_manager
        if hasattr(rollout_manager, "recovery_ledger")
        else SimpleNamespace(recovery_ledger=None),
        data_plane_barrier=getattr(controller, "_data_plane_checkpoint_barrier", None),
        environment_checkpoint=getattr(
            controller, "_environment_checkpoint", NoEnvironmentCheckpoint()
        ),
        runtime_state=controller._rollout_runtime,
        read_progress=lambda: TrainingProgress(
            train_steps=getattr(controller, "_train_steps", 0),
            trainer_version=getattr(controller, "_trainer_version", 0),
            current_epoch=getattr(controller, "_current_epoch", 0),
        ),
        log_metrics=controller._log_telemetry_metrics,
    )
    controller._rollout_snapshots = RolloutSnapshotter(
        parts,
        checkpointer=getattr(controller, "_checkpointer", None),
        timeout=getattr(controller, "_timeout", None),
        bootstrap_identity=snapshots._bootstrap_identity if snapshots else None,
        teacher_coordinator=getattr(controller, "_teacher_coordinator", None),
        records_prompt_groups=lambda: getattr(
            controller, "_rollout_recovery_enabled", False
        ),
    )
    controller._rollout_restorer = RolloutRestorer(
        parts,
        algo_cfg=getattr(controller, "_algo_cfg", None),
        buffer_capacity=getattr(controller, "_buffer_capacity", asyncio.Semaphore(0)),
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
