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

"""Checkpoint boundary checks that do not import Ray or load checkpoint payloads."""


def validate_replay_restore(
    *,
    checkpoint_path: str | None,
    ray_reference_transport: bool,
    load_replay_buffer: bool | None,
) -> None:
    """Reject disk replay whose routed-expert data belongs to an old Ray runtime.

    Fresh training and inline transport are unaffected. Without a portable route
    archive and runtime-identity proof, loading Ray-reference replay is unsafe
    even when Slurm reuses the same job ID on requeue.
    """
    if (
        checkpoint_path is not None
        and ray_reference_transport
        and load_replay_buffer is not False
    ):
        raise ValueError(
            "Resuming policy.router_replay.transport=ray requires "
            "checkpointing.load_replay_buffer=false. Saved route actor names "
            "and ObjectRefs are not portable to a new Ray cluster. Restore "
            "model/optimizer/dataloader state and regenerate uncommitted rollouts."
        )
