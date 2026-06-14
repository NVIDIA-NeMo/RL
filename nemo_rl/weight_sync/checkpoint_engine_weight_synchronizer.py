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

"""Checkpoint-engine weight synchronizer for non-colocated deployments."""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.models.generation.interfaces import RefitCheckpointEngineConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def _flatten_checkpoint_engine_metadata(metadata_results: list[Any]) -> list[Any]:
    metadata = []
    for worker_metadata in metadata_results:
        if isinstance(worker_metadata, list):
            metadata.extend(worker_metadata)
        else:
            metadata.append(worker_metadata)
    return metadata


class CheckpointEngineWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using a configured checkpoint-engine backend."""

    def __init__(
        self,
        policy: Any,
        generation: Any,
        checkpoint_engine_config: RefitCheckpointEngineConfig,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._checkpoint_engine_config = checkpoint_engine_config
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._stale = True
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )

        with timer_context:
            try:
                update_success = self._transfer_weights(kv_scales=kv_scales)
                if not update_success:
                    backend = self._checkpoint_engine_config["backend"]
                    raise RuntimeError(
                        f"Weight transfer failed during {backend} checkpoint-engine "
                        "sync. This often indicates an issue with the transfer "
                        "backend or the generation worker."
                    )
                self._stale = False
            finally:
                ray.get(
                    self._policy.finalize_checkpoint_engine()
                    + self._generation.finalize_checkpoint_engine()
                )

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        pass

    def _transfer_weights(self, *, kv_scales: Optional[dict[str, float]]) -> bool:
        backend = self._checkpoint_engine_config["backend"]
        bucket_size_bytes = (
            self._checkpoint_engine_config["update_weights_bucket_megabytes"]
            * 1024
            * 1024
        )
        engine_kwargs = self._checkpoint_engine_config["engine_kwargs"][backend]

        ray.get(
            self._policy.init_checkpoint_engine(
                backend=backend,
                bucket_size_bytes=bucket_size_bytes,
                engine_kwargs=engine_kwargs,
            )
            + self._generation.init_checkpoint_engine(
                backend=backend,
                bucket_size_bytes=bucket_size_bytes,
                engine_kwargs=engine_kwargs,
            )
        )

        policy_metadata = _flatten_checkpoint_engine_metadata(
            ray.get(self._policy.prepare_checkpoint_engine())
        )
        generation_metadata = _flatten_checkpoint_engine_metadata(
            ray.get(self._generation.prepare_checkpoint_engine())
        )

        train_world_size = len(policy_metadata)
        rollout_world_size = len(generation_metadata)
        metadata = policy_metadata + generation_metadata
        ray.get(
            self._policy.init_checkpoint_engine_process_group(
                metadata=metadata,
                train_world_size=train_world_size,
                rollout_world_size=rollout_world_size,
            )
            + self._generation.init_checkpoint_engine_process_group(
                metadata=metadata,
                train_world_size=train_world_size,
                rollout_world_size=rollout_world_size,
            )
        )

        futures_train = self._policy.send_weights_via_checkpoint_engine(
            kv_scales=kv_scales
        )
        futures_inference = self._generation.update_weights_from_checkpoint_engine()
        ray.get(futures_train)
        results = ray.get(futures_inference)
        return all(result for result in results if result is not None)
