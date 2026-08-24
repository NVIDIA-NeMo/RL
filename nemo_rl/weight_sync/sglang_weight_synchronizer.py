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

"""Weight synchronizers for the SGLang generation backend.

These run in the driver process, which is synced without a training-backend
extra, so nothing here may import ``megatron.bridge`` or ``nemo_automodel``:
the refit only drives the ``policy`` and ``policy_generation`` facades.

The refit lifecycle — connect, pause, conditional KV invalidation, a
begin/end weight-update session around the bucket transfer, then continue —
is shared; the subclasses supply the transport-specific connect and transfer
and own the GPU phase transitions around them.

Colocated:
  1. policy.offload_before_refit()                        -- free GPU for staging
  2. generation.prepare_for_generation(tags=["weights"])   -- allocate buffers
  3. _refit()                                              -- Ray CUDA-IPC transfer
  4. policy.offload_after_refit()                          -- restore optimizer state
  5. generation.prepare_for_generation(tags=["kv_cache"])  -- rebuild KV cache

Disaggregated:
  1. generation.prepare_for_generation(tags=["weights"])
  2. _refit()                                              -- NCCL broadcast
  3. generation.prepare_for_generation(tags=["kv_cache"])

The policy offload steps are skipped when disaggregated: the trainer keeps
its GPUs to itself, so there is nothing to make room for.

``prepare_for_generation`` runs on both paths. It is gated internally on
``sglang_server_config.needs_offload``, which is an independent knob — with
``needs_offload: true`` (what every shipped config sets) these calls issue
real ``resume_memory_occupation`` RPCs even when disaggregated, and the
engines need them because ``finish_generation`` released that memory. They
are only a no-op when ``needs_offload`` is false.
"""

import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class _SGLangWeightSynchronizer(WeightSynchronizer):
    """Shared plumbing for the SGLang synchronizers.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: SGLangGeneration instance.
        refit_buffer_size_gb: Fixed bucket size in GB for the weight transfer.
            If None, it is computed dynamically from free GPU memory.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        refit_buffer_size_gb: Optional[float] = None,
    ):
        self._policy = policy
        self._generation = generation
        self._refit_buffer_size_gb = refit_buffer_size_gb
        self._stale = True
        if self._generation.pause_generation_mode == "in_place":
            raise ValueError(
                "pause_generation_mode='in_place' is unsafe for weight refit because "
                "it preserves KV cache entries created by the previous weights."
            )

    @property
    def is_stale(self) -> bool:
        return self._stale

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        # Process groups live on policy/engine actors and are released with them.
        pass

    def _quantization_cfg(self) -> dict:
        return dict(self._generation.sglang_cfg["sglang_cfg"]["quantization"])

    @abstractmethod
    def _connect(self, *, rollout_engines: list, engine_gpu_counts, engine_gpu_offsets):
        """Bring up the trainer-side transport for a new engine layout."""

    @abstractmethod
    def _send_buckets(
        self,
        *,
        rollout_engines: list,
        buffer_size_bytes: int,
        target_precision: str,
        sglang_quantization_cfg: dict,
    ) -> list:
        """Dispatch the transfer to the policy workers, returning Ray futures."""

    def _reject_kv_scales(self, kv_scales: Optional[dict[str, float]]) -> None:
        # The SGLang refit carries no KV-cache scales. Reject them rather than
        # dropping them silently, so an FP8-KV config fails loudly.
        if kv_scales is not None:
            raise ValueError(
                "The SGLang weight transports do not support kv_scales; "
                f"got {sorted(kv_scales)!r}."
            )

    def _refit(self, buffer_size_bytes: int) -> None:
        from nemo_rl.models.generation.sglang.config import (
            get_sglang_quantization_scheme,
        )

        sglang_quantization_cfg = self._quantization_cfg()
        # Validating read: a misspelled scheme must raise, not fall back to BF16.
        target_precision = get_sglang_quantization_scheme(sglang_quantization_cfg)

        (
            rollout_engines,
            _rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
        ) = self._generation.get_updatable_engines_and_lock()

        if num_new_engines > 0:
            self._connect(
                rollout_engines=rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            self._generation.clear_updatable_num_new_engines()

        pause_mode = self._generation.pause_generation_mode
        # Each acquired state gets its own guard: anything that raises between
        # pausing and the transfer -- invalidate_kv_cache or begin_weight_update
        # -- must still resume generation, or every engine stays paused for the
        # rest of the run with no error pointing at why.
        try:
            self._generation.pause_generation(mode=pause_mode)
            if not self._generation.invalidate_kv_cache():
                raise RuntimeError("SGLang KV cache invalidation failed before refit.")

            self._generation.begin_weight_update()
            try:
                # The per-worker actor method awaits each chunk itself, but the
                # policy-group dispatch still returns one Ray future per worker;
                # await those here to wait for all trainer ranks.
                ray.get(
                    self._send_buckets(
                        rollout_engines=rollout_engines,
                        buffer_size_bytes=buffer_size_bytes,
                        target_precision=target_precision,
                        sglang_quantization_cfg=sglang_quantization_cfg,
                    )
                )
            finally:
                # Only closes a session that actually opened: if
                # begin_weight_update raised, this inner block never ran.
                self._generation.end_weight_update()
        finally:
            # Resume on every path, so a failed refit leaves the engine usable
            # instead of wedged in the update state.
            self._generation.continue_generation()

    def _timed_refit(self, timer: Optional[Timer]) -> None:
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            self._refit(self._compute_buffer_size())

    def _compute_buffer_size(self) -> int:
        if self._refit_buffer_size_gb is not None:
            if self._refit_buffer_size_gb <= 0:
                raise ValueError("refit_buffer_size_gb must be > 0")
            return int(self._refit_buffer_size_gb * (1024**3))

        memory_ratio_raw = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
        try:
            memory_ratio = float(memory_ratio_raw)
        except ValueError as exc:
            raise ValueError(
                f"NRL_REFIT_BUFFER_MEMORY_RATIO must be a valid float, got {memory_ratio_raw!r}"
            ) from exc
        if memory_ratio <= 0:
            raise ValueError("NRL_REFIT_BUFFER_MEMORY_RATIO must be > 0")

        return int(self._policy.get_free_memory_bytes() * memory_ratio)


class SGLangColocatedWeightSynchronizer(_SGLangWeightSynchronizer):
    """Policy and SGLang engines share GPUs; weights move over Ray CUDA IPC.

    The trainer offloads before staging weights and re-offloads afterwards so
    the engines can take the memory back for their KV cache.
    """

    def _connect(self, *, rollout_engines, engine_gpu_counts, engine_gpu_offsets):
        self._policy.connect_sglang_rollout_engines(
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
        )

    def _send_buckets(
        self,
        *,
        rollout_engines: list,
        buffer_size_bytes: int,
        target_precision: str,
        sglang_quantization_cfg: dict,
    ) -> list:
        return self._policy.update_weights_to_sglang_colocated(
            rollout_engines=rollout_engines,
            buffer_size_bytes=buffer_size_bytes,
            target_precision=target_precision,
            sglang_quantization_cfg=sglang_quantization_cfg,
        )

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        self._reject_kv_scales(kv_scales)
        self._policy.offload_before_refit()

        sync_succeeded = False
        try:
            self._generation.prepare_for_generation(tags=["weights"])
            self._timed_refit(timer)
            sync_succeeded = True
        finally:
            self._policy.offload_after_refit()
            self._generation.prepare_for_generation(tags=["kv_cache"])

        self._stale = not sync_succeeded
        return None


class SGLangDisaggregatedWeightSynchronizer(_SGLangWeightSynchronizer):
    """SGLang engines run on their own GPUs; weights move over NCCL broadcast.

    No policy offload: the trainer is not competing with the engines for
    memory, and ``prepare_for_training`` onloads unconditionally anyway.
    """

    def _connect(self, *, rollout_engines, engine_gpu_counts, engine_gpu_offsets):
        self._policy.connect_sglang_rollout_engines_distributed(
            rollout_engines=rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
        )

    def _send_buckets(
        self,
        *,
        rollout_engines: list,
        buffer_size_bytes: int,
        target_precision: str,
        sglang_quantization_cfg: dict,
    ) -> list:
        return self._policy.update_weights_to_sglang_distributed(
            rollout_engines=rollout_engines,
            rollout_engine_lock=self._generation.rollout_engine_lock,
            buffer_size_bytes=buffer_size_bytes,
            target_precision=target_precision,
            sglang_quantization_cfg=sglang_quantization_cfg,
        )

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        self._reject_kv_scales(kv_scales)

        sync_succeeded = False
        try:
            self._generation.prepare_for_generation(tags=["weights"])
            self._timed_refit(timer)
            sync_succeeded = True
        finally:
            self._generation.prepare_for_generation(tags=["kv_cache"])

        self._stale = not sync_succeeded
        return None
