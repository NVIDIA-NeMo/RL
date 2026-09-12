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

"""Weight synchronization interface for NeMo-RL.

WeightSynchronizer is a dedicated abstraction that decouples weight transfer
logic from both PolicyInterface and GenerationInterface. It owns the
transfer of model weights between training and generation components.

Transport-specific implementations (IPC/ZMQ, Ray CUDA-IPC, NCCL collectives,
checkpoint engines) each encapsulate the transfer lifecycle, so algorithm code
never branches on backend type.

Colocated transports (IPC, SGLang colocated) own GPU phase transitions
internally (offload, prepare_for_generation, restore) as part of their
sync_weights() implementation. The NCCL collective transport is a pure data
mover; the orchestrator handles phase transitions externally since policy and
generation run on separate GPU clusters. The SGLang disaggregated transport
sits in between: it drives the generation-side phases but leaves the policy
resident on its own GPUs.

This interface assumes **global weight updates**: all generation workers
are updated atomically and are always at the same weight version. Per-worker
updates (where different replicas could be at different versions) are not
supported. In async GRPO, heterogeneous weight ages are handled at the
sample level (via replay buffer ``target_weight_versions`` tracking), not
at the synchronizer level.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from nemo_rl.utils.timer import Timer


class WeightSynchronizer(ABC):
    """Abstract base class for weight synchronization between policy and generation.

    Implementations handle the weight transfer for a specific transport
    mechanism (ZMQ IPC, Ray CUDA-IPC, NCCL collectives). The orchestrator calls
    sync_weights() without knowing which transport is being used or
    whether components are colocated; per-step staleness bookkeeping is
    owned by the training loop.

    Colocated transports own phase transitions internally
    (offload_before_refit, prepare_for_generation, offload_after_refit).
    Non-colocated collective and checkpoint-engine transports are pure data movers;
    the orchestrator handles phases externally.
    """

    @abstractmethod
    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        """Transfer the latest policy weights to the generation backend.

        This method encapsulates the full sync lifecycle:
        1. Prepare the policy side (e.g., offload optimizer state to free GPU memory)
        2. Prepare the generation side (e.g., allocate weight buffers)
        3. Transfer weights via the transport mechanism
        4. Verify the transfer succeeded
        5. Restore both sides to their ready state

        Step 1 is skipped by every transport whose policy keeps its own GPUs:
        the NCCL collective, checkpoint-engine, and SGLang disaggregated
        transports. Steps 2 and 5 are skipped by the NCCL collective and
        checkpoint-engine transports.

        Step 4 (verification) is performed explicitly by the IPC and NCCL
        collective transports, which check ``update_success`` and raise on
        failure. The SGLang transports let the engine RPC errors propagate.

        Args:
            timer: Optional Timer for profiling individual phases.
            kv_scales: Optional KV cache scales for FP8 quantization.
                Honored by the IPC/ZMQ and NCCL collective transports. The
                SGLang transports do not support this parameter and raise if
                it is set.

        Returns:
            Optional transport-specific scalar metrics for the current sync.

        Raises:
            RuntimeError: If the weight transfer fails.
        """
        pass

    @property
    @abstractmethod
    def is_stale(self) -> bool:
        """Whether the generation backend's weights are out of date.

        Returns True until the first successful sync_weights()
        completes, so a fresh run always performs its initial sync (a
        synchronizer that seeds current weights at construction may start
        False to skip it). Per-step staleness is tracked by the training
        loop, not here.
        """
        pass

    @abstractmethod
    def init_communicator(self) -> None:
        """Initialize any communication channels needed for weight transfer.

        Called once during setup, after policy and generation workers are
        constructed. For the IPC and SGLang transports this only prepares
        refit metadata. For NCCL collectives this also initializes the
        process group.
        """
        pass

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool = False
    ) -> Optional[bool]:
        """Bring the transport's communicator in line with the live generation fleet.

        Called immediately before every refit, rather than in response to a death event.
        Reconciling on a schedule is idempotent and converges after a missed or reordered
        notification, and it is the only point where the refit group is provably idle and
        every rank is synchronized -- which matters because the collectives that change
        membership are themselves collectives.

        Args:
            absent_shards: shard indices whose process cannot take part in a collective
                (see ``GenerationFleetHealth.absent_shards``). Note this is not the
                complement of the serving set: a shard withheld from traffic may still be
                alive and able to refit.
            force: rebuild even when the membership is unchanged. Implementations
                skip a rebuild whose absent set matches what they last built with,
                which is what stops a lost shard costing a rebuild on every
                subsequent step. The recovery path sets this because after an abort
                the membership is identical and the communicator is *gone*.

        Returns:
            True if the communicator was rebuilt, False if nothing needed rebuilding,
            and None if this transport owns no membership to reconcile at all. The
            caller reports a failed refit differently for the last case: "no shard was
            absent" and "there was nothing to reconcile" point at different causes.

        Raises:
            NoSurvivingShards: if every generation shard is gone, so there is nothing
                left to rebuild onto.

        The default is a no-op: transports that own no NCCL world of their own -- IPC,
        HTTP, checkpoint-engine -- have no membership to reconcile. It returns None
        rather than False so the controller does not report a refit failure on one of
        them as "no shard could be identified as absent", which would send the reader
        hunting for a silent rank that does not exist.
        """
        del absent_shards
        return None

    @abstractmethod
    def shutdown(self) -> None:
        """Release all communication resources."""
        pass
