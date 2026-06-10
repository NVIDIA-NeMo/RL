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

Transport-specific implementations (IPC/ZMQ, HTTP, NCCL collectives,
checkpoint-engine backends) each encapsulate the transfer lifecycle, so
algorithm code never branches on backend type.

Colocated transports (IPC, HTTP) own GPU phase transitions internally
(offload, prepare_for_generation, restore) as part of their sync_weights()
implementation. Non-colocated transports are pure data movers; the
orchestrator handles phase transitions externally since policy and
generation run on separate GPU clusters.

This interface assumes **global weight updates**: all generation workers
are updated atomically and are always at the same weight version. Per-worker
updates (where different replicas could be at different versions) are not
supported. In async GRPO, heterogeneous weight ages are handled at the
sample level (via replay buffer ``target_weight_versions`` tracking), not
at the synchronizer level.
"""

from abc import ABC, abstractmethod
from typing import Optional

from nemo_rl.utils.timer import Timer


class WeightSynchronizer(ABC):
    """Abstract base class for weight synchronization between policy and generation.

    Implementations handle the weight transfer for a specific transport
    mechanism (ZMQ IPC, HTTP, NCCL collectives, checkpoint-engine backends).
    The orchestrator calls sync_weights() and mark_stale() without knowing
    which transport is being used or whether components are colocated.

    Colocated transports (IPC, HTTP) own phase transitions internally
    (offload_before_refit, prepare_for_generation, offload_after_refit).
    Non-colocated transports are pure data movers; the orchestrator handles
    phases externally.
    """

    @abstractmethod
    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        """Transfer the latest policy weights to the generation backend.

        This method encapsulates the full sync lifecycle:
        1. Prepare the policy side (e.g., offload optimizer state to free GPU memory)
        2. Prepare the generation side (e.g., allocate weight buffers)
        3. Transfer weights via the transport mechanism
        4. Verify the transfer succeeded
        5. Restore both sides to their ready state

        Steps 1-2 and 5 (phase transitions) are only performed by colocated
        transports (IPC, HTTP). The NCCL collective transport skips them since
        policy and generation run on separate GPUs.

        Step 4 (verification) is performed explicitly by IPC and NCCL
        transports, which check ``update_success`` and raise on failure. The
        HTTP transport relies on ``ray.get()`` to propagate any server-side
        errors (matching the existing grpo.py behavior).

        Args:
            timer: Optional Timer for profiling individual phases.
            kv_scales: Optional KV cache scales for FP8 quantization.
                **Note**: Only honored by non-colocated transports, which
                forward them to policy-side weight send/broadcast methods. IPC
                and HTTP transports ignore this parameter.

        Raises:
            RuntimeError: If the weight transfer fails.
        """
        pass

    @property
    @abstractmethod
    def is_stale(self) -> bool:
        """Whether the generation backend's weights are out of date.

        Returns True after mark_stale() is called and before the next
        successful sync_weights() completes.
        """
        pass

    @abstractmethod
    def mark_stale(self) -> None:
        """Mark weights as stale after a training step.

        Should be called after every training step so the orchestrator
        knows a sync is needed before the next generation phase. This
        applies globally — all generation workers are considered stale
        and will be updated atomically on the next ``sync_weights()`` call.
        """
        pass

    @abstractmethod
    def init_communicator(self) -> None:
        """Initialize any communication channels needed for weight transfer.

        Called once during setup, after policy and generation workers are
        constructed. For colocated IPC/HTTP transports this may prepare
        refit metadata. For NCCL collectives this initializes the
        process group.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release all communication resources."""
        pass
