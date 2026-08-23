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

Transport-specific implementations (IPC/ZMQ, HTTP, NCCL collectives, checkpoint
engines) each
encapsulate the transfer lifecycle, so algorithm code never branches on
backend type.

Colocated transports (IPC, HTTP) own GPU phase transitions internally
(offload, prepare_for_generation, restore) as part of their sync_weights()
implementation. The NCCL collective transport is a pure data mover; the
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
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nemo_rl.utils.timer import Timer


@dataclass(frozen=True, slots=True)
class WeightSyncSelection:
    """Components included in one policy-to-generation weight transfer."""

    target: bool = True
    draft: bool = True

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target policy must synchronize on every policy step")


@dataclass(frozen=True, slots=True)
class DraftApplyRequest:
    """Immutable serving-draft snapshot identity for one refit."""

    version: int
    snapshot_path: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0:
            raise ValueError("draft apply version must be a nonnegative integer")
        path = Path(self.snapshot_path)
        if not path.is_absolute():
            raise ValueError("draft apply snapshot path must be absolute")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("draft apply snapshot SHA256 must be lowercase hex")
        if not path.is_file():
            raise ValueError("draft apply snapshot must be an existing file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("draft apply snapshot SHA256 does not match its bytes")

    def receipt(self) -> Mapping[str, object]:
        """Revalidate the snapshot and bind it to a successful apply."""
        path = Path(self.snapshot_path)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError("draft apply snapshot changed or disappeared") from error
        if digest != self.sha256:
            raise RuntimeError("draft apply snapshot changed during weight transfer")
        return {
            "successful": True,
            "version": self.version,
            "snapshot_path": self.snapshot_path,
            "sha256": self.sha256,
        }


class WeightSynchronizer(ABC):
    """Abstract base class for weight synchronization between policy and generation.

    Implementations handle the weight transfer for a specific transport
    mechanism (ZMQ IPC, HTTP, NCCL collectives). The orchestrator calls
    sync_weights() without knowing which transport is being used or
    whether components are colocated; per-step staleness bookkeeping is
    owned by the training loop.

    Colocated transports (IPC, HTTP) own phase transitions internally
    (offload_before_refit, prepare_for_generation, offload_after_refit).
    Non-colocated collective and checkpoint-engine transports are pure data movers;
    the orchestrator handles phases externally.
    """

    @abstractmethod
    def sync_weights(
        self,
        *,
        selection: WeightSyncSelection = WeightSyncSelection(),
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
        draft_apply_request: DraftApplyRequest | None = None,
    ) -> Mapping[str, object] | None:
        """Transfer the latest policy weights to the generation backend.

        This method encapsulates the full sync lifecycle:
        1. Prepare the policy side (e.g., offload optimizer state to free GPU memory)
        2. Prepare the generation side (e.g., allocate weight buffers)
        3. Transfer weights via the transport mechanism
        4. Verify the transfer succeeded
        5. Restore both sides to their ready state

        Steps 1-2 and 5 (phase transitions) are only performed by colocated
        transports (IPC, HTTP). Non-colocated collective and checkpoint-engine
        transports skip them since policy and generation run on separate GPUs.

        Step 4 (verification) is performed explicitly by IPC and NCCL
        transports, which check ``update_success`` and raise on failure. The
        HTTP transport relies on ``ray.get()`` to propagate any server-side
        errors (matching the existing grpo.py behavior).

        Args:
            timer: Optional Timer for profiling individual phases.
            kv_scales: Optional KV cache scales for FP8 quantization.
                Honored by the IPC/ZMQ and NCCL collective transports. The
                HTTP transport ignores this parameter.
            draft_apply_request: Optional immutable snapshot binding. A capable
                transport emits a nested apply receipt only after the selected
                draft transfer succeeds and the snapshot digest is unchanged.

        Returns:
            Transport status and optional transport-specific metrics/receipts.
            Legacy transports may return ``None``; transports advertising
            draft-apply receipts always return a mapping.

        Raises:
            RuntimeError: If the weight transfer fails.
        """
        pass

    @property
    def supports_component_selection(self) -> bool:
        """Whether this transport can omit the draft component safely."""
        return False

    @property
    def supports_draft_apply_receipts(self) -> bool:
        """Whether this transport can bind a successful draft apply to a snapshot."""
        return False

    def validate_selection(self, selection: WeightSyncSelection) -> None:
        if not selection.draft and not self.supports_component_selection:
            raise ValueError(
                "component-selective draft refit is unsupported by "
                f"{type(self).__name__}"
            )

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
        constructed. For colocated IPC/HTTP transports this may prepare
        refit metadata. For NCCL collectives this initializes the
        process group.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release all communication resources."""
        pass


def require_component_selection(
    synchronizer: WeightSynchronizer, schedule_mode: str
) -> None:
    """Reject sparse cadence on a transport that cannot omit draft bytes."""
    if schedule_mode != "always" and not synchronizer.supports_component_selection:
        raise ValueError(
            "component-selective draft refit is unsupported by "
            f"{type(synchronizer).__name__}; use update_schedule.mode=always"
        )


def preflight_component_selection(
    *,
    schedule_mode: str,
    generation_backend: str,
    colocated: bool,
    refit_transport: str | None,
    remote_sparse: bool,
) -> None:
    """Fail before worker construction for a known-incompatible transport."""
    if schedule_mode == "always":
        return
    supported = (
        generation_backend == "vllm" and not remote_sparse and refit_transport is None
    )
    if not supported:
        raise ValueError(
            "component-selective draft refit is unsupported by the resolved "
            f"transport: backend={generation_backend!r}, colocated={colocated}, "
            f"refit_transport={refit_transport!r}, remote_sparse={remote_sparse}"
        )
