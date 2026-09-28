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

"""Colocated Energon loader extension for Megatron policy workers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from typing import Any, Mapping, Optional

import ray
from megatron.core import parallel_state

from nemo_rl.algorithms.sft import prepare_sft_batch
from nemo_rl.data.energon.sft_dataloader import (
    EnergonSFTDataLoader,
    build_energon_sft_loader,
)
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data_plane.adapters.local import local_batch_to_tensordict
from nemo_rl.data_plane.schema import MICRO_BATCH_INDICES, MICRO_BATCH_LENGTHS
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class SFTMegatronPolicyWorker(MegatronPolicyWorkerImpl):
    """Megatron policy worker with an Energon loader on each DP owner."""

    def __init__(self, *args: Any, processor: Any = None, **kwargs: Any) -> None:
        self._sft_processor = processor
        self._sft_loader: Optional[EnergonSFTDataLoader] = None
        self._sft_loader_iterator: Any = None
        self._sft_active_envelope: Optional[StepEnvelope] = None
        self._sft_next_batch_index = 0
        self._sft_logical_rank: Optional[int] = None
        self._sft_logical_world_size: Optional[int] = None
        self._ld_on = os.environ.get("NRL_LOADDIAG") == "1"
        self._ld_phase: Optional[str] = None
        self._ld_t0 = time.monotonic()
        self._ld_durations: dict[str, float] = {}
        self._ld_watchdog: Optional[threading.Thread] = None
        super().__init__(*args, **kwargs)

    def setup_sft_dataloader(
        self,
        *,
        data_config: Mapping[str, Any],
        batch_size: int,
        max_sequence_length: int,
        placement_fingerprint: str,
        only_unmask_final: bool,
        restored_state: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Build the train loader on the TP0/PP0/CP0 rank of this DP replica."""
        if not self._is_replica_leader():
            return False
        if self._sft_loader is not None:
            raise RuntimeError("The SFT Energon loader is already configured.")
        if self._sft_processor is None:
            tokenizer_config = (self.cfg or {}).get("tokenizer")
            if tokenizer_config is None:
                raise ValueError(
                    "SFTv2 requires a multimodal processor on policy workers, and "
                    "policy.tokenizer was not available to build one locally."
                )
            from nemo_rl.algorithms.utils import get_tokenizer

            self._sft_processor = get_tokenizer(tokenizer_config, get_processor=True)

        def _field(obj: Any, key: str) -> Any:
            if obj is None:
                return None
            if isinstance(obj, Mapping):
                return obj.get(key)
            return getattr(obj, key, None)

        packing = _field(
            _field(_field(data_config, "energon"), "task_encoder"), "packing"
        )
        if packing is not None:
            cp_size = parallel_state.get_context_parallel_world_size()
            pad_multiple = _field(
                _field(packing, "options"), "sequence_length_pad_multiple"
            )
            if cp_size > 1 and pad_multiple % (2 * cp_size):
                raise ValueError(
                    "Energon packing sequence_length_pad_multiple "
                    f"({pad_multiple}) must be divisible by 2 * "
                    f"context_parallel_size ({2 * cp_size})."
                )

        logical_rank = parallel_state.get_data_parallel_rank()
        logical_world_size = parallel_state.get_data_parallel_world_size()
        self._sft_loader = build_energon_sft_loader(
            data_config=data_config,
            source=data_config["train"],
            processor=self._sft_processor,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            split_role="train",
            logical_rank=logical_rank,
            logical_world_size=logical_world_size,
            placement_fingerprint=placement_fingerprint,
            only_unmask_final=only_unmask_final,
        )
        if restored_state is not None:
            self._sft_loader.load_state_dict(restored_state)
        self._sft_loader_iterator = iter(self._sft_loader)
        self._sft_logical_rank = logical_rank
        self._sft_logical_world_size = logical_world_size
        return True

    def _ld_mark(self, phase: str) -> None:
        """Close the previous load phase and enter ``phase``."""
        now = time.monotonic()
        previous = self._ld_phase
        elapsed = now - self._ld_t0
        if previous is not None:
            self._ld_durations[previous] = elapsed
            if self._ld_on:
                print(
                    "[LOADDIAG] batch=%d %s done in %.3fs -> %s"
                    % (self._sft_next_batch_index, previous, elapsed, phase),
                    flush=True,
                )
        self._ld_phase = None if phase == "idle" else phase
        self._ld_t0 = now
        if self._ld_on and self._ld_watchdog is None:
            self._ld_watchdog = threading.Thread(
                target=self._ld_watch,
                name="sft-loaddiag",
                daemon=True,
            )
            self._ld_watchdog.start()

    def _ld_watch(self) -> None:
        """Print the in-flight phase every 15 seconds after a 30-second stall."""
        while True:
            time.sleep(15)
            phase = self._ld_phase
            if phase is None:
                continue
            elapsed = time.monotonic() - self._ld_t0
            if elapsed < 30:
                continue
            try:
                process_ids = os.listdir("/proc")
            except OSError:
                workers = -1
            else:
                workers = 0
                for process_id in process_ids:
                    if not process_id.isdigit():
                        continue
                    try:
                        with open(
                            f"/proc/{process_id}/comm",
                            encoding="utf-8",
                            errors="ignore",
                        ) as stream:
                            command = stream.read()
                    except OSError:
                        continue
                    if "pt_data_worker" in command:
                        workers += 1
            print(
                "[LOADDIAG] STUCK batch=%d phase=%s elapsed=%.0fs data_workers=%d"
                % (self._sft_next_batch_index, phase, elapsed, workers),
                flush=True,
            )

    def load_next_sft_batch(
        self,
        *,
        only_unmask_final: bool,
        make_sequence_length_divisible_by: int,
    ) -> StepEnvelope:
        """Load, prepare, and publish one batch into this process's local store."""
        if self._sft_loader is None or self._sft_loader_iterator is None:
            raise RuntimeError("The SFT Energon loader is not configured on this rank.")
        if self._sft_active_envelope is not None:
            raise RuntimeError(
                "Commit or abort the active SFT batch before loading again."
            )
        if self._sft_logical_rank is None or self._sft_logical_world_size is None:
            raise RuntimeError("The SFT logical loader identity is missing.")

        started = time.monotonic()
        self._ld_durations = {}
        self._ld_phase = None
        self._ld_t0 = started
        self._ld_mark("iter")

        # restart when one epoch is exhausted
        try:
            batch = next(self._sft_loader_iterator)
        except StopIteration:
            self._sft_loader_iterator = iter(self._sft_loader)
            batch = next(self._sft_loader_iterator)
        self._ld_mark("prepare")

        prepared = prepare_sft_batch(
            batch,
            tokenizer=self.tokenizer,
            only_unmask_final=only_unmask_final,
            make_sequence_length_divisible_by=make_sequence_length_divisible_by,
        )
        self._ld_mark("post-prepare")
        batch_size = prepared.size
        source_ids = self._source_ids(prepared, batch_size=batch_size)
        partition_id = (
            f"sft_v2_dp{self._sft_logical_rank}_batch{self._sft_next_batch_index}"
        )
        sample_ids = [f"{partition_id}_row{row}" for row in range(batch_size)]
        # Source IDs are controller metadata carried by the envelope and tags.
        # Policy workers do not consume them, and replica broadcasts reject
        # Python containers to keep bulk payloads off the object collective.
        policy_batch = {
            key: value
            for key, value in prepared.items()
            if key not in {"source_ids", "sample_keys"}
        }
        self._ld_mark("tensordict")
        fields = local_batch_to_tensordict(policy_batch, batch_size=batch_size)

        self._ld_mark("publish")
        publish_phase_started = time.monotonic()
        field_names = list(fields.keys())
        client = self._require_dp_client()
        publish_setup = time.monotonic() - publish_phase_started

        publish_phase_started = time.monotonic()
        client.register_partition(
            partition_id=partition_id,
            fields=field_names,
            num_samples=batch_size,
            consumer_tasks=["train"],
        )
        publish_register_partition = time.monotonic() - publish_phase_started

        publish_phase_started = time.monotonic()
        tags = self._source_tags(prepared, batch_size=batch_size)
        publish_source_tags = time.monotonic() - publish_phase_started

        publish_phase_started = time.monotonic()
        published_meta = client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
        )
        publish_put_samples = time.monotonic() - publish_phase_started

        publish_phase_started = time.monotonic()
        lengths_tensor = prepared["input_lengths"]
        lengths = tuple(int(value) for value in lengths_tensor.tolist())
        sample_mask = prepared["sample_mask"]
        valid_tokens = int(
            (sample_mask.unsqueeze(-1) * prepared["token_mask"][:, 1:]).sum().item()
        )
        extra_info = dict(published_meta.extra_info)
        if "cu_seqlens" in prepared:
            extra_info[MICRO_BATCH_INDICES] = [
                [[index, index + 1] for index in range(batch_size)]
            ]
            extra_info[MICRO_BATCH_LENGTHS] = [list(lengths)]
        if make_sequence_length_divisible_by > 1:
            extra_info["pad_to_multiple"] = int(make_sequence_length_divisible_by)
        publish_batch_metadata = time.monotonic() - publish_phase_started
        self._ld_mark("idle")
        self._ld_durations.update(
            {
                "publish_setup": publish_setup,
                "publish_register_partition": publish_register_partition,
                "publish_source_tags": publish_source_tags,
                "publish_put_samples": publish_put_samples,
                "publish_batch_metadata": publish_batch_metadata,
            }
        )
        envelope = StepEnvelope(
            meta=replace(
                published_meta,
                task_name="train",
                extra_info=extra_info,
            ),
            logical_rank=self._sft_logical_rank,
            logical_world_size=self._sft_logical_world_size,
            source_ids=source_ids,
            field_names=tuple(field_names),
            sequence_lengths=lengths,
            # The controller blocks on this whole call, so include publishing.
            load_seconds=time.monotonic() - started,
            valid_tokens=valid_tokens,
            load_phase_seconds=dict(self._ld_durations),
        )
        self._sft_active_envelope = envelope
        self._sft_next_batch_index += 1
        return envelope

    def commit_sft_batch(self) -> None:
        """Release the active process-local batch after a successful step."""
        envelope = self._require_active_envelope()
        self._require_dp_client().clear_samples(
            sample_ids=envelope.meta.sample_ids,
            partition_id=envelope.meta.partition_id,
        )
        self._sft_active_envelope = None

    def abort_sft_batch(self) -> None:
        """Release the active batch after a failed policy step."""
        if self._sft_active_envelope is None:
            return
        self.commit_sft_batch()

    def sft_dataloader_state_dict(self) -> dict[str, Any]:
        """Capture this logical loader state after its batch is committed."""
        if self._sft_loader is None:
            raise RuntimeError("The SFT Energon loader is not configured on this rank.")
        if self._sft_active_envelope is not None:
            raise RuntimeError("Cannot checkpoint an uncommitted SFT batch.")
        return self._sft_loader.state_dict()

    def close_sft_dataloader(self) -> None:
        """Clear local batch state and release the loader reference."""
        self.abort_sft_batch()
        self._sft_loader_iterator = None
        self._sft_loader = None

    def _require_active_envelope(self) -> StepEnvelope:
        if self._sft_active_envelope is None:
            raise RuntimeError("There is no active SFT batch to commit.")
        return self._sft_active_envelope

    @staticmethod
    def _source_ids(batch: Mapping[str, Any], *, batch_size: int) -> tuple[str, ...]:
        for key in ("source_ids", "sample_keys"):
            values = batch.get(key)
            if isinstance(values, (list, tuple)) and len(values) == batch_size:
                return tuple(
                    str(source_id)
                    for value in values
                    for source_id in (
                        value if isinstance(value, (list, tuple)) else [value]
                    )
                )
        return tuple(f"unknown:{row}" for row in range(batch_size))

    @staticmethod
    def _source_tags(
        batch: Mapping[str, Any], *, batch_size: int
    ) -> list[dict[str, Any]]:
        values = batch.get("source_ids")
        if not isinstance(values, (list, tuple)) or len(values) != batch_size:
            return [{"source_id": f"unknown:{row}"} for row in range(batch_size)]
        return [
            (
                {"source_ids": [str(source_id) for source_id in value]}
                if isinstance(value, (list, tuple))
                else {"source_id": str(value)}
            )
            for value in values
        ]


__all__ = ["SFTMegatronPolicyWorker", "StepEnvelope"]
