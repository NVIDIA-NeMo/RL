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
"""Request-owned GPU payloads beside Gym's CPU digest and serving records.

The CPU mirror remains authoritative for Gym validation. These tensors are
the original generated data, retained until the same acknowledged staging
PUT, rather than reconstructed on CUDA from that mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import StagedCallRecord, StageResult

    from nemo_rl.data_plane.tq_token_sink import TQTokenSink


@dataclass(frozen=True, kw_only=True)
class GpuTokenPayload:
    """Stable GPU allocations retained for one complete model response.

    Args:
        prompt_len: Full engine prompt length, including a continuation prefix.
        generated_token_ids: One-dimensional generated CUDA IDs in int64 format.
        generated_logprobs: Corresponding selected-token CUDA logprobs in float32.
        routed_experts: Optional full aligned CUDA routes, including the prompt.

    The producer must own these allocations and stop mutating them
    before binding; references alone do not protect reusable engine buffers.
    """

    prompt_len: int
    generated_token_ids: torch.Tensor
    generated_logprobs: torch.Tensor
    routed_experts: torch.Tensor | None = None

    def device(self) -> torch.device:
        """Validate CUDA ownership and return the common source device."""
        if type(self.prompt_len) is not int or self.prompt_len < 0:
            raise ValueError("GPU payload prompt_len must be a non-negative int")
        tensors = [self.generated_token_ids, self.generated_logprobs]
        if self.routed_experts is not None:
            tensors.append(self.routed_experts)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("GPU payload must contain original CUDA tensors")
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("GPU payload tensors must use the same CUDA device")
        return device

    def staging_fields(
        self,
        record: StagedCallRecord,
        cpu_fields: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build device fields with the same layout as Gym's committed mirror.

        This step copies CPU-origin prompt-carry IDs H2D. Generated values use
        retained CUDA storage; masks and prompt logprobs are constructed on CUDA.
        Routes view the assembled GPU payload, which the host has already filled
        with any missing cached-prefix rows from vLLM's CPU router history.
        Shape and dtype checks do not transfer payloads to CPU.
        """
        device = self.device()
        carry_len = self.prompt_len - record.prev_len
        generated_len = record.delta_len - carry_len
        if carry_len < 0 or generated_len < 0:
            raise ValueError("GPU payload prompt length does not match staged delta")
        expected_mask = [0.0] * carry_len + [1.0] * generated_len
        if not torch.equal(
            cpu_fields["token_mask_delta"][0].view(torch.int32),
            torch.tensor(expected_mask, dtype=torch.float32).view(torch.int32),
        ):
            raise ValueError(
                "GPU payload prompt/generation split does not match staged mask"
            )
        if torch.count_nonzero(
            cpu_fields["generation_logprobs_delta"][0, :carry_len].view(torch.int32)
        ).item():
            raise ValueError(
                "GPU prompt logprobs must match the zero-valued CPU mirror"
            )
        if ROUTED_EXPERTS_FIELD in cpu_fields and self.routed_experts is None:
            raise ValueError("GPU payload is missing the committed routed experts")

        ids = self.generated_token_ids.detach()
        logprobs = self.generated_logprobs.detach()
        if ids.ndim != 1 or logprobs.ndim != 1:
            raise ValueError(
                "GPU generated token IDs and logprobs must be one-dimensional"
            )
        if ids.numel() != generated_len or logprobs.numel() != generated_len:
            raise ValueError("GPU generated payload length does not match staged delta")
        if ids.dtype != torch.int64 or logprobs.dtype != torch.float32:
            raise ValueError("GPU generated IDs/logprobs must already be int64/float32")
        with torch.cuda.device(device):
            if carry_len:
                full_ids = torch.empty(
                    record.delta_len, dtype=torch.int64, device=device
                )
                full_ids[:carry_len].copy_(cpu_fields["token_ids_delta"][0, :carry_len])
                full_ids[carry_len:].copy_(ids)
                full_logprobs = torch.zeros(
                    record.delta_len, dtype=torch.float32, device=device
                )
                full_logprobs[carry_len:].copy_(logprobs)
            else:
                full_ids, full_logprobs = ids, logprobs
            mask = torch.zeros(record.delta_len, dtype=torch.float32, device=device)
            mask[carry_len:] = 1.0
            fields = {
                "token_ids_delta": full_ids.unsqueeze(0),
                "token_mask_delta": mask.unsqueeze(0),
                "generation_logprobs_delta": full_logprobs.unsqueeze(0),
            }

            if self.routed_experts is not None:
                routes = self.routed_experts
                expected_routes = cpu_fields.get(ROUTED_EXPERTS_FIELD)
                if expected_routes is None:
                    raise ValueError("GPU routes have no committed CPU extras mirror")
                if routes.ndim != 3 or routes.shape[0] != record.cum_len:
                    raise ValueError(
                        "GPU routes must cover the complete engine sequence"
                    )
                if routes.dtype != expected_routes.dtype:
                    raise ValueError(
                        "GPU routes dtype does not match committed CPU mirror"
                    )
                delta_routes = routes.detach()[record.prev_len :]
                if delta_routes.shape != expected_routes[0].shape:
                    raise ValueError(
                        "GPU routes shape does not match committed CPU mirror"
                    )
                fields[ROUTED_EXPERTS_FIELD] = delta_routes.unsqueeze(0)
        return fields


class BoundGpuTokenSink:
    """Hold one call's optional GPU optimization; Gym owns its completion."""

    def __init__(self, sink: TQTokenSink) -> None:
        self._sink = sink
        self._payload: GpuTokenPayload | None = None

    def bind(self, payload: GpuTokenPayload) -> None:
        """Retain this request's payload until staging completes or aborts."""
        self._payload = payload

    def stage(self, record: StagedCallRecord) -> StageResult:
        """Use GPU data when available, otherwise preserve ordinary CPU staging."""
        payload = self._payload
        try:
            if payload is None:
                return self._sink.stage(record)
            return self._sink.stage(record, gpu_payload=payload)
        finally:
            self.clear()

    def clear(self) -> None:
        """Release a payload after staging or an aborted capture."""
        self._payload = None
