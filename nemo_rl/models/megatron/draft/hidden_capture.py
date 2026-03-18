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

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import ContextManager, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.training.utils import unwrap_model
from torch import Tensor, nn


def get_eagle3_aux_hidden_state_layers(num_layers: int) -> tuple[int, ...]:
    """Pick the default auxiliary policy layers whose activations feed Eagle training."""
    candidate_indices = (
        1,
        max(0, num_layers // 2 - 1),
        max(1, num_layers - 4),
    )
    valid_indices = sorted(set(candidate_indices))
    return tuple(valid_indices)


_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}

_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


@dataclass
class CapturedStates:
    """Container for hidden states captured from the policy model."""

    hidden_states: Optional[Tensor] = None
    inputs_embeds: Optional[Tensor] = None


class HiddenStateCapture:
    """Capture policy embeddings and auxiliary hidden states for Eagle training."""

    def __init__(
        self,
        model: nn.Module,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
    ):
        self.model = unwrap_model(model)
        self.num_layers = self.model.config.num_layers

        self.aux_layer_indices = (
            aux_layer_indices
            if aux_layer_indices is not None
            else get_eagle3_aux_hidden_state_layers(self.num_layers)
        )

        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.is_first_stage = parallel_state.is_pipeline_first_stage()
        self.is_last_stage = parallel_state.is_pipeline_last_stage()

        self._global_to_local: Dict[int, int] = {}
        self._local_aux_indices: List[int] = []
        self._compute_local_layer_mapping()
        self._layer_owner_by_global_idx = self._compute_layer_owner_map()

        self._captured: Dict[str, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _compute_local_layer_mapping(self) -> None:
        for local_idx, layer in enumerate(self.model.decoder.layers):
            global_idx = int(layer.layer_number) - 1
            if global_idx in self.aux_layer_indices:
                self._global_to_local[global_idx] = local_idx
                self._local_aux_indices.append(local_idx)

    def _compute_layer_owner_map(self) -> Dict[int, int]:
        if self.pp_size == 1 or not dist.is_initialized():
            return {layer_idx: 0 for layer_idx in range(self.num_layers)}

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        local_owner_mask = torch.zeros(
            self.num_layers,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        for layer in self.model.decoder.layers:
            global_idx = int(layer.layer_number) - 1
            if 0 <= global_idx < self.num_layers:
                local_owner_mask[global_idx] = 1

        gathered_owner_masks = [
            torch.zeros_like(local_owner_mask) for _ in range(self.pp_size)
        ]
        dist.all_gather(gathered_owner_masks, local_owner_mask, group=pp_group)

        owner_map: Dict[int, int] = {}
        for global_idx in range(self.num_layers):
            for rank_idx, owner_mask in enumerate(gathered_owner_masks):
                if int(owner_mask[global_idx].item()) == 1:
                    owner_map[global_idx] = rank_idx
                    break
        return owner_map

    def _make_layer_output_hook(self, global_idx: int):
        def hook(_module, _args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states is None:
                return
            self._captured[f"layer_{global_idx}"] = hidden_states.detach().clone()

        return hook

    def _make_embedding_hook(self):
        def hook(_module, _args, output):
            self._captured["embeds"] = output.detach().clone()

        return hook

    def register_hooks(self) -> None:
        self.clear_hooks()
        self._captured.clear()

        if self.is_first_stage and hasattr(self.model, "embedding"):
            self._hooks.append(
                self.model.embedding.register_forward_hook(self._make_embedding_hook())
            )

        for local_idx in self._local_aux_indices:
            layer = self.model.decoder.layers[local_idx]
            global_idx = int(layer.layer_number) - 1
            self._hooks.append(
                layer.register_forward_hook(self._make_layer_output_hook(global_idx))
            )

    def clear_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @contextmanager
    def capture_context(self):
        try:
            self.register_hooks()
            yield self
        finally:
            self.clear_hooks()

    def _assemble_local_states(self) -> CapturedStates:
        embeds = self._captured.get("embeds")

        hidden_chunks = []
        for global_idx in sorted(self.aux_layer_indices):
            tensor = self._captured.get(f"layer_{global_idx}")
            if tensor is not None:
                hidden_chunks.append(tensor)

        if not hidden_chunks:
            return CapturedStates(hidden_states=None, inputs_embeds=embeds)

        return CapturedStates(
            hidden_states=torch.cat(hidden_chunks, dim=-1),
            inputs_embeds=embeds,
        )

    def _owner_rank_for_global_layer(self, global_layer_idx: int) -> int:
        if self.pp_size == 1:
            return 0
        if global_layer_idx in self._layer_owner_by_global_idx:
            return self._layer_owner_by_global_idx[global_layer_idx]
        layers_per_rank = max(1, self.num_layers // self.pp_size)
        return min(global_layer_idx // layers_per_rank, self.pp_size - 1)

    @staticmethod
    def _send_tensor(
        tensor: Tensor,
        dst_rank: int,
        group: dist.ProcessGroup,
    ) -> None:
        dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
        if dtype_code is None:
            raise ValueError(f"Unsupported tensor dtype for send/recv: {tensor.dtype}")

        metadata = torch.tensor(
            [tensor.shape[0], tensor.shape[1], tensor.shape[2], dtype_code],
            dtype=torch.int64,
            device=tensor.device,
        )
        dist.send(metadata, dst=dst_rank, group=group)
        dist.send(tensor.contiguous(), dst=dst_rank, group=group)

    @staticmethod
    def _recv_tensor(
        src_rank: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> Tensor:
        metadata = torch.empty(4, dtype=torch.int64, device=device)
        dist.recv(metadata, src=src_rank, group=group)
        seq_len, batch_size, hidden_size, dtype_code = [
            int(x) for x in metadata.tolist()
        ]
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(
                f"Unsupported tensor dtype code in send/recv: {dtype_code}"
            )

        received = torch.empty(
            seq_len,
            batch_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        dist.recv(received, src=src_rank, group=group)
        return received

    def _gather_distributed(self) -> CapturedStates:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        last_rank = self.pp_size - 1
        recv_device = torch.device("cuda", torch.cuda.current_device())

        sample_tensor = None
        for tensor in self._captured.values():
            if tensor is not None:
                sample_tensor = tensor
                break

        if sample_tensor is None and not self.is_last_stage:
            return CapturedStates()

        gathered_hidden_by_layer: Dict[int, Tensor] = {}

        for global_idx in self.aux_layer_indices:
            owner_rank = self._owner_rank_for_global_layer(global_idx)
            key = f"layer_{global_idx}"

            if self.pp_rank == owner_rank:
                layer_tensor = self._captured.get(key)
                if layer_tensor is None:
                    continue
                if self.is_last_stage:
                    gathered_hidden_by_layer[global_idx] = layer_tensor
                else:
                    self._send_tensor(layer_tensor, dst_rank=last_rank, group=pp_group)
            elif self.is_last_stage:
                received = self._recv_tensor(
                    src_rank=owner_rank,
                    group=pp_group,
                    device=recv_device,
                )
                gathered_hidden_by_layer[global_idx] = received

        gathered_embeds = None
        if self.is_first_stage:
            embeds = self._captured.get("embeds")
            if embeds is not None:
                if self.is_last_stage:
                    gathered_embeds = embeds
                else:
                    self._send_tensor(embeds, dst_rank=last_rank, group=pp_group)
        elif self.is_last_stage:
            gathered_embeds = self._recv_tensor(
                src_rank=0,
                group=pp_group,
                device=recv_device,
            )

        if not self.is_last_stage:
            return CapturedStates()

        if gathered_hidden_by_layer:
            hidden_states = torch.cat(
                [
                    gathered_hidden_by_layer[layer]
                    for layer in sorted(gathered_hidden_by_layer.keys())
                ],
                dim=-1,
            )
        else:
            hidden_states = None

        return CapturedStates(
            hidden_states=hidden_states,
            inputs_embeds=gathered_embeds,
        )

    def get_captured_states(self) -> CapturedStates:
        if self.pp_size == 1:
            return self._assemble_local_states()
        return self._gather_distributed()


def get_capture_context(
    model: nn.Module,
    draft_config: Optional[Dict] = None,
    aux_layer_indices: Optional[Tuple[int, ...]] = None,
) -> Tuple[ContextManager, Optional[HiddenStateCapture]]:
    """Return a no-op context unless draft training needs hidden-state capture for this step."""
    if not isinstance(draft_config, dict) or not draft_config.get("enabled", False):
        return nullcontext(), None
    capture = HiddenStateCapture(
        model=model,
        aux_layer_indices=aux_layer_indices,
    )
    return capture.capture_context(), capture
