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
"""Forward-hook capture of the policy's per-layer hidden states for the draft."""

from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor


def _resolve_layers(
    policy_model: nn.Module,
) -> tuple[nn.Module, Any]:
    """Locate the decoder backbone (embedding owner) and its layer container.

    The container is an nn.ModuleList for HF-style stacks or an nn.ModuleDict
    keyed by stringified layer index for Automodel custom backbones (e.g.
    Qwen3_5Moe); use ``_layer_module`` to index either uniformly.
    """
    candidates = [getattr(policy_model, "model", None), policy_model]
    for base in candidates:
        if base is None:
            continue
        layers = getattr(base, "layers", None)
        if layers is not None:
            return base, layers
    raise ValueError(
        "DSpark hidden capture could not locate `.layers` on the policy "
        f"model (searched {type(policy_model).__name__}). Only HF-style "
        "decoder stacks are supported for DSpark co-training."
    )


def _layer_module(layers: Any, layer_id: int) -> nn.Module:
    if isinstance(layers, nn.ModuleDict):
        return layers[str(layer_id)]
    return layers[layer_id]


def _hook_output_tensor(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if isinstance(hidden, DTensor):
        # Under TP the layer output can be sharded or partial; materialize the
        # replicated full tensor (a no-op collective when already replicated).
        if any(not p.is_replicate() for p in hidden.placements):
            hidden = hidden.full_tensor()
        else:
            hidden = hidden.to_local()
    return hidden.detach()


class DSparkHiddenCapture:
    """Forward hooks capturing the policy's per-layer hiddens for the draft.

    Captured tensors are detached: the DSpark loss never backprops into the
    policy trunk. The hooks are "armed" by a pre-forward hook on the policy
    root and disarmed by ``collect()``: activation checkpointing replays layer
    forwards during backward (after the loss consumed the capture), and the
    disarmed hooks turn those replays into no-ops instead of re-running
    capture work (and, under TP, its collectives).
    """

    def __init__(self, policy_model: nn.Module, target_layer_ids: list[int]):
        self.target_layer_ids = [int(i) for i in target_layer_ids]
        self._policy_root = policy_model
        base, layers = _resolve_layers(policy_model)
        num_layers = len(layers)
        for layer_id in self.target_layer_ids:
            if layer_id != -1 and not (0 <= layer_id < num_layers):
                raise ValueError(
                    f"target_layer_id {layer_id} out of range for a policy with "
                    f"{num_layers} decoder layers."
                )
        self._modules_by_id: dict[int, nn.Module] = {}
        for layer_id in self.target_layer_ids:
            if layer_id == -1:
                embed = getattr(base, "embed_tokens", None)
                if embed is None:
                    raise ValueError(
                        "target_layer_ids includes -1 (embedding output) but the "
                        "policy model has no `.embed_tokens`."
                    )
                self._modules_by_id[layer_id] = embed
            else:
                self._modules_by_id[layer_id] = _layer_module(layers, layer_id)
        self._handles: list[Any] = []
        self._captured: dict[int, torch.Tensor] = {}
        self._armed = False

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def activate(self) -> None:
        if self._handles:
            return

        def arm_hook(_module: nn.Module, _args: Any, _kwargs: Any) -> None:
            self._armed = True

        def make_layer_hook(layer_id: int):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                if self._armed:
                    self._captured[layer_id] = _hook_output_tensor(output)

            return hook

        # The root pre-hook re-arms capture at each microbatch forward; the
        # checkpointed backward replay calls only layer forwards, so capture
        # stays disarmed there.
        self._handles.append(
            self._policy_root.register_forward_pre_hook(arm_hook, with_kwargs=True)
        )
        for layer_id, module in self._modules_by_id.items():
            self._handles.append(
                module.register_forward_hook(make_layer_hook(layer_id))
            )

    def deactivate(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.clear()

    def clear(self) -> None:
        self._captured = {}
        self._armed = False

    def collect(self) -> torch.Tensor:
        """Concatenated target hidden states captured by the layer hooks."""
        missing = [i for i in self.target_layer_ids if i not in self._captured]
        if missing:
            raise RuntimeError(
                "DSpark hidden capture did not observe the policy forward "
                f"(missing layer ids {missing}). The capture hooks must be "
                "active during the training forward."
            )
        self._armed = False
        return torch.cat([self._captured[i] for i in self.target_layer_ids], dim=-1)
