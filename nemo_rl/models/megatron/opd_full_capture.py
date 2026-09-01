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

"""Capture the pre-LM-head hidden states of a frozen MOPD teacher.

Full-vocabulary MOPD ships the teacher's final hidden states to the student,
which projects them with a teacher LM-head shard loaded from the teacher
checkpoint. Moving ``hidden_size`` floats per token instead of ``vocab_size``
keeps the payload roughly two orders of magnitude smaller and leaves teacher and
student parallelism fully decoupled.

The capture is a forward pre-hook on ``output_layer``. ``GPTModel.forward`` ends
in ``logits, _ = self.output_layer(hidden_states, ...)``, so the hook's first
positional argument is exactly the post-``final_layernorm`` hidden state. Hooking
``decoder`` instead would miss ``final_layernorm`` (it lives inside
``TransformerBlock``), and ``decoder.final_layernorm`` is absent under some
transformer specs.
"""

from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Iterator, Optional

import torch
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model
from torch import Tensor, nn


class OutputLayerInputCapture:
    """Capture the tensor fed into a Megatron model's ``output_layer``.

    The captured tensor is ``[S_local, B, H]`` in Megatron's sequence-first
    layout. Sequence parallelism shards ``S_local`` across the tensor-parallel
    group, so :meth:`get_hidden_states` gathers it back before returning.
    """

    def __init__(self, model: nn.Module):
        self.model = unwrap_model(model)
        self._output_layer = getattr(self.model, "output_layer", None)
        if self._output_layer is None:
            language_model = getattr(self.model, "language_model", None)
            if language_model is not None:
                self._output_layer = getattr(language_model, "output_layer", None)
        self._captured: Optional[Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    @property
    def is_available(self) -> bool:
        """Whether this rank owns an ``output_layer`` to hook (last PP stage)."""
        return self._output_layer is not None

    def _pre_hook(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        if args:
            self._captured = args[0]

    @contextmanager
    def capture_context(self) -> Iterator[None]:
        """Register the pre-hook for the duration of one forward pass."""
        self._captured = None
        if self._output_layer is None:
            yield
            return
        self._hook_handle = self._output_layer.register_forward_pre_hook(self._pre_hook)
        try:
            yield
        finally:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_hidden_states(self) -> Optional[Tensor]:
        """Return the captured hidden states in full ``[S, B, H]`` layout.

        Returns:
            The detached hidden states with sequence parallelism undone, or
            ``None`` when this rank ran no output layer (non-final pipeline
            stage) or the forward never reached it.

        Raises:
            ValueError: If the captured tensor is not rank 3.
        """
        if self._captured is None:
            return None
        hidden_states = self._captured
        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected pre-LM-head hidden states of rank 3 [S, B, H]; got "
                f"{tuple(hidden_states.shape)}."
            )
        if getattr(self._output_layer, "sequence_parallel", False):
            # Sequence parallelism hands output_layer an [S/TP, B, H] shard and
            # gathers internally, so the pre-hook sees the shard. Undo it here or
            # every TP>1 run silently mismatches the sequence dimension.
            from megatron.core.tensor_parallel import (
                gather_from_sequence_parallel_region,
            )

            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=parallel_state.get_tensor_model_parallel_group()
            )
        return hidden_states.detach()


def get_opd_full_capture_context(
    model: nn.Module, enabled: bool
) -> tuple[ContextManager[Any], Optional[OutputLayerInputCapture]]:
    """Build the capture context for one teacher forward pass.

    Args:
        model: The (possibly wrapped) Megatron model about to run forward.
        enabled: Whether full-vocabulary hidden-state capture is requested.

    Returns:
        Tuple of the context manager to wrap the forward in, and the capture
        object to read afterwards (``None`` when disabled or unavailable).
    """
    if not enabled:
        return nullcontext(), None
    capture = OutputLayerInputCapture(model)
    if not capture.is_available:
        # Non-final pipeline stages have no output layer; the last stage writes
        # the payload, so there is nothing for this rank to capture.
        return nullcontext(), None
    return capture.capture_context(), capture
