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

"""Selected-token log-probabilities from an AutoModel policy's LM head.

:class:`FusedLogprobLMHead` replaces the LM head's ``forward`` on one model
instance. Inside :meth:`FusedLogprobLMHead.capture`, the head computes the
next-token log-probabilities with
:func:`~nemo_rl.distributed.fused_linear_logprobs.linear_selected_logprobs`
instead of returning ``[batch, seq, vocab]`` logits; outside it, the original
``forward`` runs unchanged.

The computation happens inside the head's own ``forward`` so that FSDP2 keeps
managing the weight: its pre-forward hook unshards the weight, its
pre-backward hook unshards it again for the recomputation, and its
post-backward hook reduce-scatters the weight gradient. The capture reads the
result directly, so whatever the model does with the head's return value
afterwards does not change it.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_rl.distributed.fused_linear_logprobs import linear_selected_logprobs


@dataclass
class SelectedLogprobsCapture:
    """Next-token log-probabilities recorded by one :meth:`FusedLogprobLMHead.capture`.

    Attributes:
        input_ids: Token ids ``[batch, seq]`` fed to the model.
        temperature: Sampling temperature dividing the float32 logits.
        logprobs: Float32 ``[batch, seq]`` log-probabilities of
            ``input_ids.roll(-1, dims=1)``, set when the LM head runs. The last
            position scores a wrapped-around token and is dropped by
            :attr:`next_token_logprobs`.
    """

    input_ids: torch.Tensor
    temperature: float
    logprobs: Optional[torch.Tensor] = field(default=None, init=False)

    @property
    def next_token_logprobs(self) -> torch.Tensor:
        """Float32 ``[batch, seq - 1]`` log-probabilities of ``input_ids[:, 1:]``."""
        if self.logprobs is None:
            raise RuntimeError(
                "The LM head did not run inside FusedLogprobLMHead.capture()."
            )
        return self.logprobs[:, :-1]


class FusedLogprobLMHead:
    """Computes next-token log-probabilities inside a model's LM head.

    Args:
        model: Causal LM whose ``get_output_embeddings()`` returns a bias-free
            ``nn.Linear`` (FSDP2-wrapped or not) that is not tensor-parallel.
        chunk_size: Tokens per chunk for the projection, normalization and
            backward.

    Raises:
        ValueError: If the model has no linear LM head, the head has a bias or
            a custom ``forward`` (for example a LoRA adapter), the model
            transforms its logits after the head, or ``chunk_size`` is not a
            positive int.
    """

    def __init__(self, model: nn.Module, *, chunk_size: int) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise ValueError(f"chunk_size must be an int; got {chunk_size!r}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}.")
        _reject_logit_post_processing(model)

        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
        if not isinstance(lm_head, nn.Linear):
            raise ValueError(
                "Fused linear logprobs need the model's output embeddings to be an "
                f"nn.Linear LM head; got {type(lm_head).__name__}."
            )
        # FSDP2 swaps in a subclass that inherits nn.Linear.forward; a subclass
        # that overrides forward (LoRA, quantization) computes something else.
        if type(lm_head).forward is not nn.Linear.forward:
            raise ValueError(
                "Fused linear logprobs recompute hidden_states @ weight.T, which "
                f"would bypass {type(lm_head).__name__}.forward."
            )
        if lm_head.bias is not None:
            raise ValueError("Fused linear logprobs do not support an LM-head bias.")

        self._lm_head = lm_head
        self._chunk_size = chunk_size
        self._original_forward: Callable[[torch.Tensor], torch.Tensor] = lm_head.forward
        self._active: Optional[SelectedLogprobsCapture] = None
        # pyrefly: ignore[bad-assignment]  Instance-level override of nn.Linear.forward.
        lm_head.forward = self._forward

    @contextmanager
    def capture(
        self, input_ids: torch.Tensor, *, temperature: float
    ) -> Iterator[SelectedLogprobsCapture]:
        """Make the next LM-head call compute log-probabilities of ``input_ids[:, 1:]``.

        Args:
            input_ids: Token ids ``[batch, seq]``, aligned with the hidden
                states the LM head receives.
            temperature: Sampling temperature dividing the float32 logits.

        Yields:
            The capture, whose log-probabilities are set once the model's
            forward has called the LM head.
        """
        if self._active is not None:
            raise RuntimeError("FusedLogprobLMHead.capture() does not nest.")
        capture = SelectedLogprobsCapture(input_ids=input_ids, temperature=temperature)
        self._active = capture
        try:
            yield capture
        finally:
            self._active = None

    def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        capture = self._active
        if capture is None:
            return self._original_forward(hidden_states)
        if capture.logprobs is not None:
            raise RuntimeError(
                "The LM head ran twice inside one FusedLogprobLMHead.capture()."
            )
        weight = self._lm_head.weight
        if isinstance(weight, DTensor):
            raise RuntimeError(
                "Fused linear logprobs need an unsharded LM-head weight inside the "
                f"head's forward; got a DTensor with placements {weight.placements} "
                "(tensor parallelism is not supported)."
            )
        if hidden_states.shape[:-1] != capture.input_ids.shape:
            raise ValueError(
                f"The LM head received hidden states {tuple(hidden_states.shape)}, "
                f"which do not align with input_ids {tuple(capture.input_ids.shape)}."
            )

        # Match nn.Linear's dtype under autocast so both paths compute the same logits.
        device_type = hidden_states.device.type
        compute_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else weight.dtype
        )
        with torch.autocast(device_type=device_type, enabled=False):
            logprobs = linear_selected_logprobs(
                hidden_states.to(compute_dtype),
                weight.to(compute_dtype),
                capture.input_ids.roll(shifts=-1, dims=1),
                chunk_size=self._chunk_size,
                temperature=capture.temperature,
            )
        capture.logprobs = logprobs
        return logprobs


def _reject_logit_post_processing(model: nn.Module) -> None:
    """Raise if the model transforms its logits after the LM head."""
    config = getattr(model, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        if (
            getattr(cfg, "final_logit_softcapping", None)
            or getattr(cfg, "output_multiplier", 1.0) != 1.0
            or getattr(cfg, "use_mup", False)
        ):
            raise ValueError(
                "Fused linear logprobs compute log_softmax(lm_head(h)), but this "
                "model post-processes its logits after the LM head "
                "(final_logit_softcapping / output_multiplier / MuP)."
            )
