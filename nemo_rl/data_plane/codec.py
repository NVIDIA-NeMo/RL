# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Wire <-> trainer codec — jagged-on-the-wire bridge.

* Writer side: variable-length fields are encoded as
``torch.nested.nested_tensor`` with ``layout=torch.jagged`` before
``kv_batch_put``. Padding tax is paid only when a consumer needs a
rectangular tensor.

* Reader side: :func:`materialize` accepts the wire TensorDict and,
when ``layout='padded'``, calls
:func:`torch.nested.to_padded_tensor` on any nested leaves using
the per-field padding value supplied in ``pad_value_dict``. Trainer
code consumes the padded BatchedDataDict unchanged.

* Worker write-backs that produce ``response``-shaped outputs use
:func:`response_from_nested` to extract the response slice from a
(prompt+response) nested tensor.

* Non-tensor object fields ride as ``NonTensorStack`` / ``NonTensorData``
leaves (TQ-native passthrough). :func:`materialize` decodes them back
to ``np.ndarray(dtype=object)`` for the trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.schema import Layout

if TYPE_CHECKING:
    # Type-only import. At runtime, BatchedDataDict is loaded lazily
    # inside materialize() — see comment there for rationale.
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


# ── Padded ↔ nested helpers ───────────────────────────────────────────


def to_nested_by_length(
    padded: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Strip right-padding off a rectangular tensor using per-row lengths.

    Used by the producer side: convert
    :func:`batched_message_log_to_flat_message` output (already padded)
    into the wire format before ``kv_batch_put``.

    Args:
        padded: Rectangular tensor of shape ``(N, S, ...)``.
        lengths: Per-row valid lengths, shape ``(N,)``. CUDA tensors are
            moved to CPU once to avoid per-row syncs.

    Returns:
        A ``torch.jagged`` nested tensor whose i-th row is
        ``padded[i, :lengths[i], ...]``.
    """
    if padded.dim() < 2:
        raise ValueError(
            f"to_nested_by_length expects (N, S, ...); got shape {tuple(padded.shape)}"
        )
    n = padded.shape[0]
    if lengths.shape != (n,):
        raise ValueError(
            f"lengths shape {tuple(lengths.shape)} != ({n},) (rows of padded)"
        )
    # Single sync — without this, the per-row ``.item()`` below would
    # GPU-sync N times if ``lengths`` lives on CUDA.
    lens = lengths.cpu().tolist() if lengths.is_cuda else lengths.tolist()
    rows = [padded[i, : lens[i]] for i in range(n)]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def maybe_pack_jagged(
    val: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Convert ``val`` to jagged iff it looks like a per-token field.

    Used by every write site (initial put, driver delta-write, worker
    write-back) so all per-token fields land in TQ as jagged with the
    same row lengths — read-time materialization then pads them all to
    the same target shape, avoiding shape-mismatch crashes between
    mixed wire formats.

    Args:
        val: Tensor to consider. Qualifies for jagged conversion only
            when ``val.shape == (N, max(lengths), ...)`` where
            ``N == lengths.shape[0]``.
        lengths: Per-row valid lengths, shape ``(N,)``.

    Returns:
        A ``torch.jagged`` nested tensor when the shape heuristic matches;
        otherwise ``val`` passed through as a rectangular tensor.
    """
    n = lengths.shape[0]
    if n == 0:
        return val.detach().contiguous()
    max_len = int(lengths.max().item())
    if val.dim() < 2 or val.shape[0] != n or val.shape[1] != max_len:
        return val.detach().contiguous()
    return to_nested_by_length(val.detach(), lengths)


def pack_per_token_field(val: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Force-jaggedize a known per-token field, tolerating SP padding.

    Unlike :func:`maybe_pack_jagged` (which is shape-strict to avoid
    false positives on 3D extras like image features), this function is
    invoked at write-back sites where the caller already knows the
    field is per-token (e.g. ``prev_logprobs``,
    ``reference_policy_logprobs``). mcore SP rounds the forward
    output's seq dim up to a multiple of TP, so the value can be 1+
    tokens wider than ``max(lengths)``; :func:`to_nested_by_length`
    slices each row to its own length and drops the trailing SP
    padding cleanly.

    Args:
        val: Per-token tensor. Falls back to rectangular when it cannot
            be jaggedized (wrong batch dim, < 2D, or seq dim shorter
            than ``max(lengths)``).
        lengths: Per-row valid lengths, shape ``(N,)``.

    Returns:
        A ``torch.jagged`` nested tensor when the shape allows;
        otherwise ``val`` passed through as a rectangular tensor.
    """
    n = lengths.shape[0]
    if n == 0:
        return val.detach().contiguous()
    max_len = int(lengths.max().item())
    if val.dim() < 2 or val.shape[0] != n or val.shape[1] < max_len:
        return val.detach().contiguous()
    return to_nested_by_length(val.detach(), lengths)


def response_from_nested(
    full: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract the response slice from a (prompt+response) nested tensor.

    Used on the worker side for logprob / ref-logprob write-back where
    only the response-token slice is interesting downstream. The
    "left-shift by one token" convention is applied (so logprobs at
    output position i correspond to the prediction of input token i+1).

    Args:
        full: Jagged nested tensor of shape
            ``(N, prompt_len + response_len)``.
        response_mask: Jagged nested tensor of shape
            ``(N, response_len)``; its ``offsets().diff()`` gives the
            per-row response length.

    Returns:
        Jagged nested tensor of shape ``(N, response_len)`` containing
        the left-shifted response slice.
    """
    values = full.values()
    offsets = full.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift output by one token for log_probs / values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


# ── materialize: wire TensorDict → trainer BatchedDataDict ────────────


def materialize(
    td: TensorDict,
    layout: Layout = "padded",
    pad_value_dict: dict[str, int | float] | None = None,
    pad_to_multiple: int = 1,
) -> "BatchedDataDict[Any]":
    """Convert a wire TensorDict to a BatchedDataDict.

    Trainer/worker code expects rectangular tensors — this is the
    bridge from the on-wire nested format.

    The lazy ``BatchedDataDict`` import keeps
    ``import nemo_rl.data_plane`` cheap for unit tests that don't
    actually call this function (``BatchedDataDict`` transitively
    pulls multimodal deps like decord / torchvision).

    Args:
        td: Wire TensorDict to materialize.
        layout: ``"padded"`` (default) pads nested-tensor leaves via
            :func:`torch.nested.to_padded_tensor` using
            ``pad_value_dict[k]`` (or 0 if unspecified); rectangular
            leaves pass through. ``"jagged"`` passes nested leaves
            through — use only when the caller knows how to consume
            them.
        pad_value_dict: Per-field pad value used when ``layout='padded'``.
        pad_to_multiple: Round the seq dim up to the next multiple after
            ``to_padded_tensor``. Required when downstream backends
            impose alignment (mcore SP needs ``seq_len % TP == 0``;
            PyTorch CP needs ``seq_len % (CP * 2) == 0``). Default 1
            disables extra alignment.

    Returns:
        ``BatchedDataDict`` with rectangular tensors for padded layout,
        nested tensors for jagged layout, and ``np.ndarray(dtype=object)``
        for ``NonTensorStack`` leaves (TQ-native non-tensor passthrough).
    """
    from tensordict import NonTensorData, NonTensorStack

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if pad_to_multiple < 1:
        raise ValueError(
            f"pad_to_multiple must be >= 1, got {pad_to_multiple}"
        )
    pads = pad_value_dict or {}
    out: dict[str, Any] = {}
    for key, val in td.items(include_nested=False):
        if isinstance(val, NonTensorStack):
            out[key] = np.asarray(val.tolist(), dtype=object)
            continue
        if isinstance(val, NonTensorData):
            out[key] = np.asarray([val.data], dtype=object)
            continue
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"materialize() received unexpected leaf type for {key!r}: "
                f"{type(val)}. Expected Tensor or NonTensorStack."
            )
        if val.is_nested and layout == "padded":
            pad = pads.get(key, 0)
            padded = torch.nested.to_padded_tensor(val, padding=pad)
            if pad_to_multiple > 1 and padded.dim() >= 2:
                seq_dim = padded.shape[1]
                rem = seq_dim % pad_to_multiple
                if rem != 0:
                    extra = pad_to_multiple - rem
                    pad_spec = [0, 0] * (padded.dim() - 2) + [0, extra]
                    padded = torch.nn.functional.pad(padded, pad_spec, value=pad)
            out[key] = padded
        else:
            out[key] = val
    return BatchedDataDict(out)
