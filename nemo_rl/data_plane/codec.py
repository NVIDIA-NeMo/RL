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
``put_samples``. Padding tax is paid only when a consumer needs a
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
from tensordict import TensorDict, TensorDictBase

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
    into the wire format before ``put_samples``.

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


def stack_or_nest(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack equal-shape rows; reconstruct as jagged nested when ragged.

    Args:
        tensors: Per-row tensors; assumed to share leading dims modulo
            an optional ragged seq dim. Empty list returns ``torch.empty(0)``.

    Returns:
        A regular tensor when all rows share shape; otherwise a
        ``torch.jagged`` nested tensor.
    """
    if not tensors:
        return torch.empty(0)
    first_shape = tensors[0].shape
    if all(t.shape == first_shape for t in tensors):
        return torch.stack(tensors, dim=0)
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def unwrap_wire_stripped_payload(item: Any) -> Any:
    """Recover the payload of a possibly wire-stripped ``NonTensorData``.

    TQ's ``MsgpackEncoder._encode_tensordict`` serializes any
    ``TensorDictBase`` via ``dict(obj.items())`` — only the tensor
    backing dict. ``NonTensorData`` stores its payload in
    ``_non_tensordict["data"]``, so it round-trips through ZMQ as an
    empty ``TensorDict({}, batch_size=[])``. We map only that exact
    signature to ``None``; any other ``TensorDictBase`` (with tensor
    fields, non-scalar batch, or a salvageable ``_non_tensordict``
    payload) passes through unchanged so we never drop real data.
    """
    nt = getattr(item, "_non_tensordict", None)
    if isinstance(nt, dict) and "data" in nt:
        return nt["data"]
    if (
        isinstance(item, TensorDictBase)
        and item.batch_dims == 0
        and len(item.keys()) == 0
    ):
        return None
    return item


def pack_jagged_fields(
    fields: "dict[str, torch.Tensor | np.ndarray]",
    *,
    lengths: torch.Tensor | None,
    token_aligned_fields: set[str] | frozenset[str] | None = None,
) -> TensorDict:
    """Pack a column dict into the wire layout expected by ``put_samples``.

    Zero-copy where possible: explicitly named per-token tensors become
    ``torch.jagged`` views via :func:`pack_per_token_field`; all other
    tensors pass through rectangular; ``np.ndarray(dtype=object)`` is
    forwarded as-is. This is a **layout transform**, not serialization
    — the on-wire bytes are produced later by the TQ backend's msgpack
    encoder. Centralizing the transform here makes it the single source
    of truth for both :func:`kv_first_write` and :func:`write_columns`.

    Args:
        fields: Column name → tensor or object array. Other value types
            raise ``TypeError``.
        lengths: Per-row valid lengths used by :func:`pack_per_token_field`.
            ``None`` disables jagged conversion entirely.
        token_aligned_fields: Field names known to be per-token. These use
            :func:`pack_per_token_field`, which tolerates extra padded columns
            and slices each row to ``lengths``.

    Returns:
        ``TensorDict`` with ``batch_size=[N]`` (N from ``lengths`` if
        given, else 0) ready for ``put_samples``.
    """
    n = int(lengths.shape[0]) if lengths is not None else 0
    token_aligned_fields = token_aligned_fields or frozenset()
    packed: dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            # tensordict==0.12.2 wire bug: a NonTensorStack stored as a
            # TensorDict leaf returns as a LinkedList on parent
            # __getitem__, losing identity. ndarray(dtype=object)
            # round-trips intact.
            packed[k] = v
        elif isinstance(v, torch.Tensor):
            if lengths is not None and k in token_aligned_fields:
                packed[k] = pack_per_token_field(v, lengths)
            else:
                packed[k] = v.detach().contiguous()
        else:
            raise TypeError(
                f"pack_jagged_fields: unsupported value type for {k!r}: {type(v)}. "
                "Use torch.Tensor or np.ndarray(dtype=object)."
            )
    return TensorDict(packed, batch_size=[n])


def pack_per_token_field(val: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Force-jaggedize a known per-token field, tolerating SP padding.

    This function is invoked at write sites where the caller already
    knows the field is per-token (e.g. ``prev_logprobs``,
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
    pad_to_seqlen: int = 0,
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
        pad_to_seqlen: When > 0, right-pad the seq dim up to this
            absolute length after ``to_padded_tensor``. Worker-side
            ``_fetch`` passes its forward-pass target here (rounded up
            to ``sequence_length_round`` for Megatron's microbatch
            iterator); driver-side ``read_columns`` leaves it 0 and
            consumes the natural-padded shape. Default 0 disables.

    Returns:
        ``BatchedDataDict`` with rectangular tensors for padded layout,
        nested tensors for jagged layout, and ``np.ndarray(dtype=object)``
        for ``NonTensorStack`` leaves (TQ-native non-tensor passthrough).
    """
    from tensordict import NonTensorData, NonTensorStack

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    pads = pad_value_dict or {}
    out: dict[str, Any] = {}
    # pyrefly: inference cycle on tensordict.items() loop var.
    for key, val in td.items(include_nested=False):  # type: ignore[bad-assignment]
        if isinstance(val, NonTensorStack):
            # ``np.asarray(list, dtype=object)`` would probe each item's
            # ``__iter__`` to detect a nested array. A wire-stripped TD
            # has ``batch_dims=0`` → its ``__iter__`` raises
            # ``StopIteration`` → ``RuntimeError: generator raised
            # StopIteration``. ``np.empty + assignment`` skips that
            # probe; ``unwrap_wire_stripped_payload`` normalizes both
            # live ``NonTensorData`` and stripped TDs.
            items = val.tolist()
            arr = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):
                arr[i] = unwrap_wire_stripped_payload(item)
            out[key] = arr
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
        else:
            pad = pads.get(key, 0)
            padded = val
        # Apply `pad_to_seqlen` to ALL 2D+ tensors, not only the freshly-
        # padded-from-nested case. Rectangular wire payloads (vLLM's
        # right-padded output) ride the ``else`` branch above, so without
        # this they'd skip the cross-DP forward pad target and break the
        # microbatch iterator (truncate_tensors → narrow length>size).
        if (
            pad_to_seqlen > 0
            and isinstance(padded, torch.Tensor)
            and padded.dim() >= 2
            and padded.shape[1] < pad_to_seqlen
        ):
            pad_spec = [0, 0] * (padded.dim() - 2) + [
                0,
                pad_to_seqlen - padded.shape[1],
            ]
            padded = torch.nn.functional.pad(padded, pad_spec, value=pad)
        out[key] = padded
    return BatchedDataDict(out)
