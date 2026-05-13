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

* Non-tensor object fields (verl-style ``np.ndarray(dtype=object)``)
ride the same wire as variable-length tensors: each row is pickled
to ``bytes`` and packed into a jagged uint8 nested tensor via
:func:`pack_object_array`. Reader unpacks via
:func:`unpack_object_array` and emits the field as an object array
in the materialized BatchedDataDict. Backends see only tensors —
no per-backend non-tensor support required.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.schema import Layout

if TYPE_CHECKING:
    # Type-only import. At runtime, BatchedDataDict is loaded lazily
    # inside materialize() — see comment there for rationale.
    from nemo_rl.data_plane.interfaces import KVBatchMeta
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


# Stringly-typed extra_info key for the object-encoded field set;
# referenced by the writer (kv_first_write), driver-side reader
# (column_io.read_columns) and worker-side reader (worker_mixin._fetch).
META_OBJECT_FIELDS = "object_fields"


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


# ── Object-array codec (verl-style non-tensor passthrough) ────────────


def pack_object_array(arr: "np.ndarray | list[Any]") -> torch.Tensor:
    """Pickle each element and pack into a jagged uint8 nested tensor.

    Mirrors verl's ``non_tensor_batch: dict[str, np.ndarray(dtype=object)]``
    on a tensor-only wire: each row's pickled bytes ride a ``torch.jagged``
    nested tensor of dtype ``uint8``. Backends that already handle nested
    tensors (simple, mooncake_cpu) carry object payloads transparently;
    no per-backend non-tensor codepath is required.

    Pickle is used unconditionally — the wire stays inside one Ray
    cluster where producer / consumer share the venv, so format
    compatibility is implicit.

    Args:
        arr: Python list or numpy object array of items to pickle.

    Returns:
        2D jagged ``(N, *)`` uint8 nested tensor. Recover via
        :func:`unpack_object_array`.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype != object:
            raise TypeError(f"pack_object_array expects dtype=object; got {arr.dtype}")
        items: list[Any] = list(arr)
    elif isinstance(arr, list):
        items = arr
    else:
        raise TypeError(
            f"pack_object_array expects list or np.ndarray(object); got {type(arr)}"
        )

    rows: list[torch.Tensor] = []
    for item in items:
        # Trusted serialization — producer-registered `object_fields` only.
        b = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        # np.frombuffer + .copy() avoids the "non-writable buffer" warning
        # and severs the lifetime tie to the bytes object.
        rows.append(torch.from_numpy(np.frombuffer(b, dtype=np.uint8).copy()))
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def unpack_object_array(t: torch.Tensor) -> "np.ndarray":
    """Inverse of :func:`pack_object_array`.

    Each row is unpickled in isolation.

    Args:
        t: Jagged uint8 nested tensor produced by :func:`pack_object_array`.

    Returns:
        ``np.ndarray(dtype=object)`` of the decoded items.
    """
    if not t.is_nested:
        raise ValueError(
            "unpack_object_array expects a nested (jagged) tensor; "
            "got rectangular — did the wire codec change?"
        )
    rows = t.unbind()
    out = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        out[i] = pickle.loads(row.numpy().tobytes())
    return out


def select_object_fields(
    meta: "KVBatchMeta",
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Filter ``meta.extra_info[META_OBJECT_FIELDS]`` to a request set.

    Single chokepoint for the read-side filter so :func:`materialize`
    decodes the right keys regardless of caller (column_io,
    worker_mixin).

    Args:
        meta: ``KVBatchMeta`` whose ``extra_info`` carries the registered
            object-field names.
        requested: Subset of names to keep; ``None`` returns the full
            registered set.

    Returns:
        Ordered list of object-field names that appear in both the
        registered set and ``requested``.
    """
    extras = meta.extra_info or {}
    fields = extras.get(META_OBJECT_FIELDS, ())
    if requested is None:
        return list(fields)
    req = set(requested)
    return [k for k in fields if k in req]


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
    object_fields: Iterable[str] | None = None,
) -> "BatchedDataDict[Any]":
    """Convert a wire TensorDict to a BatchedDataDict.

    Trainer/worker code expects rectangular tensors — this is the
    bridge from the on-wire nested/uint8-packed format.

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
        object_fields: Names of fields written via
            :func:`pack_object_array`. Each is decoded via
            :func:`unpack_object_array` and emitted as
            ``np.ndarray(dtype=object)``; tensor padding/alignment do
            not apply. Typically read from
            ``meta.extra_info["object_fields"]`` by the driver / worker
            fetch helpers.

    Returns:
        ``BatchedDataDict`` with rectangular tensors for padded layout,
        nested tensors for jagged layout, and object arrays for fields
        listed in ``object_fields``.
    """
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if pad_to_multiple < 1:
        raise ValueError(
            f"pad_to_multiple must be >= 1, got {pad_to_multiple}"
        )
    pads = pad_value_dict or {}
    obj_set = set(object_fields or ())
    out: dict[str, Any] = {}
    for key, val in td.items(include_nested=False):
        if key in obj_set:
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"materialize() object field {key!r} is not a tensor: "
                    f"{type(val)}; wire encoding broken."
                )
            out[key] = unpack_object_array(val)
            continue
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"materialize() received non-tensor leaf {key!r}: {type(val)}. "
                "Wire format must be tensor-only."
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
