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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from tensordict import TensorDict

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

    ``padded`` has shape ``(N, S, ...)``; ``lengths`` has shape ``(N,)``.
    Returns a ``torch.jagged`` nested tensor whose i-th row is
    ``padded[i, :lengths[i], ...]``.

    Used by the producer side: convert
    :func:`batched_message_log_to_flat_message` output (already padded)
    into the wire format before ``kv_batch_put``.
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
    rows = [padded[i, : int(lengths[i].item())] for i in range(n)]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


# 1D field round-trip kill-switch for the KV-path. TQ's
# KVStorageManager silently unsqueezes 1D fields in metadata while
# row-iterating them in data (transfer_queue/metadata.py:171 vs
# storage/managers/base.py:_generate_values). Backends that go through
# that path (mooncake_cpu) need the writer to unsqueeze 1D fields to
# (N, 1) so per-row tensors match the metadata shape; the reader then
# squeezes the trailing 1 back. Default off — only the affected
# adapter flips it.
_KV_PROMOTE_1D = False


def set_kv_promote_1d(enabled: bool) -> None:
    """Adapter hook: when True, writer unsqueezes 1D bulk fields to
    (N, 1) and reader squeezes the trailing 1 in :func:`materialize`.

    Required by backends that go through TQ's KVStorageManager path
    (mooncake_cpu) — see ``_KV_PROMOTE_1D`` above for the schema/data
    mismatch.
    """
    global _KV_PROMOTE_1D
    _KV_PROMOTE_1D = bool(enabled)


def maybe_pack_jagged(
    val: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Convert ``val`` to jagged iff it looks like a per-token field.

    Heuristic: ``val`` qualifies when ``val.shape == (N, max(lengths), ...)``
    where ``N == lengths.shape[0]``. Other shapes pass through as
    rectangular tensors. Used by every write site (initial put,
    driver delta-write, worker write-back) so all per-token fields
    land in TQ as jagged with the same row lengths — read-time
    materialization then pads them all to the same target shape,
    avoiding shape-mismatch crashes between mixed wire formats.
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
    output's seq dim up to a multiple of TP, so the value can be
    1+ tokens wider than ``max(lengths)``; :func:`to_nested_by_length`
    slices each row to its own length and drops the trailing SP
    padding cleanly.

    Falls back to rectangular when ``val`` cannot be jaggedized
    (wrong batch dim, < 2D, or seq dim shorter than ``max(lengths)``).
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
    only the response-token slice is interesting downstream.

    ``full``: jagged nested tensor of shape ``(N, prompt_len + response_len)``.
    ``response_mask``: jagged nested tensor of shape ``(N, response_len)``;
    its ``offsets().diff()`` gives the per-row response length.

    Output: jagged nested tensor of shape ``(N, response_len)`` with the
    "left-shift by one token" convention applied (so logprobs at output
    position i correspond to the prediction of input token i+1).
    """
    values = full.values()
    offsets = full.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift output by one token for log_probs / values
        response_list.append(
            values[seq_offset - resp_len - 1 : seq_offset - 1]
        )
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


# ── materialize: wire TensorDict → trainer BatchedDataDict ────────────


def materialize(
    td: TensorDict,
    layout: Literal["padded", "jagged"] = "padded",
    pad_value_dict: dict[str, int | float] | None = None,
    pad_to_multiple: int = 1,
) -> "BatchedDataDict[Any]":
    """Convert a wire TensorDict to a BatchedDataDict.

    ``layout='padded'`` (default): any nested-tensor leaves are padded
    via :func:`torch.nested.to_padded_tensor` using ``pad_value_dict[k]``
    (or 0 if not specified). Regular tensor leaves pass through.
    Trainer/worker code expects rectangular tensors — this is the bridge.

    ``pad_to_multiple`` rounds the seq dim up to the next multiple after
    ``to_padded_tensor``. Required when downstream backends impose
    alignment (mcore SP needs ``seq_len % TP == 0``; PyTorch CP needs
    ``seq_len % (CP * 2) == 0``). Default 1 = no extra alignment.

    ``layout='jagged'``: nested leaves pass through; rectangular leaves
    pass through. Use only when the caller knows how to consume nested.

    The lazy ``BatchedDataDict`` import keeps ``import
    nemo_rl.data_plane`` cheap for unit tests that don't actually call
    this function (``BatchedDataDict`` transitively pulls multimodal
    deps like decord / torchvision).
    """
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    pads = pad_value_dict or {}
    out: dict[str, torch.Tensor] = {}
    for key, val in td.items(include_nested=False):
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
        # KV-path round-trip: writer side unsqueezed 1D fields to (N, 1)
        # so per-row tensors match TQ's extract_field_schema implicit
        # unsqueeze (transfer_queue/metadata.py:171-173). Squeeze the
        # trailing 1 back so consumers see the original (N,) shape.
        # Safe to apply unconditionally on the _KV_PROMOTE_1D path: none
        # of the bulk fields naturally carry shape[-1] == 1.
        if _KV_PROMOTE_1D and out[key].dim() >= 2 and out[key].shape[-1] == 1:
            out[key] = out[key].squeeze(-1)
    return BatchedDataDict(out)
