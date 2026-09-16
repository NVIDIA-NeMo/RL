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
"""Driver-side TransferQueue helpers shared by TQPolicy and TQValue."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from nemo_rl.data.multimodal_utils import present_multimodal_fields
from nemo_rl.data_plane.column_io import read_columns, round_up, write_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class TQDriverMixin:
    """Pad-target minting and column read/write against the data plane.

    Hosts must provide cfg, dp_client, and the use_dynamic_batches /
    use_sequence_packing attribute pairs that Policy and Value both set.
    """

    def _packing_args(
        self,
        mb_tokens_key: str,
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Resolve (sequence_packing_args, dynamic_batching_args) for a given stage.

        The stage is identified by ``mb_tokens_key`` (``"logprob_mb_tokens"`` or
        ``"train_mb_tokens"``).
        """
        if getattr(self, "use_dynamic_batches", False):
            args = dict(self.dynamic_batching_args)
            args["max_tokens_per_microbatch"] = self.cfg["dynamic_batching"][
                mb_tokens_key
            ]
            return None, args
        if getattr(self, "use_sequence_packing", False):
            args = dict(self.sequence_packing_args)
            args["max_tokens_per_microbatch"] = self.cfg["sequence_packing"][
                mb_tokens_key
            ]
            return args, None
        return None, None

    def _stamp_pad_seqlen(self, meta: KVBatchMeta) -> None:
        """Mint ``GLOBAL_FORWARD_PAD_SEQLEN`` onto ``meta.extra_info`` (idempotent).

        Cross-DP forward pad target. Preshard shards inherit it via
        ``dict(meta.extra_info)`` propagation.
        """
        if not meta.sequence_lengths:
            return
        if GLOBAL_FORWARD_PAD_SEQLEN in meta.extra_info:
            return
        _, dba = self._packing_args("train_mb_tokens")
        seq_round = int(dba["sequence_length_round"]) if dba is not None else 1
        pad_mult = int(meta.extra_info.get("pad_to_multiple", 1))
        meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = round_up(
            max(meta.sequence_lengths), max(pad_mult, seq_round)
        )

    def _isolated_meta(
        self,
        meta: KVBatchMeta,
        *,
        fields: list[str],
        task_name: str,
        include_multimodal: bool = True,
    ) -> KVBatchMeta:
        """Narrow ``meta`` for one model's dispatch and mint it a fresh pad target.

        The mint is idempotent, so sharing or inheriting the target would let
        whichever model dispatches first decide the forward pad for the rest --
        and with ppo_epochs > 1 the caller's meta is already stamped when the
        critic dispatches again.

        ``fields`` is the dispatch's *static* column list; the multimodal
        columns the rollout actually wrote are unioned in here rather than at
        each call site. Every forward-running dispatch of the *policy* needs
        them -- omitting them runs a VLM forward image-blind while the sibling
        dispatches saw the images -- and nothing in a static list forces the
        next caller to remember. The union is a no-op for text-only runs, since
        ``present_multimodal_fields`` intersects with ``meta.fields``.

        ``include_multimodal=False`` opts a dispatch out. A value model is the
        case that exists: its inputs are a fixed list and it never consumes
        pixels, so unioning them in would fetch and broadcast a column the
        forward discards.
        """
        extra_info = dict(meta.extra_info)
        extra_info.pop(GLOBAL_FORWARD_PAD_SEQLEN, None)
        multimodal = (
            [f for f in present_multimodal_fields(meta) if f not in fields]
            if include_multimodal
            else []
        )
        isolated = replace(
            meta,
            fields=[*fields, *multimodal],
            task_name=task_name,
            extra_info=extra_info,
        )
        self._stamp_pad_seqlen(isolated)
        return isolated

    def read_from_dataplane(
        self,
        meta: KVBatchMeta,
        *,
        select_fields: list[str],
        pad_value_dict: Optional[dict[str, Any]] = None,
    ) -> BatchedDataDict[Any]:
        """Fetch + materialize columns from the data plane (TQ).

        ``read_columns`` pads to ``meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN]``
        — the same value workers pad to in their forward pass. Driver
        and workers thus return columns at one identical seq dim, with
        no driver-side knowledge of ``sequence_length_round``.
        """
        self._stamp_pad_seqlen(meta)
        return read_columns(
            self.dp_client,
            meta,
            select_fields=select_fields,
            pad_value_dict=pad_value_dict,
        )

    def write_to_dataplane(self, meta: KVBatchMeta, fields: dict[str, Any]) -> None:
        """Write driver-computed columns to the data plane (TQ)."""
        write_columns(self.dp_client, meta, fields=fields)
