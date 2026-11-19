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

"""Stream co-trained draft weights from their PP owner stage for refit."""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch import Tensor

from nemo_rl.distributed.collectives import collective_device
from nemo_rl.models.megatron.pipeline_parallel import broadcast_obj_from_pp_rank
from nemo_rl.models.policy.draft_config import draft_refit_enabled

DRAFT_REFIT_PREFIX = "draft."

_STATUS_OK = "OK"
_STATUS_EXPORTER_ERROR = "EXPORTER_ERROR"
_STATUS_MANIFEST_INVALID = "MANIFEST_INVALID"

# (name, shape, dtype) for each exported draft tensor.
_TensorSpec = tuple[str, tuple[int, ...], torch.dtype]


def _draft_refit_error(code: str, detail: str) -> ValueError:
    return ValueError(f"[draft refit {code}] {detail}")


def _export_owner_manifest(
    draft_model: torch.nn.Module,
) -> tuple[str, str, list[tuple[str, Tensor]], list[_TensorSpec]]:
    from nemo_rl.models.megatron.draft import export_eagle_weights_to_hf

    try:
        weights = list(export_eagle_weights_to_hf(draft_model))
    except Exception as error:
        return (
            _STATUS_EXPORTER_ERROR,
            f"exporter raised {type(error).__name__}: {error}",
            [],
            [],
        )

    if not weights:
        return _STATUS_MANIFEST_INVALID, "exporter returned no tensors", [], []
    names: set[str] = set()
    for name, tensor in weights:
        if not isinstance(tensor, Tensor):
            return (
                _STATUS_MANIFEST_INVALID,
                f"exporter returned a non-tensor value for {name}",
                [],
                [],
            )
        if name in names:
            return (
                _STATUS_MANIFEST_INVALID,
                f"exporter returned duplicate tensor name {name}",
                [],
                [],
            )
        names.add(name)
    specs = [(name, tuple(tensor.shape), tensor.dtype) for name, tensor in weights]
    return _STATUS_OK, "", weights, specs


def stream_draft_weights_for_refit(
    draft_model: torch.nn.Module | None,
    draft_cfg: Mapping[str, Any] | None,
    *,
    metadata_only: bool = False,
) -> Iterator[tuple[str, Tensor]]:
    """Yield ``draft.*`` HF weights on every rank of this PP lane.

    Only the last PP stage owns the draft model. The owner exports the
    weights and broadcasts a small manifest of names, shapes, and dtypes with
    the object collective. The tensors then stream one at a time with tensor
    broadcasts on the PP group, so no rank holds the full draft as a pickled
    object. Every lane rank raises the same error when the owner export fails.

    Args:
        draft_model: The local draft model. ``None`` on non-owner PP stages.
        draft_cfg: The ``policy.draft`` config, used to tell non-owner stages
            that draft refit is enabled.
        metadata_only: Yield meta tensors with the manifest shapes and dtypes.
            No tensor broadcast runs.

    Yields:
        ``(draft.<hf_name>, tensor)`` pairs in the owner export order.
    """
    if draft_model is None and not draft_refit_enabled(draft_cfg):
        return

    weights: list[tuple[str, Tensor]] = []
    owner_manifest = None
    if draft_model is not None:
        status, detail, weights, specs = _export_owner_manifest(draft_model)
        owner_manifest = (status, detail, dist.get_rank(), specs)
    status, detail, owner_global_rank, specs = broadcast_obj_from_pp_rank(
        owner_manifest
    )
    if status != _STATUS_OK:
        raise _draft_refit_error(status, detail)

    if metadata_only:
        for name, shape, dtype in specs:
            yield (
                f"{DRAFT_REFIT_PREFIX}{name}",
                torch.empty(shape, dtype=dtype, device="meta"),
            )
        return

    if parallel_state.get_pipeline_model_parallel_world_size() == 1:
        for name, tensor in weights:
            yield f"{DRAFT_REFIT_PREFIX}{name}", tensor
        return

    pp_group = parallel_state.get_pipeline_model_parallel_group()
    device = collective_device(pp_group)
    for index, (name, shape, dtype) in enumerate(specs):
        if draft_model is not None:
            tensor = weights[index][1].to(device).contiguous()
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(tensor, src=owner_global_rank, group=pp_group)
        yield f"{DRAFT_REFIT_PREFIX}{name}", tensor


def check_draft_refit_metadata(
    name: str,
    tensor: Tensor,
    misc_meta: Mapping[str, Mapping[str, Any]],
) -> None:
    """Raise when a live draft tensor drifts from the prepared refit metadata.

    ``misc_meta`` maps each name to ``{"shape": [...], "dtype": "torch.<dtype>"}``
    as recorded by the nccl_reshard prepare step. The receiver sizes its
    buffers from that metadata, so a drift must fail before the broadcast.
    """
    meta = misc_meta.get(name)
    if meta is None:
        raise _draft_refit_error(
            "METADATA_DRIFT", f"{name} is not in the prepared refit metadata"
        )
    if list(meta["shape"]) != list(tensor.shape) or meta["dtype"] != str(tensor.dtype):
        raise _draft_refit_error(
            "METADATA_DRIFT",
            f"{name} changed after prepare: expected shape={list(meta['shape'])} "
            f"dtype={meta['dtype']}, got shape={list(tensor.shape)} "
            f"dtype={tensor.dtype}",
        )
