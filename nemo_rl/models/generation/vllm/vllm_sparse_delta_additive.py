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

"""Layout-agnostic sparse-delta apply via vLLM's own weight_loader.

The workstream doc referenced an `additive_weight_load_context` used by the
dense IPC/refit path. On the vLLM 0.20.0 pin and the current nemo_rl tree,
no such abstraction exists — this module provides one and uses it to
implement the proposed default apply path for the sparse-delta receiver.

The context manager matches on storage identity rather than the tensor's
`data_ptr()`, so vLLM's packed loaders (QKV, gate_up, MoE) — which write
to a narrowed view of `param.data` — are also redirected to `.add_()`.
This means routing any name through the additive path delegates all
packing / shard math to vLLM's own `weight_loader`; nemo_rl does not
need to replicate that logic per module family.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

import torch


@contextlib.contextmanager
def additive_weight_load_context(
    *targets: torch.Tensor,
) -> Generator[None, None, None]:
    """Turn `.copy_()` into `.add_()` for the block, for `targets` only.

    Pass specific `param.data` tensors, not a whole model — patching by
    module would silently accumulate unrelated internal copies (bias,
    scales, buffers). Non-target tensors keep overwrite semantics.

    Match uses storage identity, not the tensor's `data_ptr()`. That is
    necessary because vLLM's packed loaders (QKV, gate_up, MoE) write to
    a narrowed view of `param.data` — the view has a different `data_ptr`
    from the base tensor but shares the same underlying storage.
    """
    target_storage_ptrs = frozenset(
        t.untyped_storage().data_ptr() for t in targets
    )
    original_copy_ = torch.Tensor.copy_

    def _additive_copy_(
        self: torch.Tensor, src: Any, non_blocking: bool = False
    ) -> torch.Tensor:
        if self.untyped_storage().data_ptr() in target_storage_ptrs:
            if isinstance(src, torch.Tensor):
                src = src.to(dtype=self.dtype, device=self.device)
            return self.add_(src)
        return original_copy_(self, src, non_blocking)

    torch.Tensor.copy_ = _additive_copy_  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.Tensor.copy_ = original_copy_  # type: ignore[method-assign]


def apply_sparse_delta_via_additive_load(
    name: str,
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    target_shape: tuple[int, ...],
    target_dtype: torch.dtype,
    model: Any,
    device: torch.device,
) -> None:
    """Apply a sparse delta to a named vLLM parameter via its weight_loader.

    Plain-linear (ColumnParallelLinear / RowParallelLinear, tp_size=1)
    only in Milestone 1. Extension to QKV / gate_up / MoE / Mamba is a
    separate workstream task.
    """
    params = dict(model.named_parameters())
    target_param = params[name]

    dense = torch.zeros(target_shape, dtype=target_dtype, device=device)
    dense.view(-1).index_copy_(
        0,
        sparse_indices.to(dtype=torch.int64, device=device),
        sparse_values.to(dtype=target_dtype, device=device),
    )

    with additive_weight_load_context(target_param.data):
        model.load_weights([(name, dense)])
