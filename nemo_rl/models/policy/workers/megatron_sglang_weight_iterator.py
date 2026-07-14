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

"""SGLang-only HF weight iterator for the Megatron policy worker.

Emits buckets of HF-named tensors restored from Megatron via AutoBridge,
with no vLLM-specific KV/Q scale tensors.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch


class MegatronSGLangHfWeightIterator:
    """Yield buckets of finalized HF named tensors for SGLang weight refit.

    The iterator is bound to a Megatron bridge, the local Megatron model(s),
    and the conversion-task list precomputed by the policy worker. For each
    refit it walks ``bridge.export_hf_weights`` and packs tensors into buckets
    sized by the *post-transformation* tensor footprint.
    """

    def __init__(
        self,
        *,
        megatron_bridge: Any,
        models: list[Any],
        conversion_tasks: Any,
    ) -> None:
        self._bridge = megatron_bridge
        self._models = models
        self._conversion_tasks = conversion_tasks

    def iter_hf_weight_buckets(
        self,
        *,
        buffer_size_bytes: int,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield finalized HF tensor buckets sized by transmitted bytes."""
        if buffer_size_bytes <= 0:
            raise ValueError(
                f"buffer_size_bytes must be positive, got {buffer_size_bytes}"
            )

        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_size = 0

        for finalized in self._iter_finalized_hf_named_tensors():
            for name, tensor in finalized:
                tensor_size = tensor.numel() * tensor.element_size()
                if bucket and bucket_size + tensor_size > buffer_size_bytes:
                    yield bucket
                    bucket = []
                    bucket_size = 0
                bucket.append((name, tensor))
                bucket_size += tensor_size

        if bucket:
            yield bucket

    def _iter_finalized_hf_named_tensors(
        self,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield finalized HF (name, tensor) groups from one AutoBridge tensor.

        AutoBridge yields one HF named tensor at a time; each item produces
        exactly one finalized pair.
        """
        for hf_param_name, tensor in self._bridge.export_hf_weights(
            self._models,
            show_progress=False,
            conversion_tasks=self._conversion_tasks,
        ):
            # AutoBridge yields plain ``torch.Tensor`` for Megatron (no
            # DTensor / async-collective wrapping), so no ``.wait()`` is
            # needed here. The previous ``hasattr(tensor, "wait")`` check
            # was a copy-from-FSDP residue.
            yield [(hf_param_name, tensor)]
