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

Modeled after Miles' ``HfWeightIteratorBridge.get_hf_weight_chunks``: emit
buckets of HF-named tensors restored from Megatron via AutoBridge, with no
vLLM-specific KV/Q scale tensors. When ``target_precision == "mxfp8"`` the
iterator additionally applies Miles' offline ``should_quantize`` /
``quantize_mxfp8`` core to each finalized HF tensor.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

import torch

from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    build_dynamic_skip_substrings,
    quantize_mxfp8,
    should_quantize,
    SOURCE_FP8_SCALE_KEY_SUFFIX,
    strip_weight_suffix,
)


class MegatronSGLangHfWeightIterator:
    """Yield buckets of finalized HF named tensors for SGLang weight refit.

    The iterator is bound to a Megatron bridge, the local Megatron model(s),
    and the conversion-task list precomputed by the policy worker. For each
    refit it walks ``bridge.export_hf_weights`` and packs tensors into buckets
    sized by the *post-transformation* tensor footprint, so MXFP8 buckets
    correctly account for the added ``weight_scale_inv`` tensor.
    """

    def __init__(
        self,
        *,
        megatron_bridge: Any,
        models: list[Any],
        conversion_tasks: Any,
        quantization_config: dict[str, Any] | None = None,
        num_hidden_layers: int = 0,
    ) -> None:
        self._bridge = megatron_bridge
        self._models = models
        self._conversion_tasks = conversion_tasks
        self._quantization_config = dict(quantization_config or {})
        self._num_hidden_layers = num_hidden_layers

    def iter_hf_weight_buckets(
        self,
        *,
        target_precision: Literal["bf16", "mxfp8"] = "bf16",
        buffer_size_bytes: int,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield finalized HF tensor buckets sized by transmitted bytes."""
        if buffer_size_bytes <= 0:
            raise ValueError(
                f"buffer_size_bytes must be positive, got {buffer_size_bytes}"
            )

        skip_weight_substrings = (
            build_dynamic_skip_substrings(
                quantization_config=self._quantization_config,
                num_hidden_layers=self._num_hidden_layers,
            )
            if target_precision == "mxfp8"
            else None
        )

        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_size = 0

        for finalized in self._iter_finalized_hf_named_tensors(
            target_precision=target_precision,
            skip_weight_substrings=skip_weight_substrings,
        ):
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
        *,
        target_precision: Literal["bf16", "mxfp8"],
        skip_weight_substrings: tuple[str, ...] | None,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield finalized HF (name, tensor) groups from one AutoBridge tensor.

        AutoBridge yields one HF named tensor at a time. For BF16 each AutoBridge
        item produces exactly one finalized pair; for MXFP8 each item may
        expand to a ``(weight, weight_scale_inv)`` pair when the weight is
        quantized.
        """
        for hf_param_name, tensor in self._bridge.export_hf_weights(
            self._models,
            show_progress=False,
            conversion_tasks=self._conversion_tasks,
        ):
            tensor = tensor if not hasattr(tensor, "wait") else tensor.wait()

            if target_precision == "mxfp8" and skip_weight_substrings is not None:
                if should_quantize(
                    hf_param_name,
                    tensor,
                    skip_weight_substrings=skip_weight_substrings,
                    allow_source_fp8=False,
                ):
                    qweight, scale = quantize_mxfp8(tensor)
                    scale_name = (
                        strip_weight_suffix(hf_param_name)
                        + SOURCE_FP8_SCALE_KEY_SUFFIX
                    )
                    yield [(hf_param_name, qweight), (scale_name, scale)]
                    continue

            yield [(hf_param_name, tensor)]
