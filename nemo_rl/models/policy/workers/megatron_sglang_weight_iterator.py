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

"""SGLang-only HF weight iterator for the Megatron policy worker.

Emits buckets of HF-named tensors restored from Megatron via AutoBridge,
with no vLLM-specific KV/Q scale tensors. Quantized targets apply the same
HF-name selection and tensor conversion used by their offline checkpoint
converters.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Iterator

import torch

from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    SKIP_WEIGHT_SUBSTRINGS as MXFP8_SKIP_WEIGHT_SUBSTRINGS,
)
from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    MXFP8_SCALE_KEY_SUFFIX,
    quantize_mxfp8,
    should_quantize,
    strip_weight_suffix,
)
from nemo_rl.models.generation.sglang.quantization_utils import (
    SglangQuantizationScheme,
    build_dynamic_skip_substrings,
)


class MegatronSGLangHfWeightIterator:
    """Yield buckets of finalized HF named tensors for SGLang weight refit.

    The iterator is bound to a Megatron bridge, the local Megatron model(s),
    and the conversion-task list precomputed by the policy worker. For each
    refit it walks ``bridge.export_hf_weights`` and packs tensors into buckets
    sized by the *post-transformation* tensor footprint. Companion tensors
    produced from one source weight remain in the
    same bucket.
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
        target_precision: SglangQuantizationScheme = "bf16",
        buffer_size_bytes: int,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield finalized HF tensor buckets sized by transmitted bytes."""
        if buffer_size_bytes <= 0:
            raise ValueError(
                f"buffer_size_bytes must be positive, got {buffer_size_bytes}"
            )
        if target_precision == "mxfp8":
            # MXFP8 additionally excludes norms, embeddings, router/gate and
            # LM-head weights, which have to stay high precision.
            skip_weight_substrings = build_dynamic_skip_substrings(
                quantization_config=self._quantization_config,
                num_hidden_layers=self._num_hidden_layers,
                static_skip_substrings=MXFP8_SKIP_WEIGHT_SUBSTRINGS,
            )
        elif target_precision == "bf16":
            skip_weight_substrings = None
        else:
            raise ValueError(f"Unsupported SGLang target precision: {target_precision}")

        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_size = 0

        for finalized in self._iter_finalized_hf_named_tensors(
            target_precision=target_precision,
            skip_weight_substrings=skip_weight_substrings,
        ):
            finalized_size = sum(
                tensor.numel() * tensor.element_size() for _, tensor in finalized
            )
            if bucket and bucket_size + finalized_size > buffer_size_bytes:
                yield bucket
                bucket = []
                bucket_size = 0
            bucket.extend(finalized)
            bucket_size += finalized_size

        if bucket:
            yield bucket

    def _iter_finalized_hf_named_tensors(
        self,
        *,
        target_precision: SglangQuantizationScheme,
        skip_weight_substrings: tuple[str, ...] | None,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Yield finalized HF (name, tensor) groups from one AutoBridge tensor.

        AutoBridge yields one HF named tensor at a time. For BF16 each AutoBridge
        item produces exactly one finalized pair; quantized formats expand a
        source weight into the payload required by SGLang.
        """
        hf_weights = self._bridge.export_hf_weights(
            self._models,
            show_progress=False,
            conversion_tasks=self._conversion_tasks,
        )
        for hf_param_name, tensor in hf_weights:
            # AutoBridge yields plain ``torch.Tensor`` for Megatron (no
            # DTensor / async-collective wrapping), so no ``.wait()`` here.
            if target_precision == "mxfp8" and skip_weight_substrings is not None:
                if should_quantize(
                    hf_param_name,
                    tensor,
                    skip_weight_substrings=skip_weight_substrings,
                ):
                    qweight, scale = quantize_mxfp8(tensor)
                    scale_name = (
                        strip_weight_suffix(hf_param_name) + MXFP8_SCALE_KEY_SUFFIX
                    )
                    yield [(hf_param_name, qweight), (scale_name, scale)]
                    continue

            yield [(hf_param_name, tensor)]
