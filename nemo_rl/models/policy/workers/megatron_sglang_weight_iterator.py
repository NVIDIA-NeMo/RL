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

from typing import Any, Iterator

import torch

from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    MXFP8_SCALE_KEY_SUFFIX,
    quantize_mxfp8,
    should_quantize,
    strip_weight_suffix,
)
from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    SKIP_WEIGHT_SUBSTRINGS as MXFP8_SKIP_WEIGHT_SUBSTRINGS,
)
from nemo_rl.models.generation.sglang.nvfp4_quantization_core import (
    nvfp4_quantized_entries,
    quantize_nvfp4,
    quantize_nvfp4_pair,
    should_quantize_nvfp4,
    should_skip_nvfp4_gated_pair,
    split_gated_pair_name,
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
    produced from one source weight (and NVFP4 gate/up pairs) remain in the
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
            # LM-head weights, which have to stay high precision. NVFP4 has no
            # such static list: it only ever targets MoE expert GEMMs.
            skip_weight_substrings = build_dynamic_skip_substrings(
                quantization_config=self._quantization_config,
                num_hidden_layers=self._num_hidden_layers,
                static_skip_substrings=MXFP8_SKIP_WEIGHT_SUBSTRINGS,
            )
        elif target_precision == "nvfp4":
            skip_weight_substrings = build_dynamic_skip_substrings(
                quantization_config=self._quantization_config,
                num_hidden_layers=self._num_hidden_layers,
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
        if target_precision == "nvfp4":
            if skip_weight_substrings is None:
                raise RuntimeError("NVFP4 refit requires initialized skip rules.")
            yield from self._iter_nvfp4_hf_named_tensors(
                hf_weights,
                skip_weight_substrings=skip_weight_substrings,
            )
            return

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

    @staticmethod
    def _iter_nvfp4_hf_named_tensors(
        hf_weights: Iterable[tuple[str, torch.Tensor]],
        *,
        skip_weight_substrings: tuple[str, ...],
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Quantize NVFP4 expert weights, buffering complete gate/up pairs."""
        pending_pairs: dict[
            str,
            dict[str, tuple[str, torch.Tensor]],
        ] = {}

        for hf_param_name, tensor in hf_weights:
            if should_skip_nvfp4_gated_pair(
                hf_param_name,
                skip_weight_substrings=skip_weight_substrings,
            ):
                yield [(hf_param_name, tensor)]
                continue

            if not should_quantize_nvfp4(
                hf_param_name,
                tensor,
                skip_weight_substrings=skip_weight_substrings,
            ):
                yield [(hf_param_name, tensor)]
                continue

            pair_base, pair_role = split_gated_pair_name(hf_param_name)
            if pair_base is None or pair_role is None:
                yield nvfp4_quantized_entries(
                    hf_param_name,
                    quantize_nvfp4(tensor),
                    include_input_scale=False,
                )
                continue

            pair = pending_pairs.setdefault(pair_base, {})
            if pair_role in pair:
                raise ValueError(
                    "NVFP4 requires one complete gate/up pair per refit; "
                    f"found duplicate {pair_role} tensor for {pair_base}."
                )
            pair[pair_role] = (hf_param_name, tensor)
            if set(pair) != {"gate", "up"}:
                continue

            gate_name, gate_weight = pair["gate"]
            up_name, up_weight = pair["up"]
            gate_output, up_output = quantize_nvfp4_pair(gate_weight, up_weight)
            yield [
                *nvfp4_quantized_entries(
                    gate_name,
                    gate_output,
                    include_input_scale=False,
                ),
                *nvfp4_quantized_entries(
                    up_name,
                    up_output,
                    include_input_scale=False,
                ),
            ]
            del pending_pairs[pair_base]

        if pending_pairs:
            incomplete = {
                base: sorted(roles) for base, roles in sorted(pending_pairs.items())
            }
            raise ValueError(
                "NVFP4 gate/up weights must be quantized together; incomplete "
                f"pairs: {incomplete}."
            )
