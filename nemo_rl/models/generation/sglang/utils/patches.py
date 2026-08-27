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

import logging
import os
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def _get_sglang_file(relative_path: str) -> str:
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"sglang package not found while attempting to patch '{relative_path}'. "
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected sglang file '{relative_path}' not found at '{file_path}'. "
            "The sglang version may have moved this file; compat patch cannot be applied."
        )
    return file_path


def _write_and_verify(
    file_path: str, content: str, sentinel: str | tuple[str, ...]
) -> None:
    sentinels = (sentinel,) if isinstance(sentinel, str) else sentinel
    tmp_path = f"{file_path}.nemo_rl_compat.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

    with open(file_path, "r") as f:
        verify = f.read()
    missing = [item for item in sentinels if item not in verify]
    if missing:
        raise RuntimeError(
            f"Compat patch verification failed for {file_path}: "
            f"sentinel(s) {missing} not present after write. "
            "The write may have been silently dropped by the filesystem."
        )


def _patch_sglang_safe_unpickler() -> None:
    file_to_patch = _get_sglang_file("srt/utils/common.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    sentinel = '"nemo_rl.models.generation.sglang.utils.train_utils."'
    if sentinel in content:
        return

    anchor = '        "torch.nn.parameter.",\n'
    insertion = (
        anchor + '        "nemo_rl.models.generation.sglang.utils.train_utils.",\n'
    )
    if anchor not in content:
        raise RuntimeError(
            f"SafeUnpickler allowlist anchor '{anchor.strip()}' not found in "
            f"{file_to_patch}."
        )

    content = content.replace(anchor, insertion, 1)
    _write_and_verify(file_to_patch, content, sentinel)
    logger.info("Patched SafeUnpickler allowlist in %s.", file_to_patch)


def _patch_sglang_non_gated_fp8_moe() -> None:
    """Backport sglang#36097 for MXFP8/NVFP4 non-gated MoE models.

    The pinned ``sglang-miles`` revision always sizes the fused ``w13``
    buffers for a gated MoE. Nemotron-H uses one non-gated projection, so the
    extra uninitialized shard either fails TP loading or silently corrupts the
    expert weights. This is the source-compatible part of upstream commit
    ``46d9427b913b20ce0d72b95a9209717448b37368``.
    """
    file_to_patch = _get_sglang_file("srt/layers/quantization/fp8.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    sentinel = "w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1"
    if sentinel in content:
        return

    replacements = (
        (
            "        tp_size = get_parallel().tp_size\n\n",
            f"        tp_size = get_parallel().tp_size\n        {sentinel}\n\n",
            1,
        ),
        (
            "            is_concat=True,\n",
            "            is_concat=layer.moe_runner_config.is_gated,\n",
            1,
        ),
        (
            "        if self.with_bias:\n"
            "            w13_up_dim = (\n"
            "                2 * intermediate_size_per_partition\n"
            "                if layer.moe_runner_config.is_gated\n"
            "                else intermediate_size_per_partition\n"
            "            )\n"
            "            w13_weight_bias = torch.nn.Parameter(\n"
            "                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),\n",
            "        if self.with_bias:\n"
            "            w13_weight_bias = torch.nn.Parameter(\n"
            "                torch.empty(\n"
            "                    num_experts,\n"
            "                    w13_num_shards * intermediate_size_per_partition,\n"
            "                    dtype=torch.float32,\n"
            "                ),\n",
            1,
        ),
        (
            "2 * intermediate_size_per_partition",
            "w13_num_shards * intermediate_size_per_partition",
            5,
        ),
        (
            "2 * ((intermediate_size_per_partition + block_n - 1) // block_n)",
            "w13_num_shards\n"
            "                    * ((intermediate_size_per_partition + block_n - 1) // block_n)",
            1,
        ),
        (
            "            # Allocate 2 scales for w1 and w3 respectively.\n"
            "            # They will be combined to a single scale after weight loading.\n"
            "            w13_weight_scale = torch.nn.Parameter(\n"
            "                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False\n"
            "            )\n",
            "            # One scale per w13 shard; a gated layer combines its two into a\n"
            "            # single scale after weight loading.\n"
            "            w13_weight_scale = torch.nn.Parameter(\n"
            "                torch.ones(num_experts, w13_num_shards, dtype=torch.float32),\n"
            "                requires_grad=False,\n"
            "            )\n",
            1,
        ),
        (
            "            assert layer.w13_weight_scale is not None\n"
            "            shard_size = layer.intermediate_size_per_partition\n"
            "            max_w13_scales = layer.w13_weight_scale.max(dim=1).values\n"
            "            for expert_id in range(layer.num_local_experts):\n"
            "                start = 0\n"
            "                for shard_id in range(2):\n"
            "                    dq_weight = per_tensor_dequantize(\n"
            "                        layer.w13_weight[expert_id][start : start + shard_size, :],\n"
            "                        layer.w13_weight_scale[expert_id][shard_id],\n"
            "                    )\n"
            "                    (\n"
            "                        layer.w13_weight[expert_id][start : start + shard_size, :],\n"
            "                        _,\n"
            "                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])\n"
            "                    start += shard_size\n",
            "            assert layer.w13_weight_scale is not None\n"
            f"            {sentinel}\n"
            "            shard_size = layer.intermediate_size_per_partition\n"
            "            max_w13_scales = layer.w13_weight_scale.max(dim=1).values\n"
            "            # A single shard already carries one scale per expert; nothing to fuse.\n"
            "            if w13_num_shards > 1:\n"
            "                for expert_id in range(layer.num_local_experts):\n"
            "                    start = 0\n"
            "                    for shard_id in range(w13_num_shards):\n"
            "                        dq_weight = per_tensor_dequantize(\n"
            "                            layer.w13_weight[expert_id][\n"
            "                                start : start + shard_size, :\n"
            "                            ],\n"
            "                            layer.w13_weight_scale[expert_id][shard_id],\n"
            "                        )\n"
            "                        (\n"
            "                            layer.w13_weight[expert_id][\n"
            "                                start : start + shard_size, :\n"
            "                            ],\n"
            "                            _,\n"
            "                        ) = scaled_fp8_quant(\n"
            "                            dq_weight, max_w13_scales[expert_id]\n"
            "                        )\n"
            "                        start += shard_size\n",
            1,
        ),
        (
            "        assert layer.w13_weight_scale is not None\n"
            "        shard_size = layer.intermediate_size_per_partition\n"
            "        max_w13_scales = layer.w13_weight_scale.max(dim=1).values\n",
            "        assert layer.w13_weight_scale is not None\n"
            f"        {sentinel}\n"
            "        shard_size = layer.intermediate_size_per_partition\n"
            "        max_w13_scales = layer.w13_weight_scale.max(dim=1).values\n",
            1,
        ),
        (
            "            for shard_id in range(2):\n"
            "                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:\n",
            "            for shard_id in range(w13_num_shards):\n"
            "                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:\n",
            1,
        ),
        (
            "        intermediate_size_per_partition = layer.intermediate_size_per_partition\n\n"
            "        self.ab_strides1 = torch.full(\n",
            "        intermediate_size_per_partition = layer.intermediate_size_per_partition\n"
            f"        {sentinel}\n\n"
            "        self.ab_strides1 = torch.full(\n",
            1,
        ),
    )

    for anchor, replacement, expected_count in replacements:
        actual_count = content.count(anchor)
        if actual_count != expected_count:
            raise RuntimeError(
                "SGLang non-gated FP8 MoE compat-patch anchor mismatch in "
                f"{file_to_patch}: expected {expected_count}, found {actual_count} "
                f"for {anchor[:80]!r}."
            )
        content = content.replace(anchor, replacement)

    _write_and_verify(
        file_to_patch,
        content,
        (
            sentinel,
            "is_concat=layer.moe_runner_config.is_gated",
            "for shard_id in range(w13_num_shards)",
        ),
    )
    logger.info("Patched non-gated FP8 MoE weight sizing in %s.", file_to_patch)


def _patch_sglang_mxfp8_moe_scale_layout() -> None:
    """Pad canonical MXFP8 MoE scales before the Triton layout conversion.

    The pinned SGLang revision reshapes serialized scales as though each
    expert's row dimension were already padded to 128. Checkpoints and online
    refits intentionally carry one scale per real row, so models such as
    Nemotron-H fail when that dimension is not 128-aligned. The layout-only
    rows must be added immediately before swizzling and removed again before a
    subsequent weight update.
    """
    file_to_patch = _get_sglang_file("srt/layers/quantization/fp8.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    pad_sentinel = "scale = scale.reshape(num_experts, m, k // 32)"
    restore_sentinel = "Restore canonical MXFP8 MoE scales before hot reload."
    if pad_sentinel in content and restore_sentinel in content:
        return

    swizzle_anchor = (
        "            num_experts, m, k = weight_shape\n"
        "            aligned_m = ((m + 127) // 128) * 128\n"
        "            scale = scale.view(num_experts, aligned_m, k // 32)\n"
    )
    swizzle_replacement = (
        "            num_experts, m, k = weight_shape\n"
        "            aligned_m = ((m + 127) // 128) * 128\n"
        f"            {pad_sentinel}\n"
        "            if aligned_m != m:\n"
        "                scale = torch.nn.functional.pad(\n"
        "                    scale, (0, 0, 0, aligned_m - m)\n"
        "                )\n"
    )
    restore_anchor = (
        "            align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)\n\n"
        "    def process_weights_after_loading(self, layer: Module) -> None:\n"
    )
    restore_replacement = (
        "            align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)\n\n"
        "    def restore_weights_before_loading(self, layer: Module) -> None:\n"
        f"        # {restore_sentinel}\n"
        "        if not self.use_mxfp8:\n"
        "            return\n\n"
        "        for scale_name, weight_name in (\n"
        '            ("w13_weight_scale_inv", "w13_weight"),\n'
        '            ("w2_weight_scale_inv", "w2_weight"),\n'
        "        ):\n"
        "            scale = getattr(layer, scale_name)\n"
        "            weight = getattr(layer, weight_name)\n"
        "            num_experts, m, k = weight.shape\n"
        "            canonical_shape = (num_experts, m, k // 32)\n"
        "            if scale.data.shape != canonical_shape:\n"
        "                scale.data = scale.data.new_empty(canonical_shape)\n"
        "            scale.format_ue8m0 = True\n\n"
        "    def process_weights_after_loading(self, layer: Module) -> None:\n"
    )

    for anchor, replacement in (
        (swizzle_anchor, swizzle_replacement),
        (restore_anchor, restore_replacement),
    ):
        actual_count = content.count(anchor)
        if actual_count != 1:
            raise RuntimeError(
                "SGLang MXFP8 MoE scale-layout compat-patch anchor mismatch in "
                f"{file_to_patch}: expected 1, found {actual_count} for "
                f"{anchor[:80]!r}."
            )
        content = content.replace(anchor, replacement, 1)

    _write_and_verify(file_to_patch, content, (pad_sentinel, restore_sentinel))
    logger.info("Patched MXFP8 MoE scale layout in %s.", file_to_patch)


def _patch_sglang_mxfp8_moe_hidden_size() -> None:
    """Pad Nemotron-H's MXFP8 MoE hidden size at the FlashInfer boundary.

    FlashInfer's SM100 TRT-LLM MXFP8 kernels have no valid tactic for hidden
    size 2688, while 2816 is supported. Keep the model and refit payloads at
    their canonical size and pad only the kernel-facing weights, scales, and
    activations. The loader support below keeps those padded buffers stable
    across hot refits so captured CUDA graphs retain their parameter pointers.
    """
    runner_file = _get_sglang_file("srt/layers/moe/moe_runner/flashinfer_trtllm.py")
    loader_file = _get_sglang_file("srt/layers/moe/fused_moe_triton/layer.py")

    with open(runner_file, "r") as f:
        runner_content = f.read()

    runner_sentinel = "MXFP8 MoE: padding hidden size from 2688 to 2816"
    if runner_sentinel not in runner_content:
        helper_anchor = "def _align_mxfp8_moe_weights(\n"
        helper = (
            "def _pad_mxfp8_moe_hidden_size_for_flashinfer_trtllm(\n"
            "    w13: torch.Tensor,\n"
            "    w13_scale: torch.Tensor,\n"
            "    w2: torch.Tensor,\n"
            "    w2_scale: torch.Tensor,\n"
            ") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:\n"
            '    """Pad Nemotron-H\'s MXFP8 hidden size for supported SM100 tactics."""\n'
            "    hidden_size = w13.shape[2]\n"
            "    if hidden_size != 2688:\n"
            "        return w13, w13_scale, w2, w2_scale, hidden_size\n\n"
            "    if (\n"
            "        w2.shape[1] != hidden_size\n"
            "        or w13_scale.shape[2] * 32 != hidden_size\n"
            "        or w2_scale.shape[1] != hidden_size\n"
            "    ):\n"
            "        raise RuntimeError(\n"
            '            "Inconsistent MXFP8 MoE hidden dimensions before FlashInfer "\n'
            '            f"padding: w13={tuple(w13.shape)}, "\n'
            '            f"w13_scale={tuple(w13_scale.shape)}, "\n'
            '            f"w2={tuple(w2.shape)}, w2_scale={tuple(w2_scale.shape)}."\n'
            "        )\n\n"
            "    padded_hidden_size = 2816\n"
            "    logger.warning(\n"
            '        "MXFP8 MoE: padding hidden size from 2688 to 2816 for "\n'
            '        "FlashInfer TRT-LLM kernel compatibility."\n'
            "    )\n\n"
            "    padded_w13 = w13.new_zeros(\n"
            "        (w13.shape[0], w13.shape[1], padded_hidden_size)\n"
            "    )\n"
            "    padded_w13[:, :, :hidden_size] = w13\n"
            "    padded_w13_scale = w13_scale.new_full(\n"
            "        (w13_scale.shape[0], w13_scale.shape[1], padded_hidden_size // 32),\n"
            "        1,\n"
            "    )\n"
            "    padded_w13_scale[:, :, : w13_scale.shape[2]] = w13_scale\n\n"
            "    padded_w2 = w2.new_zeros(\n"
            "        (w2.shape[0], padded_hidden_size, w2.shape[2])\n"
            "    )\n"
            "    padded_w2[:, :hidden_size, :] = w2\n"
            "    padded_w2_scale = w2_scale.new_full(\n"
            "        (w2_scale.shape[0], padded_hidden_size, w2_scale.shape[2]), 1\n"
            "    )\n"
            "    padded_w2_scale[:, :hidden_size, :] = w2_scale\n\n"
            "    return (\n"
            "        padded_w13,\n"
            "        padded_w13_scale,\n"
            "        padded_w2,\n"
            "        padded_w2_scale,\n"
            "        padded_hidden_size,\n"
            "    )\n\n\n" + helper_anchor
        )
        align_anchor = (
            "    # Pad for kernel alignment (non-gated needs 128, gated needs 16)\n"
            "    min_alignment = 16 if is_gated else 128\n"
            "    w13_weight, w13_scale, w2_weight, w2_scale, _ = _align_mxfp8_moe_weights(\n"
            "        w13_weight, w13_scale, w2_weight, w2_scale, is_gated, min_alignment\n"
            "    )\n"
        )
        align_replacement = (
            "    (\n"
            "        w13_weight,\n"
            "        w13_scale,\n"
            "        w2_weight,\n"
            "        w2_scale,\n"
            "        _,\n"
            "    ) = _pad_mxfp8_moe_hidden_size_for_flashinfer_trtllm(\n"
            "        w13_weight, w13_scale, w2_weight, w2_scale\n"
            "    )\n\n" + align_anchor
        )
        input_anchor = (
            "    hidden_states = dispatch_output.hidden_states\n"
            "    topk_output = dispatch_output.topk_output\n"
            "    if TopKOutputChecker.format_is_bypassed(topk_output):\n"
        )
        input_replacement = (
            "    hidden_states = dispatch_output.hidden_states\n"
            "    logical_hidden_size = hidden_states.shape[1]\n"
            "    kernel_hidden_size = quant_info.w13_weight.shape[2]\n"
            "    hidden_size_is_padded = False\n"
            "    if quant_info.use_mxfp8 and kernel_hidden_size != logical_hidden_size:\n"
            "        if (logical_hidden_size, kernel_hidden_size) != (2688, 2816):\n"
            "            raise RuntimeError(\n"
            '                "Unsupported MXFP8 MoE hidden-size padding: "\n'
            '                f"logical={logical_hidden_size}, kernel={kernel_hidden_size}."\n'
            "            )\n"
            "        if quant_info.w2_weight.shape[1] != kernel_hidden_size:\n"
            "            raise RuntimeError(\n"
            '                "Inconsistent padded MXFP8 MoE weight shapes: "\n'
            '                f"w13={tuple(quant_info.w13_weight.shape)}, "\n'
            '                f"w2={tuple(quant_info.w2_weight.shape)}."\n'
            "            )\n"
            "        hidden_states = torch.nn.functional.pad(\n"
            "            hidden_states, (0, kernel_hidden_size - logical_hidden_size)\n"
            "        )\n"
            "        hidden_size_is_padded = True\n"
            "    topk_output = dispatch_output.topk_output\n"
            "    if TopKOutputChecker.format_is_bypassed(topk_output):\n"
        )
        scale_anchor = "            a_sf_t = a_sf.view(torch.uint8).reshape(hidden_states.shape[0], -1)\n"
        scale_replacement = (
            scale_anchor + "            if hidden_size_is_padded:\n"
            "                a_sf_t[:, logical_hidden_size // 32 :].fill_(1)\n"
        )
        output_anchor = "        output = symm_output\n    else:\n"
        output_replacement = (
            "        output = symm_output\n"
            "        if hidden_size_is_padded:\n"
            "            with use_symmetric_memory(\n"
            "                get_tp_group(), disabled=not is_allocation_symmetric()\n"
            "            ):\n"
            "                output = output[:, :logical_hidden_size].contiguous()\n"
            "    else:\n"
        )

        for anchor, replacement, expected_count in (
            (helper_anchor, helper, 1),
            (align_anchor, align_replacement, 1),
            (input_anchor, input_replacement, 1),
            (scale_anchor, scale_replacement, 1),
            (output_anchor, output_replacement, 1),
        ):
            actual_count = runner_content.count(anchor)
            if actual_count != expected_count:
                raise RuntimeError(
                    "SGLang MXFP8 hidden-size runner compat-patch anchor mismatch in "
                    f"{runner_file}: expected {expected_count}, found {actual_count} "
                    f"for {anchor[:80]!r}."
                )
            runner_content = runner_content.replace(anchor, replacement, 1)

        _write_and_verify(runner_file, runner_content, runner_sentinel)
        logger.info("Patched MXFP8 MoE hidden-size handling in %s.", runner_file)

    with open(loader_file, "r") as f:
        loader_content = f.read()

    loader_sentinel = "_narrow_mxfp8_hidden_padding_for_load"
    if loader_sentinel not in loader_content:
        helper_anchor = "    def _load_w13(\n"
        helper = (
            "    def _narrow_mxfp8_hidden_padding_for_load(\n"
            "        self,\n"
            "        expert_data: torch.Tensor,\n"
            "        loaded_weight: torch.Tensor,\n"
            "        shard_id: str,\n"
            "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
            '        """Load canonical H=2688 data into persistent H=2816 buffers."""\n'
            "        if expert_data.shape == loaded_weight.shape:\n"
            "            return expert_data, loaded_weight\n"
            "        if not (\n"
            "            self.use_flashinfer_trtllm_moe\n"
            '            and getattr(self.quant_method, "use_mxfp8", False)\n'
            "            and expert_data.ndim == loaded_weight.ndim == 2\n"
            "        ):\n"
            "            return expert_data, loaded_weight\n\n"
            "        pad_value = 1 if expert_data.dtype == torch.uint8 else 0\n"
            "        if (\n"
            '            shard_id in {"w1", "w3", "w13"}\n'
            "            and expert_data.shape[0] == loaded_weight.shape[0]\n"
            "            and (expert_data.shape[1], loaded_weight.shape[1])\n"
            "            in {(2816, 2688), (88, 84)}\n"
            "        ):\n"
            "            expert_data[:, loaded_weight.shape[1] :].fill_(pad_value)\n"
            "            return expert_data[:, : loaded_weight.shape[1]], loaded_weight\n\n"
            "        if (\n"
            '            shard_id == "w2"\n'
            "            and expert_data.shape[1] == loaded_weight.shape[1]\n"
            "            and (expert_data.shape[0], loaded_weight.shape[0])\n"
            "            == (2816, 2688)\n"
            "        ):\n"
            "            expert_data[loaded_weight.shape[0] :, :].fill_(pad_value)\n"
            "            return expert_data[: loaded_weight.shape[0], :], loaded_weight\n\n"
            "        return expert_data, loaded_weight\n\n" + helper_anchor
        )
        w13_anchor = "        expert_data.copy_(loaded_weight)\n\n    def _load_w2(\n"
        w13_replacement = (
            "        expert_data, loaded_weight = (\n"
            "            self._narrow_mxfp8_hidden_padding_for_load(\n"
            "                expert_data, loaded_weight, shard_id\n"
            "            )\n"
            "        )\n"
            "        expert_data.copy_(loaded_weight)\n\n"
            "    def _load_w2(\n"
        )
        w2_anchor = (
            "        # w2, down_proj: Load into only logical weight of w2.\n"
            "        expert_data.copy_(loaded_weight)\n\n"
            "    def _maybe_load_fp8_shared_expert_as_fp4(\n"
        )
        w2_replacement = (
            "        # w2, down_proj: Load into only logical weight of w2.\n"
            "        expert_data, loaded_weight = (\n"
            "            self._narrow_mxfp8_hidden_padding_for_load(\n"
            "                expert_data, loaded_weight, shard_id\n"
            "            )\n"
            "        )\n"
            "        expert_data.copy_(loaded_weight)\n\n"
            "    def _maybe_load_fp8_shared_expert_as_fp4(\n"
        )

        for anchor, replacement in (
            (helper_anchor, helper),
            (w13_anchor, w13_replacement),
            (w2_anchor, w2_replacement),
        ):
            actual_count = loader_content.count(anchor)
            if actual_count != 1:
                raise RuntimeError(
                    "SGLang MXFP8 hidden-size loader compat-patch anchor mismatch in "
                    f"{loader_file}: expected 1, found {actual_count} for "
                    f"{anchor[:80]!r}."
                )
            loader_content = loader_content.replace(anchor, replacement, 1)

        _write_and_verify(loader_file, loader_content, loader_sentinel)
        logger.info("Patched MXFP8 MoE hidden-size refit loading in %s.", loader_file)


def _override_sglang_imbalance_check_env() -> None:
    """Force-disable sglang's per-GPU memory imbalance check.

    Pop the legacy names so the shim has nothing to copy, then set
    ``ENABLE=false`` directly. Inherited env reaches the subprocesses
    cleaned, so the shim no longer overwrites our ENABLE on re-import.
    """
    for legacy in (
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    ):
        os.environ.pop(legacy, None)
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"


def _get_megatron_file(subpackage: str, relative_path: str) -> str | None:
    """Locate a file inside ``megatron.<subpackage>`` (e.g. ``core``, ``training``).

    Returns ``None`` if megatron isn't importable so callers can treat that
    as "nothing to patch". Raises if the package is present but the
    expected file is missing (signals a megatron version mismatch).
    """
    full_pkg = f"megatron.{subpackage}"
    try:
        spec = find_spec(full_pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected megatron file '{full_pkg}/{relative_path}' not found at "
            f"'{file_path}'. The megatron version may have moved this file; "
            "compat patch cannot be applied."
        )
    return file_path


def _patch_megatron_hook_mode_in(file_path: str) -> None:
    """Comment out ``torch_memory_saver.hook_mode = "torch"`` in a megatron file.

    Megatron sets ``tms.hook_mode = "torch"`` at module import time on the
    global ``torch_memory_saver`` singleton. That mutation breaks sglang's
    pauseable CUDA graph path, which asserts ``_hook_mode == "preload"``
    inside ``TorchMemorySaver.cuda_graph(...)``. Commenting the line out
    leaves the singleton at its default ``"preload"`` mode that sglang
    expects.
    """
    with open(file_path, "r") as f:
        content = f.read()

    sentinel = '# torch_memory_saver.hook_mode = "torch"'
    if sentinel in content:
        return

    anchor = '    torch_memory_saver.hook_mode = "torch"\n'
    if anchor not in content:
        raise RuntimeError(
            f"Megatron hook_mode anchor '{anchor.strip()}' not found in "
            f"{file_path}; the megatron version may have moved or removed it."
        )

    replacement = (
        '    # torch_memory_saver.hook_mode = "torch"  '
        "# patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n"
    )
    content = content.replace(anchor, replacement, 1)
    _write_and_verify(file_path, content, sentinel)
    logger.info("Patched megatron tms.hook_mode mutation in %s.", file_path)


def _patch_megatron_dynamic_context_hook_mode() -> None:
    file_path = _get_megatron_file("core", "inference/contexts/dynamic_context.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_megatron_training_hook_mode() -> None:
    file_path = _get_megatron_file("training", "training.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _apply_sglang_compat_patches() -> None:
    _patch_sglang_safe_unpickler()
    _patch_sglang_non_gated_fp8_moe()
    _patch_sglang_mxfp8_moe_scale_layout()
    _patch_sglang_mxfp8_moe_hidden_size()
    _override_sglang_imbalance_check_env()
    _patch_megatron_dynamic_context_hook_mode()
    _patch_megatron_training_hook_mode()
