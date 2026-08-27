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
    _override_sglang_imbalance_check_env()
    _patch_megatron_dynamic_context_hook_mode()
    _patch_megatron_training_hook_mode()
