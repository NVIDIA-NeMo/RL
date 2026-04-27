#!/usr/bin/env bash
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
set -euo pipefail

PYTHON_BIN="${1:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python}"

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import vllm.model_executor.layers.deepseek_v4_attention as dsv4_attention
import vllm.model_executor.models.deepseek_v4 as dsv4


def patch_file(path: Path, replacements: list[tuple[str, str] | tuple[str, str, bool]]) -> None:
    source = path.read_text()
    original = source

    for item in replacements:
        old, new = item[0], item[1]
        optional = bool(item[2]) if len(item) > 2 else False
        if new in source:
            continue
        if old not in source:
            if optional:
                print(f"optional target not found, skipping in {path}:\n{old}")
                continue
            raise SystemExit(
                "target block not found; vLLM source may have changed or was patched differently:\n"
                f"{path}\nmissing:\n{old}"
            )
        source = source.replace(old, new, 1)

    if source == original:
        print(f"already patched: {path}")
        return

    backup = path.with_suffix(path.suffix + ".base-fp8.bak")
    if not backup.exists():
        backup.write_text(original)

    path.write_text(source)
    print(f"patched: {path}")
    print(f"backup : {backup}")


def patch_dsv4_for_causal_lm_mapper(path: Path) -> None:
    source = path.read_text()
    class_marker = "class DeepseekV4ForCausalLM(nn.Module):\n"
    if class_marker not in source:
        raise SystemExit(f"cannot find DeepseekV4ForCausalLM in {path}")

    before_class, class_source = source.split(class_marker, 1)
    next_class_index = class_source.find("\nclass ")
    if next_class_index == -1:
        class_body = class_source
        after_class = ""
    else:
        class_body = class_source[:next_class_index]
        after_class = class_source[next_class_index:]

    if "hf_to_vllm_mapper_base_fp8 = WeightsMapper(" in class_body:
        print(f"DeepseekV4ForCausalLM already has base fp8 mapper: {path}")
        return

    init_marker = "\n    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
    if init_marker not in class_body:
        raise SystemExit(f"cannot find DeepseekV4ForCausalLM.__init__ in {path}")

    mapper = (
        "\n"
        "    hf_to_vllm_mapper_base_fp8 = WeightsMapper(\n"
        "        orig_to_new_prefix={\n"
        "            \"layers.\": \"model.layers.\",\n"
        "            \"embed.\": \"model.embed.\",\n"
        "            \"norm.\": \"model.norm.\",\n"
        "            \"hc_head\": \"model.hc_head\",\n"
        "            \"mtp.\": \"model.mtp.\",\n"
        "        },\n"
        "        orig_to_new_regex={\n"
        "            # Base routed MoE experts are raw block-FP8, not MXFP4.\n"
        "            # Fp8MoEMethod(block_quant=True) expects weight_scale_inv.\n"
        "            re.compile(r\"\\.scale$\"): \".weight_scale_inv\",\n"
        "        },\n"
        "        orig_to_new_suffix={\n"
        "            \"head.weight\": \"lm_head.weight\",\n"
        "            \"embed.weight\": \"embed_tokens.weight\",\n"
        "            \".ffn.gate.bias\": \".ffn.gate.e_score_correction_bias\",\n"
        "        },\n"
        "        orig_to_new_substr={\n"
        "            \".attn.compressor.\": \".attn.mla_attn.compressor.\",\n"
        "            \".shared_experts.w2\": \".shared_experts.down_proj\",\n"
        "        },\n"
        "    )\n"
    )
    class_body = class_body.replace(init_marker, mapper + init_marker, 1)
    backup = path.with_suffix(path.suffix + ".base-fp8.bak")
    if not backup.exists():
        backup.write_text(source)
    path.write_text(before_class + class_marker + class_body + after_class)
    print(f"patched mapper into DeepseekV4ForCausalLM: {path}")
    print(f"backup : {backup}")


patch_file(
    Path(dsv4.__file__),
    [
        (
            "import typing\n",
            "import os\nimport typing\n",
        ),
        (
            "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n",
            (
                "from vllm.model_executor.layers.quantization.fp8 import (\n"
                "    Fp8Config,\n"
                "    Fp8MoEMethod,\n"
                ")\n"
            ),
        ),
        (
            "\n\nclass DeepseekV4FP8Config(Fp8Config):\n",
            (
                "\n\n"
                "def _is_dsv4_base_fp8_enabled() -> bool:\n"
                "    value = os.environ.get(\"VLLM_DSV4_BASE_FP8\", \"\")\n"
                "    return value.lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
                "\n\n"
                "class DeepseekV4FP8Config(Fp8Config):\n"
            ),
        ),
        (
            "        self.is_scale_e8m0: bool = True\n",
            "        self.is_scale_e8m0: bool = not _is_dsv4_base_fp8_enabled()\n",
        ),
        (
            "            return Mxfp4MoEMethod(layer.moe_config)\n",
            (
                "            if _is_dsv4_base_fp8_enabled():\n"
                "                return Fp8MoEMethod(self, layer)\n"
                "            return Mxfp4MoEMethod(layer.moe_config)\n"
            ),
        ),
        (
            "        return isinstance(layer, FusedMoE)\n",
            "        return isinstance(layer, FusedMoE) and not _is_dsv4_base_fp8_enabled()\n",
        ),
        (
            # Anchor for the post-#40860 main wheel structure (load_weights now
            # captures the result and calls finalize_mega_moe_weights() before
            # returning, so we hook the assignment line). The earlier internal
            # builds (g306b63f67 / g62d441ee8) had a single `return loader...`
            # line that this anchor will not match — for those wheels, replace
            # with the prior anchor:
            #   "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
            #   -> "        mapper = (...)\n        return loader.load_weights(weights, mapper=mapper)\n"
            "        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n",
            (
                "        mapper = (\n"
                "            self.hf_to_vllm_mapper_base_fp8\n"
                "            if _is_dsv4_base_fp8_enabled()\n"
                "            else self.hf_to_vllm_mapper\n"
                "        )\n"
                "        loaded_params = loader.load_weights(weights, mapper=mapper)\n"
            ),
        ),
        (
            # Force use_mega_moe = False under VLLM_DSV4_BASE_FP8. Base routes
            # routed experts through Fp8MoEMethod (FusedMoE), not the
            # DeepseekV4MegaMoEExperts class. Without this, the post-load call
            # `self.model.finalize_mega_moe_weights()` would invoke
            # `experts.finalize_weights()` on a non-MegaMoE layer (no such
            # attr) and crash. Keeping this gating self-contained inside the
            # patch means users only need to set the env var; recipe-level
            # `moe_backend` overrides are not also required.
            (
                "        if vllm_config.parallel_config.enable_expert_parallel:\n"
                "            self.use_mega_moe = (\n"
                "                vllm_config.kernel_config.moe_backend == \"deep_gemm_mega_moe\"\n"
                "            )\n"
                "        else:\n"
                "            self.use_mega_moe = False\n"
            ),
            (
                "        if _is_dsv4_base_fp8_enabled():\n"
                "            # Base FP8 routes routed experts through Fp8MoEMethod,\n"
                "            # not DeepseekV4MegaMoEExperts. Force the MegaMoE finalize\n"
                "            # path off so finalize_mega_moe_weights() is a no-op.\n"
                "            self.use_mega_moe = False\n"
                "        elif vllm_config.parallel_config.enable_expert_parallel:\n"
                "            self.use_mega_moe = (\n"
                "                vllm_config.kernel_config.moe_backend == \"deep_gemm_mega_moe\"\n"
                "            )\n"
                "        else:\n"
                "            self.use_mega_moe = False\n"
            ),
        ),
    ],
)

patch_dsv4_for_causal_lm_mapper(Path(dsv4.__file__))

patch_file(
    Path(dsv4_attention.__file__),
    [
        (
            "from dataclasses import dataclass\n",
            "import os\nfrom dataclasses import dataclass\n",
        ),
        (
            "\n\n@dataclass\nclass DeepseekV4MLAModules:\n",
            (
                "\n\n"
                "def _is_dsv4_base_fp8_enabled() -> bool:\n"
                "    value = os.environ.get(\"VLLM_DSV4_BASE_FP8\", \"\")\n"
                "    return value.lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
                "\n\n"
                "@dataclass\nclass DeepseekV4MLAModules:\n"
            ),
        ),
        (
            "        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)\n",
            (
                "        if _is_dsv4_base_fp8_enabled():\n"
                "            # DeepSeek-V4-Flash-Base checkpoints do not contain\n"
                "            # indexer.k_norm weights, and this module is not used by\n"
                "            # DeepseekV4Indexer.forward in this vLLM path.\n"
                "            self.k_norm = nn.Identity()\n"
                "        else:\n"
                "            self.k_norm = LayerNorm(self.head_dim, eps=1e-6)\n"
            ),
            True,
        ),
    ],
)

print("Set VLLM_DSV4_BASE_FP8=1 for DeepSeek-V4-Flash-Base FP8 tests.")
PY
