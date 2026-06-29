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

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERF_CONFIG_DIR = PROJECT_ROOT / "examples/configs/recipes/llm/performance"
PERF_SUITE_DIR = PROJECT_ROOT / "tests/test_suites/llm/performance"
GB200_SUITE = PROJECT_ROOT / "tests/test_suites/performance_gb200.txt"

MXFP8_CASES = {
    "grpo-qwen3-30ba3b-4n4g-mxfp8-rollout": {
        "nodes": 4,
        "gpus_per_node": 4,
        "segment_size": 4,
        "async_engine": None,
        "moe_backend": None,
    },
    "grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout": {
        "nodes": 4,
        "gpus_per_node": 4,
        "segment_size": 2,
        "async_engine": True,
        "moe_backend": None,
    },
    "grpo-qwen3-32b-4n4g-mxfp8-rollout": {
        "nodes": 4,
        "gpus_per_node": 4,
        "segment_size": 4,
        "async_engine": None,
        "moe_backend": None,
    },
    "grpo-qwen3-32b-8n4g-async-1off-mxfp8-rollout": {
        "nodes": 8,
        "gpus_per_node": 4,
        "segment_size": 4,
        "async_engine": True,
        "moe_backend": None,
    },
    "grpo-qwen3-235b-16n4g-mxfp8-rollout": {
        "nodes": 16,
        "gpus_per_node": 4,
        "segment_size": 16,
        "async_engine": True,
        "tensor_parallel_size": 4,
        "moe_backend": "flashinfer_trtllm",
    },
    "grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout": {
        "nodes": 32,
        "gpus_per_node": 4,
        "segment_size": 16,
        "async_engine": True,
        "tensor_parallel_size": 4,
        "moe_backend": "flashinfer_trtllm",
    },
}

REMOVED_BACKEND_VARIANT_SUFFIXES = (
    "-flashinfer-trtllm",
    "-triton",
    "-marlin",
)


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_resolved_yaml(path: Path, seen: set[Path] | None = None) -> dict:
    if seen is None:
        seen = set()
    path = path.resolve()
    assert path not in seen

    config = _load_yaml(path)
    defaults = config.get("defaults")
    if not isinstance(defaults, str):
        return config

    parent = (path.parent / defaults).resolve()
    base_config = _load_resolved_yaml(parent, seen | {path})
    return _deep_merge(base_config, config)


@pytest.mark.parametrize(("case_name", "expected"), MXFP8_CASES.items())
def test_mxfp8_rollout_recipe_matrix(case_name: str, expected: dict) -> None:
    config_path = PERF_CONFIG_DIR / f"{case_name}.yaml"
    script_path = PERF_SUITE_DIR / f"{case_name}.sh"

    assert config_path.is_file()
    assert script_path.is_file()

    config = _load_resolved_yaml(config_path)
    vllm_cfg = config["policy"]["generation"]["vllm_cfg"]
    cluster = config["cluster"]

    assert vllm_cfg["precision"] == "fp8"
    assert vllm_cfg["is_mx"] is True
    assert vllm_cfg["quantization_ignored_layer_kws"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert cluster["num_nodes"] == expected["nodes"]
    assert cluster["gpus_per_node"] == expected["gpus_per_node"]
    assert cluster["segment_size"] == expected["segment_size"]
    if expected.get("tensor_parallel_size") is not None:
        assert vllm_cfg["tensor_parallel_size"] == expected["tensor_parallel_size"]
    if expected["async_engine"] is not None:
        assert vllm_cfg["async_engine"] is expected["async_engine"]
    if expected["moe_backend"] is not None:
        assert (
            config["policy"]["generation"]["vllm_kwargs"]["moe_backend"]
            == expected["moe_backend"]
        )


def test_mxfp8_rollout_recipes_are_in_gb200_performance_suite() -> None:
    suite_text = GB200_SUITE.read_text(encoding="utf-8")

    for case_name in MXFP8_CASES:
        assert f"tests/test_suites/llm/performance/{case_name}.sh" in suite_text


def test_mxfp8_rollout_backend_variant_recipes_are_not_kept() -> None:
    for base_name in MXFP8_CASES:
        for suffix in REMOVED_BACKEND_VARIANT_SUFFIXES:
            variant_name = f"{base_name}{suffix}"
            assert not (PERF_CONFIG_DIR / f"{variant_name}.yaml").exists()
            assert not (PERF_SUITE_DIR / f"{variant_name}.sh").exists()


def test_qwen3_235b_async_tp4_baseline_overlay() -> None:
    config = _load_yaml(PERF_CONFIG_DIR / "grpo-qwen3-235b-32n4g-async-1off-tp4.yaml")

    assert config["defaults"] == "./grpo-qwen3-235b-32n4g-async-1off.yaml"
    assert config["policy"]["generation"]["vllm_cfg"]["tensor_parallel_size"] == 4
    assert config["cluster"]["segment_size"] == 16


def test_qwen3_235b_scripts_append_distributed_timeout_override() -> None:
    for case_name in (
        "grpo-qwen3-235b-16n4g-mxfp8-rollout",
        "grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout",
    ):
        script_text = (PERF_SUITE_DIR / f"{case_name}.sh").read_text(encoding="utf-8")

        assert "export RAY_CGRAPH_get_timeout=2400" in script_text
        assert (
            "+policy.generation.vllm_kwargs.distributed_timeout_seconds=2400"
            in script_text
        )
        assert (
            "\n    policy.generation.vllm_kwargs.distributed_timeout_seconds=2400 \\"
            not in script_text
        )


def test_qwen3_235b_mxfp8_recipes_keep_baseline_runtime_knobs() -> None:
    for case_name in (
        "grpo-qwen3-235b-16n4g-mxfp8-rollout",
        "grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout",
    ):
        recipe = _load_yaml(PERF_CONFIG_DIR / f"{case_name}.yaml")
        grpo_config = recipe.get("grpo", {})

        assert "max_num_steps" not in grpo_config
        assert "val_batch_size" not in grpo_config
        assert "max_val_samples" not in grpo_config
