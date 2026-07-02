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

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest


def _load_smoke_module() -> Any:
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[3]
    smoke_path = (
        repo_root / "tools" / "model_diagnostics" / "kimi_k2_6_generation_smoke.py"
    )
    assert smoke_path.exists(), f"Expected diagnostic at {smoke_path}"
    spec = importlib.util.spec_from_file_location(
        "kimi_k2_6_generation_smoke", str(smoke_path)
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def smoke() -> Any:
    return _load_smoke_module()


def _minimal_kimi_config() -> dict[str, Any]:
    return {
        "policy": {
            "model_name": "moonshotai/Kimi-K2.6",
            "refit_offload_max_workers_per_node": 1,
            "megatron_cfg": {
                "tensor_model_parallel_size": 8,
                "pipeline_model_parallel_size": 8,
                "expert_model_parallel_size": 8,
                "expert_tensor_parallel_size": 1,
            },
            "generation": {
                "backend": "vllm",
                "max_new_tokens": 16,
                "vllm_cfg": {
                    "tensor_parallel_size": 8,
                    "pipeline_parallel_size": 1,
                    "expert_parallel_size": 64,
                    "load_format": "dummy",
                    "ipc_refit_metadata_in_payload": True,
                    "max_model_len": 128,
                },
            },
        },
        "cluster": {
            "num_nodes": 16,
            "gpus_per_node": 4,
        },
    }


def test_preflight_summary_accepts_expected_kimi_shape(smoke: Any) -> None:
    summary = smoke._preflight_summary(_minimal_kimi_config())

    smoke._validate_preflight(summary)

    assert summary["policy_parallelism"] == {"tp": 8, "pp": 8, "ep": 8, "etp": 1}
    assert summary["vllm_parallelism"] == {"tp": 8, "pp": 1, "ep": 64}
    assert summary["vllm_load_format"] == "dummy"


def test_preflight_rejects_non_dummy_load_format(smoke: Any) -> None:
    config = _minimal_kimi_config()
    config["policy"]["generation"]["vllm_cfg"]["load_format"] = "auto"
    summary = smoke._preflight_summary(config)

    with pytest.raises(ValueError, match="vllm_load_format"):
        smoke._validate_preflight(summary)


def test_preflight_accepts_local_kimi_snapshot_path(smoke: Any) -> None:
    config = _minimal_kimi_config()
    config["policy"][
        "model_name"
    ] = "/cache/hub/models--moonshotai--Kimi-K2.6/snapshots/abcdef"
    summary = smoke._preflight_summary(config)

    smoke._validate_preflight(summary)


def test_validate_generated_texts_accepts_readable_outputs(smoke: Any) -> None:
    smoke.validate_generated_texts(
        ["Solve 2 + 3.", "Say hello."],
        ["The answer is 5.", "Hello there."],
        min_output_chars=4,
    )


@pytest.mark.parametrize(
    "texts,match",
    [
        (["", "Hello there."], "too short"),
        (["dummy", "Hello there."], "placeholder"),
        (["aaaaaa", "Hello there."], "character diversity"),
    ],
)
def test_validate_generated_texts_rejects_bad_outputs(
    smoke: Any, texts: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        smoke.validate_generated_texts(
            ["Prompt one.", "Prompt two."],
            texts,
            min_output_chars=4,
        )


def test_validate_generated_texts_rejects_count_mismatch(smoke: Any) -> None:
    with pytest.raises(ValueError, match="Expected 2 outputs"):
        smoke.validate_generated_texts(
            ["Prompt one.", "Prompt two."],
            ["Only one output."],
            min_output_chars=4,
        )


def test_preflight_rejects_wrong_parallelism(smoke: Any) -> None:
    config = deepcopy(_minimal_kimi_config())
    config["policy"]["megatron_cfg"]["pipeline_model_parallel_size"] = 4
    summary = smoke._preflight_summary(config)

    with pytest.raises(ValueError, match="Unexpected policy parallelism"):
        smoke._validate_preflight(summary)
