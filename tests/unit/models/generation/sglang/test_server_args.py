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

"""CPU tests for native server options and NeMo-RL placement invariants."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from nemo_rl.models.generation.sglang import sglang_worker
from nemo_rl.utils.config import load_config


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    config = load_config(
        Path(__file__).resolve().parents[5]
        / "examples/configs/grpo_math_1B_sglang.yaml"
    ).policy.generation
    monkeypatch.setattr(sglang_worker, "_apply_sglang_compat_patches", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker_class = sglang_worker.SGLangGenerationWorker.__ray_metadata__.modified_class
    return worker_class(
        gpus_per_node=1,
        sglang_cfg=OmegaConf.to_container(config, resolve=True),
        rank=0,
        base_gpu_id=0,
    )


def _server_args(worker: Any) -> dict[str, Any]:
    return worker._compute_server_args(
        dist_init_addr="127.0.0.1:1500",
        nccl_port=1501,
        host="127.0.0.1",
        port=1502,
    )


def test_extra_native_server_options_are_forwarded_without_mutation(
    worker: Any,
) -> None:
    extra = {
        "attention_backend": "triton",
        "sampling_backend": "pytorch",
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "disable_custom_all_reduce": True,
        "max_mamba_cache_size": 4,
        "custom_weight_loader": ["example.models.load_weights"],
    }
    worker.sglang_cfg["sglang_kwargs"] = extra
    original = deepcopy(worker.sglang_cfg)

    args = _server_args(worker)

    assert {key: args[key] for key in extra} == extra
    args["custom_weight_loader"].append("example.models.other_loader")
    assert worker.sglang_cfg == original


def test_empty_extra_options_preserve_server_defaults(worker: Any) -> None:
    expected = _server_args(worker)
    worker.sglang_cfg["sglang_kwargs"] = {}
    assert _server_args(worker) == expected


@pytest.mark.parametrize(
    "key",
    [
        "model_path",
        "trust_remote_code",
        "random_seed",
        "host",
        "port",
        "nccl_port",
        "dist_init_addr",
        "nnodes",
        "node_rank",
        "gpu_id_step",
        "base_gpu_id",
        "tp_size",
        "pp_size",
        "dp_size",
        "ep_size",
        "enable_memory_saver",
        "enable_weights_cpu_backup",
        "enable_draft_weights_cpu_backup",
        "disable_outlines_disk_cache",
        "skip_server_warmup",
        "dtype",
        "context_length",
        "mem_fraction_static",
        "disable_cuda_graph",
    ],
)
def test_conflicting_extra_options_fail_before_launch(worker: Any, key: str) -> None:
    worker.sglang_cfg["sglang_kwargs"] = {key: "conflicting-value"}
    with pytest.raises(ValueError, match=rf"sglang_kwargs\.{key} conflicts"):
        _server_args(worker)


def test_identical_duplicate_options_are_accepted(worker: Any) -> None:
    expected = _server_args(worker)
    worker.sglang_cfg["sglang_kwargs"] = {
        key: expected[key] for key in ("tp_size", "dtype", "context_length")
    }
    assert _server_args(worker) == expected


def test_extra_options_cannot_bypass_refit_precision_configuration(worker: Any) -> None:
    worker.sglang_cfg["sglang_kwargs"] = {"quantization": "fp8"}
    with pytest.raises(ValueError, match="sglang_cfg.quantization"):
        _server_args(worker)
