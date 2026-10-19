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

"""CPU-only validation of driver-owned SGLang runtime settings."""

import gc
import sys
import weakref
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation.sglang.config import (
    SGLangFaultToleranceConfig,
    SGLangHttpClientConfig,
    SglangSpecificArgs,
)
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.sglang import sglang_worker
from nemo_rl.utils.config import load_config


@pytest.fixture
def worker_for_server_args(monkeypatch):
    monkeypatch.setattr(sglang_worker, "_apply_sglang_compat_patches", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker_class = sglang_worker.SGLangGenerationWorker.__ray_metadata__.modified_class
    return worker_class(
        gpus_per_node=1,
        sglang_cfg={
            "sglang_cfg": {
                "model_path": "dummy",
                "random_seed": 42,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "ep_size": 1,
                "skip_server_warmup": True,
                "sglang_server_config": {
                    "num_gpus_per_engine": 1,
                    "needs_offload": False,
                    "cpu_weight_backup": False,
                },
            }
        },
        rank=0,
        base_gpu_id=0,
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        pytest.param({}, True, id="omitted-disables-cache"),
        pytest.param(
            {"disable_outlines_disk_cache": True}, True, id="explicit-disable"
        ),
        pytest.param(
            {"disable_outlines_disk_cache": False}, False, id="explicit-opt-in"
        ),
    ],
)
def test_worker_forwards_outlines_cache_policy(
    worker_for_server_args, overrides, expected
):
    worker_for_server_args.sglang_cfg["sglang_cfg"].update(overrides)

    args = worker_for_server_args._compute_server_args(
        dist_init_addr="127.0.0.1:1500",
        nccl_port=1501,
        host="127.0.0.1",
        port=1502,
    )

    assert args["disable_outlines_disk_cache"] is expected


@pytest.mark.parametrize("value", [None, "false", 0])
def test_worker_rejects_non_boolean_cache_policy(worker_for_server_args, value):
    worker_for_server_args.sglang_cfg["sglang_cfg"]["disable_outlines_disk_cache"] = (
        value
    )

    with pytest.raises(
        ValueError, match="disable_outlines_disk_cache must be a boolean"
    ):
        worker_for_server_args._compute_server_args(
            dist_init_addr="127.0.0.1:1500",
            nccl_port=1501,
            host="127.0.0.1",
            port=1502,
        )


def test_fault_tolerance_defaults_are_centralized_and_disabled():
    expected = {
        "use_fault_tolerance": False,
        "rollout_health_check_interval": 60.0,
        "rollout_health_check_timeout": 60.0,
        "rollout_health_check_first_wait": 60.0,
        "rollout_max_restart_attempts": 3,
    }
    assert SGLangFaultToleranceConfig().model_dump() == expected
    assert SGLangFaultToleranceConfig.model_validate({}).model_dump() == expected


def test_fault_tolerance_validates_nested_settings_without_mutation():
    values = {
        "use_fault_tolerance": True,
        "rollout_health_check_interval": 0.5,
        "rollout_health_check_timeout": 2,
        "rollout_health_check_first_wait": 0,
        "rollout_max_restart_attempts": 0,
    }
    config = {"sglang_fault_tolerance_config": values, "model_path": "unrelated/model"}
    original = deepcopy(config)

    actual = SGLangFaultToleranceConfig.model_validate(
        config["sglang_fault_tolerance_config"]
    )

    assert actual.model_dump() == values
    assert config == original


def test_fault_tolerance_accepts_an_already_validated_model():
    config = SGLangFaultToleranceConfig(use_fault_tolerance=True)
    assert SGLangFaultToleranceConfig.model_validate(config) is config


def test_fault_tolerance_block_is_required_without_flat_fields():
    assert "sglang_fault_tolerance_config" in SglangSpecificArgs.__required_keys__
    assert not SGLangFaultToleranceConfig.model_fields.keys() & (
        SglangSpecificArgs.__annotations__.keys()
    )


@pytest.mark.parametrize(
    "config_path",
    sorted(
        (Path(__file__).resolve().parents[5] / "examples/configs").rglob(
            "*sglang*.yaml"
        )
    ),
    ids=lambda path: path.name,
)
def test_sglang_recipes_inherit_required_fault_tolerance_block(config_path):
    config = load_config(config_path)
    sglang_cfg = config.policy.generation.sglang_cfg
    fault_tolerance = SGLangFaultToleranceConfig.model_validate(
        dict(sglang_cfg["sglang_fault_tolerance_config"])
    )
    assert not fault_tolerance.use_fault_tolerance
    assert not SGLangFaultToleranceConfig.model_fields.keys() & sglang_cfg.keys()


def test_missing_fault_tolerance_block_fails_before_cluster_allocation(monkeypatch):
    cluster = MagicMock()
    loop_factory = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.AsyncLoopThread",
        loop_factory,
    )
    with pytest.raises(KeyError, match="sglang_fault_tolerance_config"):
        SGLangGeneration(cluster, {"sglang_cfg": {}})
    cluster._init_placement_groups.assert_not_called()
    loop_factory.assert_not_called()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("use_fault_tolerance", "true"),
        ("rollout_health_check_interval", 0),
        ("rollout_health_check_interval", -1),
        ("rollout_health_check_interval", True),
        ("rollout_health_check_interval", float("inf")),
        ("rollout_health_check_timeout", 0),
        ("rollout_health_check_timeout", "60"),
        ("rollout_health_check_timeout", float("nan")),
        ("rollout_health_check_first_wait", -1),
        ("rollout_health_check_first_wait", float("inf")),
        ("rollout_max_restart_attempts", -1),
        ("rollout_max_restart_attempts", True),
        ("rollout_max_restart_attempts", 1.5),
    ],
)
def test_fault_tolerance_rejects_invalid_values(key, value):
    with pytest.raises(ValidationError, match=key):
        SGLangFaultToleranceConfig.model_validate({key: value})


def test_invalid_fault_tolerance_fails_before_cluster_allocation(monkeypatch):
    cluster = MagicMock()
    loop_factory = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.AsyncLoopThread",
        loop_factory,
    )
    with pytest.raises(ValidationError, match="rollout_max_restart_attempts"):
        SGLangGeneration(
            cluster,
            {
                "sglang_cfg": {
                    "sglang_fault_tolerance_config": {
                        "rollout_max_restart_attempts": -1
                    }
                }
            },
        )
    cluster._init_placement_groups.assert_not_called()
    loop_factory.assert_not_called()


@pytest.mark.parametrize("invalid_config", [True, False])
def test_failed_constructor_cleanup_is_repeatable_and_destructor_safe(
    monkeypatch, invalid_config
):
    cluster = MagicMock()

    def fail_placement(**kwargs):
        # A stored exception instance would retain the failed constructor's
        # traceback and keep the generation object alive during gc.collect().
        raise RuntimeError("placement failed")

    cluster._init_placement_groups.side_effect = fail_placement
    loop = MagicMock()
    loop_factory = MagicMock(return_value=loop)
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.AsyncLoopThread",
        loop_factory,
    )
    http_factory = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.HttpClient",
        http_factory,
    )
    unraisable_errors = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable_errors.append)
    destructor_calls = []
    original_destructor = SGLangGeneration.__del__

    def record_destructor(instance):
        destructor_calls.append(True)
        original_destructor(instance)

    monkeypatch.setattr(SGLangGeneration, "__del__", record_destructor)
    config = {"sglang_cfg": {"sglang_fault_tolerance_config": {}}}
    if invalid_config:
        config["sglang_cfg"]["sglang_fault_tolerance_config"] = {
            "rollout_max_restart_attempts": -1
        }
    expected_error = ValidationError if invalid_config else RuntimeError
    expected_message = (
        "rollout_max_restart_attempts" if invalid_config else "placement failed"
    )
    # Keep the failed instance reachable so cleanup is checked synchronously,
    # then let real garbage collection invoke its actual destructor too.
    generation = object.__new__(SGLangGeneration)
    with pytest.raises(expected_error, match=expected_message):
        generation.__init__(cluster, config)

    assert generation.shutdown()
    assert generation.shutdown()
    http_factory.assert_not_called()
    if invalid_config:
        cluster._init_placement_groups.assert_not_called()
        loop_factory.assert_not_called()
        loop.close.assert_not_called()
    else:
        cluster._init_placement_groups.assert_called_once_with(
            strategy="PACK", use_unified_pg=True
        )
        loop_factory.assert_called_once_with()
        loop.close.assert_called_once_with()

    reference = weakref.ref(generation)
    del generation
    gc.collect()
    assert reference() is None
    assert destructor_calls == [True]
    assert unraisable_errors == []
    if not invalid_config:
        loop.close.assert_called_once_with()


def test_http_client_default_counts_total_attempts():
    assert SGLangHttpClientConfig().max_retries == 3
    assert SGLangHttpClientConfig(max_retries=1).max_retries == 1


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "3", None])
def test_http_client_rejects_invalid_attempt_budget(budget):
    with pytest.raises(ValidationError, match="max_retries"):
        SGLangHttpClientConfig(max_retries=budget)
