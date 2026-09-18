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
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation.sglang.config import (
    SGLangFaultToleranceConfig,
    SGLangHttpClientConfig,
    get_sglang_fault_tolerance_config,
)
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration


def test_fault_tolerance_defaults_are_centralized_and_disabled():
    expected = {
        "use_fault_tolerance": False,
        "rollout_health_check_interval": 60.0,
        "rollout_health_check_timeout": 60.0,
        "rollout_health_check_first_wait": 60.0,
        "rollout_max_restart_attempts": 3,
    }
    assert SGLangFaultToleranceConfig().model_dump() == expected
    assert get_sglang_fault_tolerance_config({}).model_dump() == expected
    assert (
        get_sglang_fault_tolerance_config(
            {"sglang_fault_tolerance_config": {}}
        ).model_dump()
        == expected
    )


@pytest.mark.parametrize("nested", [False, True])
def test_fault_tolerance_normalizes_both_spellings_without_mutation(nested):
    values = {
        "use_fault_tolerance": True,
        "rollout_health_check_interval": 0.5,
        "rollout_health_check_timeout": 2,
        "rollout_health_check_first_wait": 0,
        "rollout_max_restart_attempts": 0,
    }
    config = {"sglang_fault_tolerance_config": values} if nested else values.copy()
    config["model_path"] = "unrelated/model"
    original = deepcopy(config)

    actual = get_sglang_fault_tolerance_config(config)

    assert actual.model_dump() == values
    assert config == original


def test_fault_tolerance_accepts_an_already_validated_model():
    config = SGLangFaultToleranceConfig(use_fault_tolerance=True)
    assert (
        get_sglang_fault_tolerance_config({"sglang_fault_tolerance_config": config})
        is config
    )


@pytest.mark.parametrize(
    "legacy_key",
    [
        "use_fault_tolerance",
        "rollout_health_check_interval",
        "rollout_health_check_timeout",
        "rollout_health_check_first_wait",
        "rollout_max_restart_attempts",
    ],
)
def test_fault_tolerance_rejects_mixed_spellings_even_when_values_agree(legacy_key):
    defaults = SGLangFaultToleranceConfig().model_dump()
    with pytest.raises(ValueError, match="Do not mix"):
        get_sglang_fault_tolerance_config(
            {
                "sglang_fault_tolerance_config": defaults,
                legacy_key: defaults[legacy_key],
            }
        )


@pytest.mark.parametrize("nested", [False, True])
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
def test_fault_tolerance_rejects_invalid_values_in_both_spellings(nested, key, value):
    values = {key: value}
    config = {"sglang_fault_tolerance_config": values} if nested else values
    with pytest.raises(ValidationError, match=key):
        get_sglang_fault_tolerance_config(config)


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
    config = {"sglang_cfg": {}}
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
