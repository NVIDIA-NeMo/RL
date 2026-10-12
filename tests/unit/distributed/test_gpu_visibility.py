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

import importlib

import pytest

from nemo_rl.distributed.gpu_visibility import (
    ONE_GPU_PER_RANK_ENV,
    one_gpu_per_rank_enabled,
    resolve_local_device_index,
    single_gpu_worker_env_vars,
)

NOSET = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("0", False), ("", False), ("true", False), ("1", True)],
)
def test_one_gpu_per_rank_only_enabled_by_exactly_one(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv(ONE_GPU_PER_RANK_ENV, raising=False)
    else:
        monkeypatch.setenv(ONE_GPU_PER_RANK_ENV, value)
    assert one_gpu_per_rank_enabled() is enabled
    assert single_gpu_worker_env_vars() == ({} if enabled else NOSET)


def test_env_vars_are_a_fresh_dict_per_call(monkeypatch):
    monkeypatch.delenv(ONE_GPU_PER_RANK_ENV, raising=False)
    first = single_gpu_worker_env_vars()
    first["EXTRA"] = "x"
    assert single_gpu_worker_env_vars() == NOSET


@pytest.mark.parametrize(
    ("cvd", "physical_gpu_id", "expected"),
    [
        ("0,1,2,3", 2, 2),  # NOSET: whole node visible, id is the index
        ("0,1,2,3", 0, 0),
        (None, 3, 3),  # unset: CUDA sees every GPU in physical order
        ("", 3, 3),
        ("2", 2, 0),  # Ray narrowed to this worker's GPU
        (" 3 ,", 3, 0),  # stray whitespace / trailing comma
    ],
)
def test_resolve_local_device_index(monkeypatch, cvd, physical_gpu_id, expected):
    if cvd is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    assert resolve_local_device_index(physical_gpu_id) == expected


def test_resolve_local_device_index_rejects_wrong_narrowed_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match=r"CUDA_VISIBLE_DEVICES=1 .* \(3\)"):
        resolve_local_device_index(3)


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("module", "cls"),
    [
        (
            "nemo_rl.models.policy.workers.megatron_policy_worker",
            "MegatronPolicyWorkerImpl",
        ),
        (
            "nemo_rl.models.value.workers.megatron_value_worker",
            "MegatronValueWorkerImpl",
        ),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_megatron_configure_worker_honors_one_gpu_per_rank(
    monkeypatch, module, cls, enabled
):
    worker_cls = getattr(importlib.import_module(module), cls)
    if enabled:
        monkeypatch.setenv(ONE_GPU_PER_RANK_ENV, "1")
    else:
        monkeypatch.delenv(ONE_GPU_PER_RANK_ENV, raising=False)

    resources, env_vars, init_kwargs, runtime_env = worker_cls.configure_worker(
        num_gpus=1, bundle_indices=None, num_gpus_per_node=4
    )

    assert resources == {"num_gpus": 1}
    assert env_vars == ({} if enabled else NOSET)
    assert init_kwargs == {}
    assert runtime_env == {}
