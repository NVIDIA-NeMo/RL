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

"""End-to-end fault tolerance against real SGLang engines (Qwen3-0.6B).

Runs 2 engines × TP=1 on 2 GPUs with ``use_fault_tolerance: true``, crashes
one engine with ``SGLangGenerationWorker._simulate_crash``, and checks that
the health monitor kills it and ``recover_updatable_engines`` brings it back
without taking generation down.
"""

import gc
import time

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

from .helpers import MODEL_PATH

pytestmark = pytest.mark.sglang

PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645
# The monitor probes every CHECK_INTERVAL seconds; a crash must be noticed
# within DETECT_TIMEOUT so a wedged run fails the test instead of hanging.
CHECK_INTERVAL = 5
CHECK_TIMEOUT = 30
DETECT_TIMEOUT = 300


def _make_fault_tolerant_cfg(pad_token_id):
    return {
        "backend": "sglang",
        "model_name": MODEL_PATH,
        "model_path": MODEL_PATH,
        "tokenizer": {"name": MODEL_PATH},
        "dtype": "bfloat16",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": [EOS_TOKEN_ID],
        "stop_strings": None,
        "_pad_token_id": pad_token_id,
        "sglang_cfg": {
            "model_path": MODEL_PATH,
            "dtype": "bfloat16",
            "random_seed": 42,
            "context_length": 1024,
            "log_level": "info",
            "skip_server_warmup": True,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "disable_piecewise_cuda_graph": True,
            "disable_cuda_graph": True,
            "mem_fraction_static": 0.3,
            "sglang_server_config": {
                "num_gpus": 2,
                "num_gpus_per_engine": 1,
                # No memory saver: this test is about crash detection and engine
                # restart, not about the offload/onload state machine.
                "needs_offload": False,
                "cpu_weight_backup": False,
                "sglang_server_concurrency": 64,
                "pause_generation_mode": "retract",
            },
            "sglang_router_config": {
                "sglang_router_ip": None,
                "sglang_router_port": None,
            },
            "use_fault_tolerance": True,
            "rollout_health_check_interval": CHECK_INTERVAL,
            "rollout_health_check_timeout": CHECK_TIMEOUT,
            "rollout_health_check_first_wait": 0,
        },
        "sglang_kwargs": {},
    }


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def ft_gen(ray_cluster, tokenizer):
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=2,
        name="fault-tolerance-test",
    )
    gen = SGLangGeneration(cluster, _make_fault_tolerant_cfg(tokenizer.pad_token_id))
    yield gen
    try:
        gen.shutdown()
    except Exception:
        pass
    try:
        cluster.shutdown()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def _make_input(tokenizer, prompt):
    token_ids = tokenizer.encode(prompt)
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "input_lengths": torch.tensor([len(token_ids)], dtype=torch.long),
        }
    )


def _wait_for_dead_slot(gen, index, timeout=DETECT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.all_engines[index] is None:
            return True
        time.sleep(1)
    return False


def test_monitor_starts_with_fault_tolerance_enabled(ft_gen):
    """The monitor thread exists and is idle until generation is prepared."""
    assert ft_gen._health_monitor is not None
    assert ft_gen._health_monitor._thread is not None
    assert ft_gen._health_monitor.is_checking_enabled() is False


def test_crashed_engine_is_detected_recovered_and_generation_survives(
    ft_gen, tokenizer
):
    ft_gen.prepare_for_generation()
    assert ft_gen._health_monitor.is_checking_enabled() is True

    ray.get(ft_gen.all_engines[0]._simulate_crash.remote())
    assert _wait_for_dead_slot(ft_gen, 0), (
        "health monitor did not kill the crashed engine"
    )

    ft_gen.recover_updatable_engines()
    assert all(engine is not None for engine in ft_gen.all_engines)
    assert ft_gen.num_new_engines == 1
    assert ft_gen._health_monitor.is_checking_enabled() is False

    ft_gen.prepare_for_generation(tags=["kv_cache"])
    assert ft_gen._health_monitor.is_checking_enabled() is True

    result = ft_gen.generate(_make_input(tokenizer, "The capital of France is"))
    assert result["generation_lengths"][0].item() > 0

    ft_gen.finish_generation()
    assert ft_gen._health_monitor.is_checking_enabled() is False
