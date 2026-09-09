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

"""Real SGLang generation and recovery with 2 engines × TP=1 on 2 GPUs.

Exercises healthy offloaded engines, server/actor death while offloaded, and
SIGKILL during an observed generation request. These tests load checkpoint
weights on replacement engines; trainer weight transfer is tested separately.
"""

import gc
import threading
import time
from concurrent.futures import Future

import pytest
import ray
import requests
import torch
from ray.exceptions import RayActorError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

from .fault_injection import (
    kill_server_process_tree,
    snapshot_server_process_tree,
    wait_for_server_process_tree_exit,
)
from .helpers import MODEL_PATH

pytestmark = pytest.mark.sglang

PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645
# The monitor probes every CHECK_INTERVAL seconds; a crash must be noticed
# within DETECT_TIMEOUT so a wedged run fails the test instead of hanging.
CHECK_INTERVAL = 5
CHECK_TIMEOUT = 30
DETECT_TIMEOUT = 300


def _make_fault_tolerant_cfg(pad_token_id: int) -> dict:
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
            "context_length": 4096,
            "log_level": "info",
            "skip_server_warmup": True,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "disable_cuda_graph": True,
            "cuda_graph_backend_prefill": "disabled",
            "mem_fraction_static": 0.3,
            "sglang_server_config": {
                "num_gpus": 2,
                "num_gpus_per_engine": 1,
                "needs_offload": True,
                # Restore checkpoint weights without a trainer in this harness.
                "cpu_weight_backup": True,
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
            "rollout_max_restart_attempts": 8,
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
    gen.finish_generation()
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


def _make_input(tokenizer, prompt: str, *, batch_size: int = 1) -> BatchedDataDict:
    token_ids = tokenizer.encode(prompt)
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([token_ids] * batch_size, dtype=torch.long),
            "input_lengths": torch.tensor(
                [len(token_ids)] * batch_size, dtype=torch.long
            ),
        }
    )


def _wait_for_dead_slot(
    gen: SGLangGeneration, index: int, timeout: float = DETECT_TIMEOUT
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.all_engines[index] is None:
            return True
        time.sleep(1)
    return False


def _router_worker_urls(gen: SGLangGeneration) -> set[str]:
    response = requests.get(
        f"http://{gen.router_ip}:{gen.router_port}/workers", timeout=CHECK_TIMEOUT
    )
    response.raise_for_status()
    return {worker["url"] for worker in response.json()["workers"]}


def _engine_node_strategy(gen: SGLangGeneration) -> PlacementGroupSchedulingStrategy:
    return PlacementGroupSchedulingStrategy(
        placement_group=gen.pg,
        placement_group_bundle_index=gen.pg_reordered_bundle_indices[0],
    )


def test_monitor_starts_with_fault_tolerance_enabled(ft_gen) -> None:
    """The monitor thread exists and is idle until generation is prepared."""
    assert ft_gen._health_monitor is not None
    assert ft_gen._health_monitor._thread is not None
    assert ft_gen._health_monitor._pause_event.is_set()
    assert not ft_gen._health_monitor._stop_event.is_set()


def test_liveness_preserves_healthy_offloaded_engines(ft_gen, tokenizer) -> None:
    ft_gen.prepare_for_generation()
    ft_gen.finish_generation()
    engines = list(ft_gen.all_engines)
    ft_gen.clear_updatable_num_new_engines()

    assert ft_gen._health_monitor._pause_event.is_set()
    assert ray.get([engine.is_alive.remote() for engine in engines]) == [True, True]
    ft_gen.recover_updatable_engines()

    assert ft_gen.all_engines == engines
    assert ft_gen.num_new_engines == 0
    ft_gen.prepare_for_generation()
    result = ft_gen.generate(_make_input(tokenizer, "The capital of France is"))
    assert result["generation_lengths"][0].item() > 0
    ft_gen.finish_generation()


@pytest.mark.parametrize("kill_actor", [False, True], ids=["server", "actor"])
def test_offloaded_death_is_detected_at_recovery(
    ft_gen, tokenizer, kill_actor: bool
) -> None:
    ft_gen.prepare_for_generation()
    ft_gen.finish_generation()
    victim = ft_gen.all_engines[0]
    assert ft_gen._health_monitor._pause_event.is_set()

    strategy = _engine_node_strategy(ft_gen)
    tree = ray.get(
        snapshot_server_process_tree.options(scheduling_strategy=strategy).remote(
            ft_gen._engine_urls[0]
        ),
        timeout=CHECK_TIMEOUT,
    )
    assert len(tree.processes) > 1, "No GPU descendants captured before server death"
    if kill_actor:
        ray.kill(victim)
        with pytest.raises(RayActorError):
            ray.get(victim.is_alive.remote(), timeout=CHECK_TIMEOUT)
    else:
        ray.get(victim.shutdown.remote(), timeout=CHECK_TIMEOUT)

    # shutdown() sends signals through kill_process_tree without waiting for
    # exit. Establish actual process death before asking the liveness probe
    # to detect it; the same check proves actor death leaves no GPU orphans.
    ray.get(
        wait_for_server_process_tree_exit.options(scheduling_strategy=strategy).remote(
            tree, timeout=60
        ),
        timeout=90,
    )
    print(f"Server teardown reaped {tree.url}: processes={tree.processes}")
    if not kill_actor:
        # The actor remains callable after graceful server shutdown, proving
        # that checking only Ray actor death misses a dead child process.
        assert ray.get(victim.is_alive.remote(), timeout=CHECK_TIMEOUT) is False
    assert ft_gen.all_engines[0] == victim

    ft_gen.recover_updatable_engines()
    assert all(engine is not None for engine in ft_gen.all_engines)
    assert ft_gen.all_engines[0] != victim
    assert ft_gen.num_new_engines == 1
    assert ft_gen._health_monitor._pause_event.is_set()

    ft_gen.prepare_for_generation()
    result = ft_gen.generate(_make_input(tokenizer, "The capital of France is"))
    assert result["generation_lengths"][0].item() > 0
    ft_gen.finish_generation()


def test_inflight_server_sigkill_survives_and_recovers(
    ft_gen, tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    ft_gen.prepare_for_generation()
    monitor = ft_gen._health_monitor
    # Exclude health_generate requests from the load observation. Detection is
    # resumed immediately after injection while the user batch remains active.
    monitor.pause()
    server_urls = list(ft_gen._engine_urls)
    assert all(url is not None for url in server_urls)
    assert set(server_urls) <= _router_worker_urls(ft_gen)
    original_engines = list(ft_gen.all_engines)

    max_new_tokens = 2048
    batch_size = 4
    build_sampling_params = ft_gen._build_sampling_params

    def long_sampling_params(**kwargs) -> dict:
        params = build_sampling_params(**kwargs)
        params.update(ignore_eos=True, stop=[], stop_token_ids=[])
        return params

    future = Future()

    def generate_batch() -> None:
        try:
            result = ft_gen.generate(
                _make_input(
                    tokenizer, "Count the positive integers:", batch_size=batch_size
                )
            )
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    with monkeypatch.context() as patch:
        patch.setitem(ft_gen.sglang_cfg, "max_new_tokens", max_new_tokens)
        patch.setattr(ft_gen, "_build_sampling_params", long_sampling_params)
        # The task uses the same placement-group node as both TP=1 engines.
        injector = kill_server_process_tree.options(
            scheduling_strategy=_engine_node_strategy(ft_gen)
        )
        injection = injector.remote(server_urls, wait_for_inflight=True, timeout=60)
        generation_thread = threading.Thread(target=generate_batch, daemon=True)
        generation_thread.start()
        try:
            killed = ray.get(injection, timeout=100)
        finally:
            monitor.resume()
        assert killed.running_requests > 0
        assert killed.child_pids, "No GPU subprocesses were captured for cleanup"
        print(
            f"SIGKILL {killed.url}: server_pid={killed.pid}, "
            f"children={killed.child_pids}, running_requests={killed.running_requests}"
        )
        result = future.result(timeout=DETECT_TIMEOUT)
        assert result["generation_lengths"].tolist() == [max_new_tokens] * batch_size

    victim_index = server_urls.index(killed.url)
    assert _wait_for_dead_slot(ft_gen, victim_index), (
        "health monitor did not detect the server killed during generation"
    )
    assert killed.url not in _router_worker_urls(ft_gen), (
        "dead worker remained registered after monitor cleanup"
    )

    ft_gen.finish_generation()
    ft_gen.recover_updatable_engines()
    assert all(engine is not None for engine in ft_gen.all_engines)
    assert ft_gen.all_engines[victim_index] != original_engines[victim_index]
    assert ft_gen.num_new_engines == 1
    assert monitor._pause_event.is_set()
    assert set(ft_gen._engine_urls) <= _router_worker_urls(ft_gen)

    ft_gen.prepare_for_generation(tags=["weights"])
    ft_gen.prepare_for_generation(tags=["kv_cache"])
    assert not monitor._pause_event.is_set()
    result = ft_gen.generate(_make_input(tokenizer, "The capital of France is"))
    assert result["generation_lengths"][0].item() > 0
    ft_gen.finish_generation()
    assert monitor._pause_event.is_set()


def test_serving_server_shutdown_is_detected_and_recovered(ft_gen, tokenizer) -> None:
    ft_gen.prepare_for_generation()
    assert not ft_gen._health_monitor._pause_event.is_set()

    ray.get(ft_gen.all_engines[0].shutdown.remote(), timeout=CHECK_TIMEOUT)
    assert _wait_for_dead_slot(ft_gen, 0), (
        "health monitor did not kill the crashed engine"
    )

    ft_gen.finish_generation()
    ft_gen.recover_updatable_engines()
    assert all(engine is not None for engine in ft_gen.all_engines)
    assert ft_gen.num_new_engines == 1
    assert ft_gen._health_monitor._pause_event.is_set()

    ft_gen.prepare_for_generation(tags=["weights"])
    ft_gen.prepare_for_generation(tags=["kv_cache"])
    assert not ft_gen._health_monitor._pause_event.is_set()

    result = ft_gen.generate(_make_input(tokenizer, "The capital of France is"))
    assert result["generation_lengths"][0].item() > 0

    ft_gen.finish_generation()
    assert ft_gen._health_monitor._pause_event.is_set()
