# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared helpers for redesign tests.

Kept in a regular module (not conftest.py) so test files can import it
directly.  conftest.py also imports from here for fixture definitions.
"""

import os

import ray

from nemo_rl.models.generation.redesign.misc import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from nemo_rl.models.generation.redesign.ray_utils import (
    find_available_port,
    get_host_info,
)
from nemo_rl.models.generation.redesign.sglang_worker import SGLangGenerationWorker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen3-0.6B"

# Qwen3-0.6B model dimensions (verified against HuggingFace config)
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
QKV_OUTPUT_DIM = NUM_ATTENTION_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM  # 4096


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def make_cluster_cfg(gpus_per_node=4):
    return {"gpus_per_node": gpus_per_node}


def make_sglang_cfg(
    model_path=MODEL_PATH,
    tp_size=1,
    num_gpus=4,
    router_ip=None,
    router_port=None,
):
    return {
        "sglang_cfg": {
            "model_path": model_path,
            "random_seed": 42,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "skip_server_warmup": True,
            "dtype": "bfloat16",
            "context_length": 1024,
            "log_level": "warning",
            "disable_piecewise_cuda_graph": True,
        },
        "sglang_server": {
            "num_gpus": num_gpus,
            "num_gpus_per_engine": tp_size,
            "needs_offload": True,
            "sglang_server_concurrency": 64,
        },
        "sglang_router": {
            "sglang_router_ip": router_ip,
            "sglang_router_port": router_port,
        },
    }


def make_actor_env_vars():
    """Build env-vars dict for SGLang worker actors."""
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        env_vars["CUDA_VISIBLE_DEVICES"] = cvd
    return env_vars


def create_worker(router_info, base_gpu_id=0, tp_size=1, rank=0):
    """Create and initialise a real SGLangGenerationWorker Ray actor.

    Returns the actor handle after ``init`` completes.
    """
    cluster_cfg = make_cluster_cfg()
    sglang_cfg = make_sglang_cfg(
        tp_size=tp_size,
        router_ip=router_info["ip"],
        router_port=router_info["port"],
    )

    worker = SGLangGenerationWorker.options(
        num_cpus=0.2,
        num_gpus=0.2,
        runtime_env={"env_vars": make_actor_env_vars()},
    ).remote(
        cluster_cfg,
        sglang_cfg,
        rank=rank,
        base_gpu_id=base_gpu_id,
        num_gpus_per_engine=tp_size,
    )

    host_ip = get_host_info()[1]
    port = find_available_port(30000 + rank * 1000)
    nccl_port = find_available_port(40000 + rank * 1000)
    dist_init_port = find_available_port(50000 + rank * 1000)

    ray.get(
        worker.init.remote(
            dist_init_addr=f"{host_ip}:{dist_init_port}",
            port=port,
            nccl_port=nccl_port,
            router_ip=router_info["ip"],
            router_port=router_info["port"],
        )
    )
    return worker


def make_generation_sampling_params(
    max_new_tokens=16, temperature=0.0, top_p=1.0, stop=None,
):
    """Build sampling_params dict for generate_one_sample / router /generate."""
    params = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if stop is not None:
        params["stop"] = stop
    return params


# ---------------------------------------------------------------------------
# HTTP helpers for tests that want an explicit status-code check
# ---------------------------------------------------------------------------
def post_and_assert_200(base_url, endpoint, payload=None):
    """POST ``payload`` to ``{base_url}/{endpoint}`` and assert HTTP 200.

    Tests that exercise ``release_memory_occupation`` / ``resume_memory_occupation``
    use this instead of ``_make_request`` so the 200 check is visible in the
    test body (``_make_request`` consumes the status code inside
    ``raise_for_status()`` and returns only the parsed JSON).
    """
    import requests

    resp = requests.post(f"{base_url}/{endpoint}", json=payload or {})
    assert resp.status_code == 200, (
        f"POST {endpoint} expected 200, got {resp.status_code}: {resp.text}"
    )
    return resp.json()
