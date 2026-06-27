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

"""Shared helpers for sglang tests.

Importing this module installs:
  • env-var defaults that must be set before sglang imports
  • MagicMock stubs for non-sglang heavy dependencies that may not be present

Order matters: env vars and stubs are installed at the top of this file,
before any ``from nemo_rl...`` import, so test files only need to
``import helpers`` (directly or transitively) to pick them up.
"""

# ---------------------------------------------------------------------------
# Env vars — must be set BEFORE any sglang / nemo_rl import.
# ---------------------------------------------------------------------------
import os

# Disable sglang's per-GPU memory imbalance check — when running tests on a
# shared host other processes may already hold memory on some of our GPUs.
os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "false")

import ray

from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker
from nemo_rl.models.generation.sglang.utils.ray_utils import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
    find_available_port,
    get_host_info,
)
from nemo_rl.utils.venvs import make_actor_runtime_env

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
DEFAULT_GPUS_PER_NODE = 2


def make_sglang_cfg(
    model_path=MODEL_PATH,
    tp_size=1,
    num_gpus=2,
    router_ip=None,
    router_port=None,
):
    return {
        "sglang_cfg": {
            "model_path": model_path,
            "random_seed": 42,
            "tp_size": tp_size,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "skip_server_warmup": True,
            "dtype": "bfloat16",
            "context_length": 1024,
            "log_level": "warning",
            "disable_piecewise_cuda_graph": True,
            "disable_cuda_graph": True,
            "mem_fraction_static": 0.3,
            "sglang_server_config": {
                "num_gpus": num_gpus,
                "num_gpus_per_engine": tp_size,
                "needs_offload": True,
                "cpu_weight_backup": False,
                "sglang_server_concurrency": 64,
                "pause_generation_mode": "retract",
            },
            "sglang_router_config": {
                "sglang_router_ip": router_ip,
                "sglang_router_port": router_port,
            },
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
    gpus_per_node = DEFAULT_GPUS_PER_NODE
    sglang_cfg = make_sglang_cfg(
        tp_size=tp_size,
        router_ip=router_info["ip"],
        router_port=router_info["port"],
    )

    # Materialize the sglang venv so the actor doesn't inherit the driver's
    # base venv (e.g. /opt/nemo_rl_venv in CI containers).
    runtime_env = make_actor_runtime_env(
        "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
    )
    runtime_env["env_vars"].update(make_actor_env_vars())

    worker = SGLangGenerationWorker.options(
        num_cpus=0.2,
        num_gpus=0.2,
        runtime_env=runtime_env,
    ).remote(
        gpus_per_node,
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
            host=host_ip,
            router_ip=router_info["ip"],
            router_port=router_info["port"],
        )
    )
    return worker


def make_generation_sampling_params(
    max_new_tokens=16,
    temperature=0.0,
    top_p=1.0,
    stop=None,
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
