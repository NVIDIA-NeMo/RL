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

"""Ray actor that launches a dynamo vllm worker as a subprocess.

Each actor reserves GPU resources via Ray placement groups and launches
``python -m dynamo.vllm`` with the appropriate CUDA_VISIBLE_DEVICES.
The dynamo worker handles tensor parallelism internally, so one
subprocess per data-parallel shard is sufficient.
"""

import os
import signal
import subprocess
from typing import Any, Optional

import ray

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.dynamo.config import DynamoVllmConfig


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("dynamo_vllm_worker")}
)  # pragma: no cover
class DynamoVllmWorker:
    """Ray actor wrapping a ``python -m dynamo.vllm`` subprocess.

    For TP groups the leader actor (bundle_indices provided) launches the
    subprocess while non-leader actors are lightweight resource reservations.
    """

    def __repr__(self) -> str:
        return "DynamoVllmWorker"

    @staticmethod
    def configure_worker(
        num_gpus: int | float, bundle_indices: Optional[tuple[int, list[int]]] = None
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Configure worker resources for Ray placement.

        Follows the same pattern as BaseVllmGenerationWorker.configure_worker
        in vllm_worker.py.
        """
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        local_bundle_indices = None
        if bundle_indices is not None:
            local_bundle_indices = bundle_indices[1]
            init_kwargs["bundle_indices"] = local_bundle_indices

        # For parallel groups (TP > 1), let Ray reserve resources without
        # setting CUDA_VISIBLE_DEVICES — the subprocess manages GPU assignment.
        is_part_of_parallel_workers = (
            local_bundle_indices is not None and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None

        if is_part_of_parallel_workers:
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: DynamoVllmConfig,
        bundle_indices: Optional[list[int]] = None,
    ):
        self.cfg = config
        self.is_model_owner = bundle_indices is not None
        self._process: Optional[subprocess.Popen] = None

        if not self.is_model_owner:
            return

        vllm_cfg = config["vllm_cfg"]

        # Build CUDA_VISIBLE_DEVICES from bundle indices.
        # Ray sets CUDA_VISIBLE_DEVICES for the actor process; we read it
        # and remap based on the bundle indices.
        ray_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if ray_cuda_devices:
            available = ray_cuda_devices.split(",")
            selected = [available[i] for i in bundle_indices if i < len(available)]
            cuda_visible = ",".join(selected)
        else:
            cuda_visible = ",".join(str(i) for i in bundle_indices)

        # Use the vllm venv python resolved by DynamoVllmGeneration and
        # passed via the DYNAMO_VLLM_PYTHON env var.
        vllm_python = os.environ.get("DYNAMO_VLLM_PYTHON", "python")

        cmd = [
            vllm_python,
            "-m",
            "dynamo.vllm",
            "--model",
            config["model_name"],
            "--tensor-parallel-size",
            str(vllm_cfg["tensor_parallel_size"]),
            "--gpu-memory-utilization",
            str(vllm_cfg["gpu_memory_utilization"]),
            "--max-model-len",
            str(vllm_cfg["max_model_len"]),
        ]

        # Pass through extra vllm args
        for key, value in vllm_cfg.get("extra_vllm_args", {}).items():
            cmd.extend([f"--{key}", str(value)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        # etcd_endpoint and namespace are set by DynamoVllmGeneration via
        # the worker group env_vars mechanism.

        self._process = subprocess.Popen(
            cmd,
            env=env,
        )
        print(
            f"  [DynamoVllmWorker] Launched dynamo.vllm (pid={self._process.pid}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"TP={vllm_cfg['tensor_parallel_size']})",
            flush=True,
        )

    def is_alive(self) -> bool:
        if self._process is None:
            return self.is_model_owner is False  # non-leader is always "alive"
        return self._process.poll() is None

    def shutdown(self) -> bool:
        if self._process is None:
            return True
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        print(
            f"  [DynamoVllmWorker] Stopped dynamo.vllm (pid={self._process.pid})",
            flush=True,
        )
        self._process = None
        return True
