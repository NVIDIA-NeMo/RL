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

"""Ray actors used by the fixed, managed Dynamo vLLM fleet."""

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import ray

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_vllm_argv,
    build_managed_worker_env,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import DynamoCfg


@ray.remote(num_cpus=0)
class DynamoGpuReservation:
    """Hold one placement-group GPU while a sibling actor owns the engine."""

    def metadata(self) -> dict[str, Any]:
        gpu_ids = [int(float(gpu_id)) for gpu_id in ray.get_gpu_ids()]
        if len(gpu_ids) != 1:
            raise RuntimeError(
                f"Expected one GPU for Dynamo reservation, got {gpu_ids}."
            )
        return {"node_ip": _get_node_ip_local(), "gpu_id": gpu_ids[0]}

    def cleanup_process_group(self, pid: int) -> bool:
        """Best-effort cleanup if the subprocess-owning actor died first."""
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        time.sleep(2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True


@ray.remote(num_cpus=0)
class DynamoVllmWorker:
    """Own one ``dynamo.vllm`` subprocess for a model-parallel GPU group."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        namespace: str,
        group_name: str,
        cuda_devices: list[int],
        system_port: int,
        manager_env: dict[str, str],
        startup_timeout_s: float,
        seed: int,
    ) -> None:
        self._group_name = group_name
        self._node_ip = _get_node_ip_local()
        self._system_port = system_port
        self._process: subprocess.Popen | None = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("0.0.0.0", system_port))
        except OSError as exc:
            raise RuntimeError(
                f"DYN_SYSTEM_PORT {system_port} is unavailable for {group_name}."
            ) from exc

        dynamo_cfg = DynamoCfg.model_validate(config["dynamo_cfg"])
        dynamo_venv = str(Path(dynamo_cfg.dynamo_python).resolve().parent.parent)
        vllm_cfg = dict(config.get("vllm_cfg") or {})
        vllm_kwargs = dict(config.get("vllm_kwargs") or {})
        configured_env = dict(vllm_cfg.get("env_vars") or {})
        worker_env = build_managed_worker_env(
            base_env=os.environ,
            configured_env=configured_env,
            manager_env={
                **manager_env,
                "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in cuda_devices),
                "DYN_SYSTEM_PORT": str(system_port),
                "PYTHONHASHSEED": "0",
                "VLLM_SKIP_P2P_CHECK": "1",
                "VIRTUAL_ENV": dynamo_venv,
                "UV_PROJECT_ENVIRONMENT": dynamo_venv,
            },
        )
        argv = build_dynamo_vllm_argv(
            model_name=config["model_name"],
            namespace=namespace,
            seed=seed,
            vllm_cfg=vllm_cfg,
            vllm_kwargs=vllm_kwargs,
            dynamo_cfg=dynamo_cfg,
        )
        self._validate_argv(dynamo_cfg.dynamo_python, argv, worker_env)

        command = [dynamo_cfg.dynamo_python, "-m", "dynamo.vllm", *argv]
        relevant_env = {
            key: value
            for key, value in worker_env.items()
            if key.startswith(("CUDA_", "DYN_", "ETCD_", "NATS_", "NCCL_", "VLLM_"))
        }
        print(
            f"  [Dynamo:{group_name}] launching argv={redact_argv(command)!r} "
            f"env={redact_environment(relevant_env)!r} "
            f"system_url={self.system_url}",
            flush=True,
        )
        try:
            self._process = subprocess.Popen(
                command, env=worker_env, start_new_session=True
            )
            self._wait_for_system_port(startup_timeout_s)
        except Exception:
            self._stop_process()
            raise

    @property
    def system_url(self) -> str:
        host = f"[{self._node_ip}]" if ":" in self._node_ip else self._node_ip
        return f"http://{host}:{self._system_port}"

    @staticmethod
    def _validate_argv(
        dynamo_python: str, argv: list[str], env: dict[str, str]
    ) -> None:
        validator = Path(__file__).with_name("validate_dynamo_vllm_args.py")
        result = subprocess.run(
            [dynamo_python, str(validator), json.dumps(argv)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Resolved dynamo.vllm arguments failed validation: "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}"
            )

    def _wait_for_system_port(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"dynamo.vllm for {self._group_name} exited with "
                    f"code {process.returncode} before its system endpoint was ready."
                )
            try:
                with socket.create_connection(
                    ("127.0.0.1", self._system_port), timeout=1
                ):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(
            f"dynamo.vllm for {self._group_name} did not open DYN_SYSTEM_PORT "
            f"{self._system_port} within {timeout_s}s."
        )

    def metadata(self) -> dict[str, Any]:
        process = self._process
        if process is None:
            raise RuntimeError(f"dynamo.vllm for {self._group_name} is not running.")
        return {
            "instance_id": self._group_name,
            "system_url": self.system_url,
            "process_pid": process.pid,
        }

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self._process = None

    def shutdown(self) -> bool:
        process = self._process
        if process is None:
            return True
        self._stop_process()
        print(
            f"  [Dynamo:{self._group_name}] stopped pid={process.pid}",
            flush=True,
        )
        return True
