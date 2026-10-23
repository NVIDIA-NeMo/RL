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

import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_vllm_argv,
    build_managed_worker_env,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.dynamo.engine_host import (
    RAY_ADDRESS_ENV,
    SAMPLING_PROFILE_ENV,
    sampling_profile_env,
)
from nemo_rl.models.generation.vllm.draft_utils import (
    DRAFT_DISABLE_MODULE_SHARING_ENV,
    draft_module_sharing_disable_required,
)


def _fp8_engine_args(
    model_name: str,
    vllm_cfg: dict[str, Any],
    vllm_kwargs: dict[str, Any],
    model_parallel_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    """Translate ``precision: fp8`` into ``dynamo.vllm`` arguments and environment.

    Uses the vLLM backend's FP8 builder, so both backends quantize identically.

    Returns:
        ``vllm_cfg`` with vLLM's working dtype (the quantization config makes the
        weights FP8), ``vllm_kwargs`` with the FP8 quantization merged in, default
        engine environment, and the environment that hands the FP8 config to NeMo
        RL's worker extension inside each engine.
    """
    from nemo_rl.models.generation.vllm.quantization.fp8 import (
        build_fp8_config,
        fp8_engine_env,
    )
    from nemo_rl.models.generation.vllm.quantization.fp8_kwargs import (
        FP8_CONFIG_ENV_VAR,
        merge_fp8_kwargs,
    )

    fp8_config, fp8_kwargs = build_fp8_config(vllm_cfg, model_name, model_parallel_size)
    # Already passed as --kv-cache-dtype from vllm_cfg.
    fp8_kwargs.pop("kv_cache_dtype")
    vllm_kwargs = dict(vllm_kwargs)
    merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)
    return (
        {**vllm_cfg, "precision": "bfloat16"},
        vllm_kwargs,
        fp8_engine_env(vllm_cfg, fp8_config),
        {FP8_CONFIG_ENV_VAR: json.dumps(dataclasses.asdict(fp8_config))},
    )


@ray.remote(num_cpus=0)
class DynamoGpuReservation:  # pragma: no cover
    """Hold one placement-group GPU while a sibling actor owns the engine."""

    def __init__(self) -> None:
        self._process_pid: int | None = None

    def metadata(self) -> dict[str, Any]:
        gpu_ids = [int(float(gpu_id)) for gpu_id in ray.get_gpu_ids()]
        if len(gpu_ids) != 1:
            raise RuntimeError(
                f"Expected one GPU for Dynamo reservation, got {gpu_ids}."
            )
        return {"node_ip": _get_node_ip_local(), "gpu_id": gpu_ids[0]}

    def select_free_port(
        self,
        *,
        port_range_low: int,
        port_range_high: int,
        excluded_ports: list[int],
    ) -> int:
        """Select an unused node-local port from a half-open range."""
        return _get_free_port_local(
            port_range_low,
            port_range_high,
            max_retries=None,
            excluded_ports=set(excluded_ports),
        )

    def register_process_group(self, pid: int) -> bool:
        """Record the colocated worker process group for failure cleanup."""
        if self._process_pid not in (None, pid):
            raise RuntimeError(
                "Dynamo GPU reservation already owns process group "
                f"{self._process_pid}; cannot register {pid}."
            )
        self._process_pid = pid
        return True

    def cleanup_process_group(self) -> bool:
        """Best-effort cleanup if the subprocess-owning actor died first."""
        pid = self._process_pid
        if pid is None:
            return True
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._process_pid = None
            return True
        time.sleep(2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._process_pid = None
        return True


@ray.remote(num_cpus=0)
class DynamoVllmWorker:  # pragma: no cover
    """Own one ``dynamo.vllm`` subprocess for a model-parallel GPU group."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        namespace: str,
        group_name: str,
        cuda_devices: list[int],
        system_port: int,
        vllm_port: int,
        manager_env: dict[str, str],
        startup_timeout_s: float,
        seed: int,
        cleanup_reservation: ray.actor.ActorHandle,
    ) -> None:
        self._group_name = group_name
        self._node_ip = _get_node_ip_local()
        self._system_port = system_port
        self._vllm_port = vllm_port
        self._process: subprocess.Popen | None = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("0.0.0.0", system_port))
        except OSError as exc:
            raise RuntimeError(
                f"DYN_SYSTEM_PORT {system_port} is unavailable for {group_name}."
            ) from exc

        validated_config = DynamoConfig.model_validate(config)
        dynamo_cfg = validated_config.dynamo_cfg
        # This actor runs in the dynamo extra's venv (actor_environments.py), which
        # holds nemo_rl, vLLM and ai-dynamo, so dynamo.vllm can load NeMo RL's
        # vLLM worker extension.
        dynamo_python = sys.executable
        dynamo_venv = sys.prefix
        # The same on-disk vLLM source fixes the vLLM backend applies (tool-parser
        # namespace, Eagle3 draft lm_head, model loaders). Imports only vLLM's logger, never CUDA: this
        # actor holds no GPU.
        from nemo_rl.models.generation.vllm.config import (
            vllm_nemotron_h_fp32_lm_head_enabled,
        )
        from nemo_rl.models.generation.vllm.patches import (
            VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR,
            ensure_vllm_source_compat,
            require_nemotron_h_fp32_lm_head_patch,
        )

        ensure_vllm_source_compat()
        vllm_cfg = validated_config.vllm_cfg.model_dump()
        # As on the vLLM backend: Nemotron-H logits from an fp32 LM head, to match a
        # trainer with megatron_cfg.fp32_lm_head. The source patch is inert unless
        # the engine environment turns it on, so set the flag either way rather than
        # inherit it from this actor.
        fp32_lm_head = vllm_nemotron_h_fp32_lm_head_enabled(vllm_cfg)
        if fp32_lm_head:
            require_nemotron_h_fp32_lm_head_patch()
        fp32_lm_head_env = {
            VLLM_NEMOTRON_H_FP32_LM_HEAD_ENV_VAR: "1" if fp32_lm_head else "0"
        }
        vllm_kwargs = dict(validated_config.vllm_kwargs)
        engine_env_defaults: dict[str, str] = {}
        fp8_env: dict[str, str] = {}
        if vllm_cfg["precision"] == "fp8":
            vllm_cfg, vllm_kwargs, engine_env_defaults, fp8_env = _fp8_engine_args(
                config["model_name"],
                vllm_cfg,
                vllm_kwargs,
                validated_config.engine_world_size,
            )
        # As on the vLLM backend: a full-stream co-trained drafter refits its own
        # embed_tokens/lm_head, so they must not alias the target model's.
        draft_env = (
            {DRAFT_DISABLE_MODULE_SHARING_ENV: "1"}
            if draft_module_sharing_disable_required(config)
            else {}
        )
        configured_env = dict(vllm_cfg.get("env_vars") or {})
        worker_env = build_managed_worker_env(
            base_env={**os.environ, **engine_env_defaults},
            configured_env=configured_env,
            manager_env={
                **manager_env,
                **fp8_env,
                **draft_env,
                **fp32_lm_head_env,
                # NeMo RL's engine host (engine_main): the on-policy sampling it
                # enforces on rollouts, and the cluster its token capture attaches
                # to. The engine is a plain subprocess, not a Ray worker.
                SAMPLING_PROFILE_ENV: sampling_profile_env(config),
                RAY_ADDRESS_ENV: ray.get_runtime_context().gcs_address,
                "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in cuda_devices),
                "DYN_SYSTEM_PORT": str(system_port),
                "PYTHONHASHSEED": "0",
                "VLLM_PORT": str(vllm_port),
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
        self._validate_argv(
            dynamo_python,
            argv,
            worker_env,
            timeout_s=startup_timeout_s,
        )

        command = [
            dynamo_python,
            "-m",
            "nemo_rl.models.generation.dynamo.engine_main",
            *argv,
        ]
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
            ray.get(
                cleanup_reservation.register_process_group.remote(self._process.pid)
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
        dynamo_python: str,
        argv: list[str],
        env: dict[str, str],
        *,
        timeout_s: float,
    ) -> None:
        validator = Path(__file__).with_name("validate_dynamo_vllm_args.py")
        command = [dynamo_python, str(validator), json.dumps(argv)]
        try:
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Resolved dynamo.vllm argument validation exceeded "
                f"startup_timeout_s={timeout_s}."
            ) from error
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
            "vllm_port": self._vllm_port,
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
