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

import copy
import json
import os
import signal
import subprocess
from typing import Any, Optional

import ray

from nemo_rl.distributed.virtual_cluster import _get_free_port_local
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.dynamo.config import DynamoVllmConfig


def _build_vllm_cli_args(
    model_name: str,
    vllm_cfg: dict[str, Any],
    vllm_kwargs: dict[str, Any],
    kv_events_config_json: str,
    seed: int,
) -> list[str]:
    """Build the full CLI arg list for ``python -m dynamo.vllm``.

    Mirrors the arg-building logic in VllmGenerationWorker.__init__
    (vllm_worker.py:426-549), translated to CLI flags.
    """
    args: list[str] = ["--model", model_name]

    # Direct config → CLI flag mappings.
    # Maps config key → CLI flag name (None means same as key with underscores → dashes).
    _DIRECT_FLAGS: dict[str, Optional[str]] = {
        "tensor_parallel_size": None,
        "pipeline_parallel_size": None,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "kv_cache_dtype": None,
        "load_format": None,
        "precision": "dtype",  # nemo-rl "precision" → vLLM "--dtype"
    }

    for cfg_key, flag_name in _DIRECT_FLAGS.items():
        value = vllm_cfg.get(cfg_key)
        if value is None:
            continue
        cli_flag = f"--{(flag_name or cfg_key).replace('_', '-')}"
        args.extend([cli_flag, str(value)])

    # Boolean flags: True → --flag, False → --no-flag
    _BOOL_FLAGS: dict[str, Optional[str]] = {
        "enforce_eager": None,
    }

    for cfg_key, flag_name in _BOOL_FLAGS.items():
        value = vllm_cfg.get(cfg_key)
        if value is None:
            continue
        cli_flag = (flag_name or cfg_key).replace("_", "-")
        args.append(f"--{cli_flag}" if value else f"--no-{cli_flag}")

    # Expert parallelism maps to --enable-expert-parallel (bool flag in vLLM)
    if vllm_cfg.get("expert_parallel_size", 1) > 1:
        args.append("--enable-expert-parallel")

    # Hardcoded flags matching the regular vllm worker
    args.append("--trust-remote-code")
    args.extend(["--seed", str(seed)])
    args.extend(["--kv-events-config", kv_events_config_json])

    # hf_overrides — merge FP8 overrides with user-provided ones, serialize as JSON
    hf_overrides = vllm_kwargs.pop("hf_overrides", {})
    cfg_hf_overrides = vllm_cfg.get("hf_overrides") or {}
    if cfg_hf_overrides:
        # cfg_hf_overrides go first, vllm_kwargs hf_overrides (from FP8) override
        merged = {**cfg_hf_overrides, **hf_overrides}
        hf_overrides = merged
    if hf_overrides:
        args.extend(["--hf-overrides", json.dumps(hf_overrides)])

    # vllm_kwargs passthrough (compilation_config, speculative_config, etc.)
    for key, value in vllm_kwargs.items():
        cli_flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            args.append(cli_flag if value else f"--no-{key.replace('_', '-')}")
        elif isinstance(value, dict):
            args.extend([cli_flag, json.dumps(value)])
        else:
            args.extend([cli_flag, str(value)])

    # extra_vllm_args escape hatch (merged last — overrides everything)
    for key, value in vllm_cfg.get("extra_vllm_args", {}).items():
        cli_flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            args.append(cli_flag if value else f"--no-{key.replace('_', '-')}")
        elif isinstance(value, dict):
            args.extend([cli_flag, json.dumps(value)])
        else:
            args.extend([cli_flag, str(value)])

    return args


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
        """Configure worker resources for Ray placement."""
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        local_bundle_indices = None
        if bundle_indices is not None:
            node_idx = bundle_indices[0]
            local_bundle_indices = bundle_indices[1]
            init_kwargs["bundle_indices"] = local_bundle_indices

            # Compute a unique seed from node_idx and bundle_indices
            # (mirrors VllmGenerationWorker.configure_worker)
            if len(local_bundle_indices) == 1:
                seed = node_idx * 1024 + local_bundle_indices[0]
            else:
                bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
                seed = node_idx * 1024 + bundle_id
            init_kwargs["seed"] = seed

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
        seed: int = 0,
    ):
        self.cfg = config
        self.is_model_owner = bundle_indices is not None
        self._process: Optional[subprocess.Popen] = None

        if not self.is_model_owner:
            return

        vllm_cfg = config["vllm_cfg"]
        model_name = config["model_name"]
        tp_size = vllm_cfg["tensor_parallel_size"]
        pp_size = vllm_cfg.get("pipeline_parallel_size", 1)
        ep_size = vllm_cfg.get("expert_parallel_size", 1)
        model_parallel_size = tp_size * pp_size

        # Build CUDA_VISIBLE_DEVICES from bundle indices.
        ray_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if ray_cuda_devices:
            available = ray_cuda_devices.split(",")
            selected = [available[i] for i in bundle_indices if i < len(available)]
            cuda_visible = ",".join(selected)
        else:
            cuda_visible = ",".join(str(i) for i in bundle_indices)

        vllm_python = os.environ.get("DYNAMO_VLLM_PYTHON", "python")

        # --- Build vllm_kwargs (passthrough dict) ---
        vllm_kwargs: dict[str, Any] = copy.deepcopy(config.get("vllm_kwargs", {}))

        # --- load_format override for specific models ---
        from nemo_rl.models.huggingface.common import ModelFlag

        load_format = vllm_cfg.get("load_format", "auto")
        if ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(model_name):
            load_format = "auto"
        vllm_cfg_copy = dict(vllm_cfg)
        vllm_cfg_copy["load_format"] = load_format

        # --- FP8 support ---
        if vllm_cfg.get("precision") == "fp8":
            from nemo_rl.models.generation.vllm.quantization.fp8 import (
                compute_fp8_engine_kwargs,
            )

            fp8_kwargs = compute_fp8_engine_kwargs(
                vllm_cfg, model_name, model_parallel_size
            )
            # Merge FP8 quantization and kv_cache_dtype into vllm_kwargs
            vllm_kwargs["quantization"] = fp8_kwargs["quantization"]
            vllm_kwargs["kv_cache_dtype"] = fp8_kwargs["kv_cache_dtype"]
            # FP8 hf_overrides will be merged in _build_vllm_cli_args
            if "hf_overrides" in fp8_kwargs:
                existing = vllm_kwargs.get("hf_overrides", {})
                existing.update(fp8_kwargs["hf_overrides"])
                vllm_kwargs["hf_overrides"] = existing
            # Override precision to bfloat16 (vllm complains otherwise)
            vllm_cfg_copy["precision"] = "bfloat16"

        # --- Model-specific fixups (mirrors vllm_worker.py:506-523) ---
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if "GptOssForCausalLM" in getattr(hf_config, "architectures", []):
            if "quantization_config" in hf_config:
                assert load_format == "dummy", (
                    "Loading quantized GPT-OSS models is currently only supported with load_format='dummy'."
                )
                hf_ov = vllm_kwargs.get("hf_overrides", {})
                hf_ov["quantization_config"] = {}
                vllm_kwargs["hf_overrides"] = hf_ov

        # --- Build CLI args ---
        kv_event_port = _get_free_port_local()
        kv_events_json = json.dumps({
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{kv_event_port}",
            "enable_kv_cache_events": True,
        })

        cmd = [
            vllm_python,
            "-m",
            "dynamo.vllm",
            *_build_vllm_cli_args(
                model_name=model_name,
                vllm_cfg=vllm_cfg_copy,
                vllm_kwargs=vllm_kwargs,
                kv_events_config_json=kv_events_json,
                seed=seed,
            ),
        ]

        # --- Subprocess environment ---
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["DYN_FORWARDPASS_METRIC_PORT"] = str(_get_free_port_local())
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_SKIP_P2P_CHECK"] = "1"

        # Parallelism env vars (mirrors vllm_worker.py:432-469)
        if model_parallel_size > 1:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            fraction_of_gpus = float(os.environ.get("VLLM_RAY_PER_WORKER_GPUS", "1"))
            env["VLLM_RAY_PER_WORKER_GPUS"] = str(
                fraction_of_gpus / model_parallel_size
            )
            env["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

        if ep_size > tp_size:
            world_size = int(os.environ["VLLM_DP_SIZE"]) * model_parallel_size
            rank = int(os.environ["RANK"]) % world_size
            env["VLLM_DP_RANK"] = str(rank // model_parallel_size)
            env["VLLM_DP_RANK_LOCAL"] = str((rank % 8) // model_parallel_size)
            leader_rank = int(os.environ["RANK"]) // world_size * world_size
            addr_list = eval(os.environ["AVAILABLE_ADDR_LIST"])
            port_list = eval(os.environ["AVAILABLE_PORT_LIST"])
            env["VLLM_DP_MASTER_IP"] = addr_list[leader_rank]
            env["VLLM_DP_MASTER_PORT"] = str(port_list[leader_rank])

        if vllm_cfg.get("use_deep_gemm", False):
            env["VLLM_USE_DEEP_GEMM"] = "1"
            env["VLLM_USE_DEEP_GEMM_E8M0"] = "0"

        self._process = subprocess.Popen(cmd, env=env)
        print(
            f"  [DynamoVllmWorker] Launched dynamo.vllm (pid={self._process.pid}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"TP={tp_size})",
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
