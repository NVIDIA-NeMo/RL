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
import os
import warnings
from typing import TYPE_CHECKING, Optional, cast

from transformers import PreTrainedTokenizerBase

from nemo_rl.models.generation.interfaces import GenerationConfig, GenerationInterface
from nemo_rl.models.generation.trtllm import TrtllmConfig
from nemo_rl.models.generation.vllm import VllmConfig
from nemo_rl.models.generation.vllm.config import VLLM_SPARSE_REFIT_TRANSPORTS

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import FleetHealthConfig

TokenizerType = PreTrainedTokenizerBase


def maybe_configure_engine_reaping_env(
    fleet_health: "Optional[FleetHealthConfig]",
) -> None:
    """Let the raylet reap a dead generation worker's EngineCore.

    Call this on the driver **before** ``init_ray()``, like
    :func:`~nemo_rl.data_plane.factory.maybe_configure_data_plane_env`. The raylet reads
    this setting when ``ray.init`` spawns it, and the raylet is the whole point: it
    outlives the worker, so it can kill what the worker no longer can.

    A vLLM generation worker spawns its EngineCore as a plain multiprocessing child --
    ``context.Process(target=EngineCoreProc.run_engine_core)``, no ``setsid``, not a
    daemon -- and that child owns the CUDA context and the KV cache. Every cleanup path
    that exists runs INSIDE the dying process, so SIGKILL defeats all of them: vLLM's
    ``weakref.finalize`` never fires, a non-daemon multiprocessing child is designed to
    outlive its parent, and Ray's default ``RAY_kill_child_processes_on_worker_exit`` is
    the worker's own exit handler. The orphan then holds its GPU for the life of the job.
    Because the EngineCore never calls ``setsid`` it stays in the worker's process group,
    which is exactly what per-worker process-group cleanup reaches.

    Job 6720618 measured the leak on 4xGB200: after one shard was SIGKILLed, cuda:0
    reported 69.36/184.31 GiB free -- 114.95 GiB still held, one engine at
    ``gpu_memory_utilization=0.6`` plus overhead -- and that number did not move across
    five restart attempts over 370s. The replacement could not fit, so re-admission
    failed while the survivors carried on.

    Scoped to fleet-health runs rather than set on ``import nemo_rl``. Two reasons, and
    the second is why this function exists at all:

    - Only a run that tolerates a dead shard can leave an orphan behind. This is NOT
      specific to restart -- any shard loss leaks that GPU for the rest of the run -- so
      the gate is ``fleet_health.enabled``, not ``restart_dead_shards``.
    - It is a raylet-wide behaviour change, and importing ``nemo_rl`` is not consent to
      it. A plain SFT, DPO or distillation run has no EngineCore to reap, so the flag is
      pure risk there.

    ``setdefault``, not assignment: an operator debugging a wedged engine may want the
    corpse, and an explicit ``RAY_process_group_cleanup_enabled=0`` in the environment
    stays honoured.

    Args:
        fleet_health: The generation fleet-health config, or ``None`` on a path that has
            no fleet health at all. No-op unless it is enabled.
    """
    enabled = getattr(fleet_health, "enabled", False) if fleet_health else False
    if not enabled:
        return
    os.environ.setdefault("RAY_process_group_cleanup_enabled", "1")


def resolve_generation_class(
    generation_config: GenerationConfig,
) -> type[GenerationInterface]:
    """Map `generation_config` to its GenerationInterface class."""
    backend = generation_config["backend"]
    if backend == "vllm":
        from nemo_rl.models.generation.vllm import VllmGeneration

        return VllmGeneration
    if backend == "sglang":
        from nemo_rl.models.generation.sglang.sglang_generation import (
            SGLangGeneration,
        )

        return SGLangGeneration
    if backend == "megatron":
        from nemo_rl.models.generation.megatron.megatron_generation import (
            MegatronGeneration,
        )

        return MegatronGeneration
    if backend == "trtllm":
        from nemo_rl.models.generation.trtllm import TrtllmGeneration

        return TrtllmGeneration
    if backend == "dynamo":
        from nemo_rl.models.generation.dynamo import DynamoGeneration

        return DynamoGeneration
    raise ValueError(f"Unknown generation backend: {backend!r}")


def configure_generation_config(
    config: GenerationConfig,
    tokenizer: TokenizerType,
    is_eval: bool = False,
    has_refit_draft_weights: bool = False,
    trains_mtp: bool = False,
) -> GenerationConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    if "_pad_token_id" in config:
        warnings.warn(
            "'_pad_token_id' found in generation config and will be overridden with tokenizer.pad_token_id. "
            "Note: '_pad_token_id' is intended for internal use and has no effect when set in user-provided configs.",
            UserWarning,
        )
    config["_pad_token_id"] = tokenizer.pad_token_id
    if config["stop_token_ids"] is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # vLLM setting shared by the standard and managed Dynamo backends.
    if config["backend"] in ("vllm", "dynamo"):
        vllm_backed_config = cast(VllmConfig, config)
        vllm_backed_config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"

    if config["backend"] == "vllm":
        config = cast(VllmConfig, config)
        if config.get("real_quant"):
            export_cpu_offload = config.get("real_quant_export_cpu_offload")
            if not isinstance(export_cpu_offload, bool):
                raise ValueError(
                    "generation.real_quant_export_cpu_offload must be a boolean"
                )
            colocated = config.get("colocated")
            if not export_cpu_offload and (
                colocated is None
                or not colocated["enabled"]
                or config.get("refit_transport") is not None
            ):
                raise ValueError(
                    "generation.real_quant_export_cpu_offload=false requires "
                    "colocated CUDA-IPC refit with no explicit refit_transport"
                )

        # set load_format
        if config.get("refit_transport") in VLLM_SPARSE_REFIT_TRANSPORTS:
            config["vllm_cfg"]["load_format"] = "auto"
        speculative_config = config.get("vllm_kwargs", {}).get("speculative_config")
        if speculative_config and not is_eval and not has_refit_draft_weights:
            # Speculative decoding needs real draft weights at startup, since the
            # draft is not covered by the initial refit.
            if speculative_config.get("method") not in ("deepseek_mtp", "mtp"):
                # Non-MTP methods (e.g. Eagle) must read the drafter's real
                # weights from the checkpoint, so load everything.
                warnings.warn(
                    "Speculative decoding is enabled without draft refit sync. "
                    "Setting vllm_cfg['load_format'] to 'auto' so the drafter does "
                    "not start from dummy weights."
                )
                config["vllm_cfg"]["load_format"] = "auto"

        # MTP draft weights arrive via refit if the trainer trains the MTP layer.
        # If the trainer does not train the MTP layer, the weights need to be
        # loaded from the checkpoint.
        config["_mtp_weights_from_refit"] = trains_mtp

        # Respect the skip_tokenizer_init setting from the config. VLMs for example, require this to be False.
        if "skip_tokenizer_init" not in config["vllm_cfg"]:
            # set skip_tokenizer_init
            if (
                is_eval
                or config["stop_strings"] is not None
                or config["vllm_cfg"].get("expose_http_server", None)
            ):
                config["vllm_cfg"]["skip_tokenizer_init"] = False
            else:
                config["vllm_cfg"]["skip_tokenizer_init"] = True

    elif config["backend"] == "trtllm":
        config = cast(TrtllmConfig, config)

    return config
