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

"""Fixed-topology reproducibility controls and schedule-independent RNG keys."""

import hashlib
import math
import os
import random
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.models.generation.megatron.config import MCoreGenerationConfig
    from nemo_rl.models.policy import MegatronConfig


DETERMINISTIC_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":0:0",
    "CUBLASLT_WORKSPACE_SIZE": "0",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    # Mamba selects its Triton configurations at import time, before the
    # worker constructor enables torch determinism. Cached timing winners
    # can differ across ranks or cold starts, so use its fixed configuration.
    "MAMBA_DETERMINISTIC": "1",
    "CAUSAL_CONV1D_DETERMINISTIC": "1",
    "TRITON_CACHE_AUTOTUNING": "0",
}


def keyed_seed(seed: int, *coordinates: int | str) -> int:
    """Derive a portable signed-int64 seed without Python's randomized hash.

    Coordinates identify logical work (phase, step, sample, turn, token), never
    worker ranks, arrival order, wall time, or engine-assigned request IDs.
    """
    payload = repr((seed, *coordinates)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def configure_torch_determinism() -> None:
    """Enable strict deterministic PyTorch execution in the calling process."""
    for name, value in DETERMINISTIC_ENV.items():
        if os.environ.get(name) != value:
            raise RuntimeError(
                f"Determinism requires {name}={value} before worker startup"
            )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def configure_grpo_determinism(config: "MasterConfig") -> None:
    """Resolve strict GRPO settings before creating Ray actors or preprocessing data.

    Supports text-only synchronous and fixed-lag asynchronous Megatron GRPO
    on a fixed software stack and parallel layout. Environment determinism remains
    the caller's responsibility outside the supported math verifier.
    """
    if not config.grpo.deterministic:
        return
    if not 0 <= config.grpo.seed < 2**32:
        raise ValueError("Deterministic GRPO requires a seed in [0, 2**32)")
    policy = config.policy
    if config.data["use_multiple_dataloader"]:
        raise ValueError("Deterministic GRPO requires a single input data stream")
    if (
        not policy["megatron_cfg"]["enabled"]
        or policy["generation"]["backend"] != "megatron"
    ):
        raise ValueError("grpo.deterministic requires Megatron training and generation")
    generation = cast("MCoreGenerationConfig", policy["generation"])
    mcfg = cast("MegatronConfig", policy["megatron_cfg"])
    if (
        generation["temperature"] != 1.0
        or generation["top_p"] != 1.0
        or generation["top_k"] not in (None, 0)
    ):
        raise ValueError(
            "Batch-invariant GRPO currently requires temperature=1.0, top_p=1.0, top_k=null"
        )
    asynchronous = config.grpo.async_grpo is not None and config.grpo.async_grpo.enabled
    if asynchronous:
        async_config = config.grpo.async_grpo
        if (
            async_config.max_trajectory_age_steps != 1
            or async_config.in_flight_weight_updates
        ):
            raise ValueError(
                "Deterministic async GRPO requires one-step lag and no in-flight weight updates"
            )
        if async_config.max_generation_failures != 0:
            raise ValueError(
                "Deterministic async GRPO cannot drop failed prompt groups"
            )
        if generation["colocated"]["enabled"]:
            raise ValueError(
                "Deterministic async GRPO requires separate training and inference GPUs"
            )
        if config.checkpointing["enabled"]:
            raise ValueError(
                "Deterministic async checkpoint/resume is not supported yet; disable checkpointing"
            )
        if config.grpo.batch_multiplier != 1:
            raise ValueError("Deterministic async GRPO requires batch_multiplier=1")
    if mcfg["optimizer"]["optimizer_cpu_offload"]:
        raise ValueError(
            "Deterministic execution currently supports GPU-only optimizers"
        )
    if (
        config.data_plane is not None and config.data_plane["enabled"]
    ) or config.on_policy_distillation is not None:
        raise ValueError(
            "Deterministic GRPO currently requires the direct data path without distillation"
        )
    if (
        config.grpo.use_dynamic_sampling
        or policy["sequence_packing"].get("algorithm") == "first_fit_shuffle"
    ):
        raise ValueError(
            "Deterministic GRPO requires fixed sampling and non-random sequence packing"
        )
    if policy["precision"] != "bfloat16":
        raise ValueError("Batch-invariant Megatron requires policy.precision=bfloat16")
    if mcfg.get("flash_attention_version") not in (3, 4):
        raise ValueError(
            "Set policy.megatron_cfg.flash_attention_version to 3 or 4 for batch invariance"
        )
    if (
        mcfg.get("batch_invariant_backend") not in ("te_native", "triton")
        or mcfg.get("batch_invariant_collective") != "ordered"
    ):
        raise ValueError(
            "Strict GRPO requires batch_invariant_backend=te_native or triton and batch_invariant_collective=ordered"
        )
    unverified_environments = []
    for name, env_config in config.env.items():
        if (
            name == "math"
            and env_config.get("math_verify_impl") == "hf_math_verify"
            and (
                "verifier_type" not in env_config
                or env_config["verifier_type"] == "math"
            )
        ):
            env_config["deterministic"] = True
        else:
            unverified_environments.append(name)
    if unverified_environments:
        warnings.warn(
            "Deterministic GRPO training and inference settings remain enabled, "
            "but end-to-end reproducibility cannot be guaranteed for environments: "
            f"{', '.join(sorted(unverified_environments))}. These environments must "
            "produce deterministic observations and rewards for the same inputs "
            "and seeds, independently of worker scheduling, external state, and "
            "wall-clock timeouts. Environment settings are left unchanged.",
            UserWarning,
            stacklevel=2,
        )
    inference = generation["mcore_generation_config"]
    if inference["expose_http_server"] or inference["num_speculative_tokens"]:
        raise ValueError(
            "Deterministic GRPO does not support HTTP or speculative generation"
        )
    if config.checkpointing["checkpoint_must_save_by"] is not None:
        raise ValueError("Use step-based checkpointing for deterministic GRPO")
    if (
        config.checkpointing["enabled"]
        and config.checkpointing.get("save_optimizer") is False
    ):
        raise ValueError(
            "Deterministic resume requires checkpointing.save_optimizer=true"
        )
    for settings in (mcfg, inference):
        for key in ("fp8_cfg", "fp4_cfg"):
            precision_cfg = settings.get(key)
            if isinstance(precision_cfg, BaseModel):
                precision_cfg = precision_cfg.model_dump()
            if precision_cfg and precision_cfg["enabled"]:
                raise ValueError(f"Deterministic GRPO does not support {key}")
        settings["deterministic_mode"] = True
        settings["batch_invariant_mode"] = True
        settings["attention_backend"] = "flash"
        settings["moe_permute_fusion"] = False
        settings["flash_attention_version"] = mcfg["flash_attention_version"]
        settings["batch_invariant_backend"] = mcfg["batch_invariant_backend"]
        settings["batch_invariant_collective"] = mcfg["batch_invariant_collective"]
    inference["request_seeded_sampling"] = True
    inference["async_sched_mode"] = "legacy"
    # Preserve inference graph/cache/chunking choices. These change execution
    # shape, so their numerical invariance is supplied by the kernels above.
    # Training graph capture is a separate contract from inference graph replay.
    mcfg["cuda_graph_impl"] = "none"
    inference["logprobs_mode"] = "raw_logprobs"
    mcfg["seed"] = config.grpo.seed
    if mcfg["batch_invariant_backend"] == "triton":
        # Triton's batch-invariant matmul does not implement FP64 router GEMM.
        mcfg["moe_router_dtype"] = "fp32"
    else:
        model_overrides = mcfg.get("model_overrides")
        if (
            model_overrides
            and model_overrides.get("moe_use_grouped_tensor")
            and mcfg["moe_router_dtype"] == "fp64"
        ):
            raise ValueError(
                "TE grouped-tensor probability padding does not support FP64; "
                "set policy.megatron_cfg.moe_router_dtype=fp32"
            )
        # TE native RMSNorm changes reduction paths with M % 32. Match MCore
        # dynamic inference's 64-row alignment, including training microbatches.
        policy["make_sequence_length_divisible_by"] = math.lcm(
            policy["make_sequence_length_divisible_by"], 64
        )
    mcfg["distributed_data_parallel_config"]["overlap_grad_reduce"] = False
    mcfg["distributed_data_parallel_config"]["overlap_param_gather"] = False
    env = dict(mcfg.get("env_vars") or {})
    env.update(DETERMINISTIC_ENV)
    env["PYTHONHASHSEED"] = str(config.grpo.seed)
    mcfg["env_vars"] = env
    inference["env_vars"] = dict(env)
    os.environ.update(env)
    configure_torch_determinism()
    random.seed(config.grpo.seed)
    np.random.seed(config.grpo.seed)
    torch.manual_seed(config.grpo.seed)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot driver RNGs; dataloader state is saved separately by the trainer."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot after setup so initialization cannot consume replay RNGs."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
