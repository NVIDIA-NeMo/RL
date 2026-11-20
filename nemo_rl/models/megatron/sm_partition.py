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
"""Concurrent Megatron training and inference with shared GPU weight storage.

Each selected inference GPU also runs a learner under MPS. The default
``mps_gemm`` backend gives both clients full SM access and applies an advisory
Transformer Engine grouped-GEMM budget on shared learner GPUs. Explicit
``mps_percentage`` caps the learner instead. Neither reserves disjoint SMs.

With ``engine_weights=shared``, compatible contiguous parameter views alias
learner storage through same-GPU CUDA IPC. Strided/packed weights and persistent
buffers retain inference storage. Native logical shard metadata identifies local
sources across different TP, EP, ETP and training CP/PP layouts. When a destination
needs another GPU's shard, the native reshard executor runs among learner ranks
and writes directly into the paired inference process's exported receive buffers.
There is no additional persistent shadow weight model.

Inference pauses and drains admitted CUDA work before the optimizer can write
shared weights. Publication verifies learner storage, refreshes receive handles
when inference rehomes private tensors, completes residual and local copies,
refreshes derived caches, and advances the inference weight epoch before resume.
The same publication protocol serves ``engine_weights=copy``.

Native DDP learners can pack expert projections in inference order before buffer
allocation. Compatible full expert stacks remain shared; other layouts use
Megatron's own packing. Optimizer checkpoints for reordered parameters must use
model-space names rather than bucket positions.

Both worker worlds use the same GPUs. Inference remains subject to its backend's
CP1/PP1 and expert-TP constraints. Virtual pipeline chunks, GTP rematerialization
and quantized refit storage are unsupported. New layouts require distributed GPU
and full-model validation before performance or deployment claims.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional

import torch

if TYPE_CHECKING:
    from megatron.bridge.training.config import CheckpointConfig
    from megatron.core.resharding.utils import ParameterMetadata

Role = Literal["engine", "learner"]
MPS_PCT_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
_FOREACH_COPY_MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class RefitSlices:
    """Local learner views that populate one private inference weight."""

    shape: tuple[int, ...]
    parts: tuple[tuple[torch.Tensor, tuple[slice, ...]], ...]


RefitSource = torch.Tensor | RefitSlices


@dataclass(frozen=True)
class RefitPlacement:
    """Physical GPU identity and native distributed rank of one worker."""

    device_id: str
    rank: int


@dataclass(frozen=True)
class RefitExport:
    """Stable learner storage exported to its colocated inference process."""

    placement: RefitPlacement
    metadata: list[dict[str, Any]]
    handles: dict[str, tuple]
    fingerprint: str


@dataclass(frozen=True)
class RefitBinding:
    """Local bindings and remaining destinations in learner rank coordinates."""

    residual_metadata: list[dict[str, Any]]
    stats: dict[str, float]


@dataclass(frozen=True)
class RefitDestination:
    """Inference-owned receive storage; handles change only when storage moves."""

    device_id: str
    fingerprint: str
    handles: Optional[dict[str, tuple]]


def engine_sm_fraction(policy_cfg: Any) -> Optional[float]:
    """The configured engine SM fraction, or None (not co-resident)."""
    gen = (policy_cfg or {}).get("generation") or {}
    frac = (gen.get("colocated") or {}).get("engine_sm_fraction")
    return None if frac is None else float(frac)


def validate(
    policy_cfg: Any,
    *,
    env: Optional[dict] = None,
    unsupported: Optional[dict[str, bool]] = None,
) -> None:
    """Reject configurations co-residence cannot serve (the driver calls it at setup).

    ``unsupported``: other features of the run, by name, that co-residence does not support yet (true = in use).
    """
    frac = engine_sm_fraction(policy_cfg)
    if frac is None:
        gen = (policy_cfg or {}).get("generation") or {}
        placement = gen.get("colocated") or {}
        for key in ("learner_gpu_indices", "engine_gpu_indices"):
            if placement.get(key) is not None:
                raise ValueError(f"{key} requires co-resident Megatron generation")
        return
    env = os.environ if env is None else env
    if not 0.0 <= frac < 1.0:
        raise ValueError(
            f"policy.generation.colocated.engine_sm_fraction must be in [0, 1), got {frac}"
        )
    gen = policy_cfg["generation"]
    if gen.get("backend") != "megatron":
        raise ValueError(
            "engine_sm_fraction (co-resident generation) needs the Megatron generation backend"
        )
    if gen["colocated"].get("enabled", False):
        raise ValueError(
            "engine_sm_fraction runs the engine as its own processes on the training GPUs: set "
            "policy.generation.colocated.enabled=false (the in-process colocated engine stands down for training)"
        )
    if engine_weights_mode(policy_cfg) not in ("copy", "shared"):
        raise ValueError(
            "policy.generation.colocated.engine_weights must be 'copy' or 'shared'"
        )
    backend = partition_backend(policy_cfg)
    if backend == "mps_gemm":
        worker_env = policy_cfg["megatron_cfg"].get("env_vars") or {}
        for key in ("NVTE_EXT_MARGIN_SM", "CUDA_MPS_SM_PARTITION"):
            if key in env or key in worker_env:
                raise ValueError(
                    f"mps_gemm manages its GEMM budget and requires full SM visibility; remove {key}"
                )
    if backend == "mps_static" and frac == 0.0:
        raise ValueError("mps_static requires a positive engine_sm_fraction")
    if backend not in BACKENDS:
        raise ValueError(
            f"policy.generation.colocated.sm_partition_backend must be one of {BACKENDS}, got {backend!r}"
        )
    if backend == "mps_static" and not env.get(MPS_PARTITION_MAP_ENV):
        raise ValueError(
            f"sm_partition_backend='mps_static' needs {MPS_PARTITION_MAP_ENV}: the per-node map of MPS static SM "
            "partitions the job created before Ray (driver R590+ / CUDA 13.1+, MPS daemon started with -S)"
        )
    if gen.get("refit_transport") != "ipc":
        raise ValueError(
            "engine_sm_fraction needs policy.generation.refit_transport=ipc (same-GPU CUDA IPC refit)"
        )
    if not env.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise ValueError(
            "engine_sm_fraction needs an MPS control daemon on every node, started before Ray, with "
            "CUDA_MPS_PIPE_DIRECTORY exported (without MPS, two processes on one GPU time-slice)"
        )
    # Use the same CP/ETP resolution as the actual inference model.
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    engine = merged_inference_megatron_cfg(policy_cfg)
    if engine["pipeline_model_parallel_size"] != 1:
        raise ValueError(
            "engine_sm_fraction requires inference pipeline parallelism PP1"
        )
    for what, on in (unsupported or {}).items():
        if on:
            raise ValueError(f"engine_sm_fraction with {what} is not built yet")


#: Worker scheduling backends selected by ``colocated.sm_partition_backend``.
#: ``mps_gemm``: full SM access with a learner TE grouped-GEMM hint (default).
#: ``mps_percentage``: learner active-thread cap; inference stays uncapped.
#: ``mps_static``: pre-created partitions from NRL_MPS_SM_PARTITION_MAP;
#: requires an R590+ daemon started with -S and remains hardware-unqualified.
BACKENDS = ("mps_percentage", "mps_static", "mps_gemm")
MPS_PARTITION_MAP_ENV = "NRL_MPS_SM_PARTITION_MAP"


def partition_backend(policy_cfg: Any) -> str:
    gen = (policy_cfg or {}).get("generation") or {}
    return (gen.get("colocated") or {}).get("sm_partition_backend") or "mps_gemm"


def engine_weights_mode(policy_cfg: Any) -> str:
    """Return whether inference weights are copied or shared.

    ``colocated.engine_weights``: "copy" (the engine keeps its own weights, refit = per-step copy) or "shared" (the
    engine maps the learner's weights wherever the local shards match; see bind_shared).
    """
    gen = (policy_cfg or {}).get("generation") or {}
    return (gen.get("colocated") or {}).get("engine_weights") or "copy"


def mps_percentage(frac: float, role: Role) -> Optional[int]:
    """Return the learner cap; zero fraction leaves both MPS clients uncapped."""
    if role == "engine":
        return None
    if frac == 0.0:
        return 100
    return 100 - max(1, min(99, round(100 * frac)))


def _worker_percentage(policy_cfg: Any, role: Role) -> Optional[int]:
    backend = partition_backend(policy_cfg)
    if backend == "mps_gemm":
        return mps_percentage(0.0, role)
    if backend == "mps_percentage":
        return mps_percentage(engine_sm_fraction(policy_cfg), role)
    return None


def actor_env(
    policy_cfg: Any, role: Role, *, env: Optional[dict] = None
) -> dict[str, str]:
    """Environment of a co-resident worker of ``role`` (empty when not co-resident)."""
    frac = engine_sm_fraction(policy_cfg)
    if frac is None:
        return {}
    env = os.environ if env is None else env
    out = {}
    pct = _worker_percentage(policy_cfg, role)
    if pct is not None:
        out[MPS_PCT_ENV] = str(pct)
    for k in ("CUDA_MPS_PIPE_DIRECTORY", MPS_PARTITION_MAP_ENV):
        if env.get(k):
            out[k] = env[k]
    return out


def select_static_partition(
    policy_cfg: Any, role: Role, gpu_index: int
) -> Optional[str]:
    """Select the worker static MPS partition before CUDA initializes.

    ``mps_static``: in a co-resident worker BEFORE its first CUDA call, pick this GPU's partition for ``role``
    (``CUDA_MPS_SM_PARTITION``). No-op for the other backends.
    """
    if (
        engine_sm_fraction(policy_cfg) is None
        or partition_backend(policy_cfg) != "mps_static"
    ):
        return None
    import json

    with open(os.environ[MPS_PARTITION_MAP_ENV]) as f:
        part = json.load(f)[str(gpu_index)]
    os.environ["CUDA_MPS_SM_PARTITION"] = str(part[role])
    os.environ["NRL_MPS_SM_PARTITION_SMS"] = str(part[f"{role}_sms"])
    return os.environ["CUDA_MPS_SM_PARTITION"]


def check_worker(
    policy_cfg: Any, role: Role, *, rank: int = 0
) -> Optional[dict[str, Any]]:
    """Verify MPS registration; ``rank`` follows the role's logical GPU order."""
    if engine_sm_fraction(policy_cfg) is None:
        return None
    seen = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if partition_backend(policy_cfg) == "mps_static":
        want = int(os.environ.get("NRL_MPS_SM_PARTITION_SMS", "0"))
        info = {
            "role": role,
            "mps_sm_partition": os.environ.get("CUDA_MPS_SM_PARTITION"),
            "sms_seen": seen,
            "sms_partition": want,
        }
        if seen != want:
            raise RuntimeError(
                f"co-resident {role} worker: the MPS static partition was not applied: {info}"
            )
        print(f"[sm_partition] rank={rank} {info}", flush=True)
        return info
    pct = os.environ.get(MPS_PCT_ENV)
    expected_pct = _worker_percentage(policy_cfg, role)
    if (role == "engine" and pct not in (None, "100")) or (
        role == "learner" and pct != str(expected_pct)
    ):
        raise RuntimeError(
            f"co-resident {role} worker: {MPS_PCT_ENV}={pct!r}, expected "
            f"{'unset or 100 (uncapped)' if role == 'engine' else str(expected_pct)}; "
            "check the inherited environment and policy.megatron_cfg.env_vars before CUDA initialization"
        )
    if role == "engine":
        info = {"role": role, "mps_pct": pct, "sms_seen": seen}
        print(f"[sm_partition] rank={rank} {info}", flush=True)
        return info
    full = _device_sm_count()
    if expected_pct == 100:
        info = _check_uncapped_client(role, pct, seen=seen, full=full, rank=rank)
        if partition_backend(policy_cfg) == "mps_gemm":
            shares_engine_gpu = _learner_shares_engine_gpu(policy_cfg, rank=rank)
            fraction = engine_sm_fraction(policy_cfg) if shares_engine_gpu else 0.0
            margin = min(full - 1, max(0, round(full * fraction)))
            os.environ["NVTE_EXT_MARGIN_SM"] = str(margin)
            info["shares_engine_gpu"] = shares_engine_gpu
            info["te_grouped_gemm_target_sms"] = full - margin
            print(f"[sm_partition] rank={rank} {info}", flush=True)
        return info
    return _check_limit(role, pct, seen, full, rank)


def _learner_shares_engine_gpu(policy_cfg: Any, *, rank: int) -> bool:
    """Map a learner rank through its ordered view into the parent GPU cluster."""
    placement = policy_cfg["generation"]["colocated"]
    learner_indices = placement.get("learner_gpu_indices")
    if rank < 0 or (learner_indices is not None and rank >= len(learner_indices)):
        raise ValueError(f"Learner rank {rank} is outside its GPU subset")
    engine_indices = placement.get("engine_gpu_indices")
    if engine_indices is None:
        return True
    # RayVirtualCluster.subset and Policy preserve this logical order even
    # when the parent's topology sort permutes physical placement bundles.
    parent_index = rank if learner_indices is None else learner_indices[rank]
    return parent_index in engine_indices


def _check_uncapped_client(
    role: Role, pct: Optional[str], *, seen: int, full: int, rank: int
) -> dict[str, Any]:
    """Verify daemon registration when an uncapped client cannot prove MPS by its SM count."""
    if seen != full:
        raise RuntimeError(
            f"uncapped co-resident {role} worker sees {seen} SMs, expected {full}"
        )
    torch.cuda.current_stream().synchronize()

    def control(command: str) -> list[str]:
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.split()

    for attempt in range(5):
        try:
            servers = control("get_server_list")
            registered = any(
                str(os.getpid()) in control(f"get_client_list {server}")
                for server in servers
            )
        except subprocess.CalledProcessError as error:
            if attempt == 4:
                raise RuntimeError(
                    "Cannot verify uncapped learner registration with its MPS daemon"
                ) from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                "Cannot verify uncapped learner registration with its MPS daemon"
            ) from error
        else:
            if registered:
                break
            if attempt == 4:
                raise RuntimeError(
                    "Uncapped learner is not registered with its MPS daemon"
                )
        time.sleep(0.1)
    info = {
        "role": role,
        "mps_pct": pct,
        "sms_seen": seen,
        "sms_device": full,
        "mps_client_registered": True,
    }
    print(f"[sm_partition] rank={rank} {info}", flush=True)
    return info


def _check_limit(
    role: Role, pct: Optional[str], seen: int, full: int, rank: int
) -> dict[str, Any]:
    info = {"role": role, "mps_pct": pct, "sms_seen": seen, "sms_device": full}
    if pct is None or not seen < full:
        raise RuntimeError(
            f"co-resident {role} worker is not running under its MPS limit (no MPS daemon?): {info}"
        )
    print(f"[sm_partition] rank={rank} {info}", flush=True)
    return info


def _device_sm_count() -> int:
    """Query the physical GPU SM count outside the current MPS client cap.

    The GPU's full SM count, unaffected by this process's MPS limit: a short CUDA probe in a child process started
    without the percentage (under MPS the CUDA attribute shows the limit; NVML's attribute query is MIG-only and raises
    NotSupported on a full GB200).
    """
    import sys

    env = {k: v for k, v in os.environ.items() if k != MPS_PCT_ENV}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; "
            f"print(torch.cuda.get_device_properties({torch.cuda.current_device()}).multi_processor_count)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    return int(out.stdout.strip().splitlines()[-1])


# ---- refit by CUDA IPC (same GPU) ----


def refit_core(model: Any) -> torch.nn.Module:
    """The module whose ``named_refit_tensors`` a refit walks (the unwrapped model)."""
    from megatron.core.utils import unwrap_model

    if isinstance(model, (list, tuple)):
        if len(model) != 1:
            raise ValueError(
                "co-resident refit requires one model chunk per worker; virtual pipeline is unsupported"
            )
        model = model[0]
    return unwrap_model(model)


def export_refit_handles(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, tuple]:
    """Learner side, once: picklable CUDA IPC handles of its local refit tensors, by name."""
    from torch.multiprocessing.reductions import reduce_tensor

    return {name: reduce_tensor(t.detach()) for name, t in named}


def storage_fingerprint(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Host-only hash of (name, data_ptr, numel) of the refit tensors: changes iff any of them moved."""
    import hashlib

    h = hashlib.sha1()
    for name, t in named:
        h.update(
            f"{name}:{t.data_ptr()}:{tuple(t.shape)}:{t.stride()}:{t.dtype}:{t.device};".encode()
        )
    return h.hexdigest()


@dataclass(frozen=True)
class ParallelismLayout:
    """Resolved parameter ownership dimensions used for worker placement."""

    tp: int
    ep: int
    etp: int
    pp: int
    cp: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ParallelismLayout":
        tp = config["tensor_model_parallel_size"]
        # Megatron resolves an omitted expert TP size to the dense TP size.
        etp = config.get("expert_tensor_parallel_size")
        return cls(
            tp,
            config["expert_model_parallel_size"],
            tp if etp is None else etp,
            config["pipeline_model_parallel_size"],
            config["context_parallel_size"],
        )

    def validate_world(self, world: int) -> None:
        if type(world) is not int or world < 1:
            raise ValueError(
                "co-resident placement needs a positive integer world size"
            )
        if any(
            type(size) is not int or size < 1
            for size in (self.tp, self.ep, self.etp, self.pp, self.cp)
        ):
            raise ValueError("co-resident parallelism sizes must be positive integers")
        if world % (self.tp * self.cp * self.pp) or world % (
            self.etp * self.ep * self.pp
        ):
            raise ValueError(
                "co-resident world size must be divisible by both dense and expert model-parallel group sizes"
            )


def validate_gpu_subsets(policy_cfg: Any, world: int) -> None:
    """Validate ordered roles against the physical pool before Ray reserves GPUs."""
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    placement = policy_cfg["generation"]["colocated"]
    if all(
        placement.get(key) is None
        for key in ("learner_gpu_indices", "engine_gpu_indices")
    ):
        return
    physical = range(world)
    learner = placement.get("learner_gpu_indices")
    engine = placement.get("engine_gpu_indices")
    learner = physical if learner is None else learner
    engine = physical if engine is None else engine
    for role, indices in (("learner", learner), ("engine", engine)):
        if (
            not indices
            or any(type(i) is not int or i not in physical for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(
                f"{role}_gpu_indices must be nonempty, unique and in range [0, {world})"
            )
    if not set(engine) <= set(learner):
        raise ValueError("Each engine_gpu_indices entry must also be a learner GPU")
    ParallelismLayout.from_config(policy_cfg["megatron_cfg"]).validate_world(
        len(learner)
    )
    ParallelismLayout.from_config(
        merged_inference_megatron_cfg(policy_cfg)
    ).validate_world(len(engine))


def engine_rank_gpus(
    world: int, *, learner: ParallelismLayout, engine: ParallelismLayout
) -> list[int]:
    """Match joint dense/expert ownership when all shards can remain local.

    Megatron uses TP-fastest and PP-last ordering. CP changes sequence ownership,
    while dense parameters repeat across CP ranks. Expert TP/EP coordinates use
    their independent rank lattice. Unaligned layouts retain normal placement
    and use the residual transfer plan for missing local weights.
    """
    for layout in (learner, engine):
        layout.validate_world(world)
    identity = list(range(world))
    if learner.pp != 1 or engine.pp != 1:
        return identity
    if engine.tp % learner.tp or engine.etp % learner.etp or engine.ep % learner.ep:
        return identity
    tp_ratio, etp_ratio, ep_ratio = (
        engine.tp // learner.tp,
        engine.etp // learner.etp,
        engine.ep // learner.ep,
    )
    owners: dict[tuple[int, int, int], deque[int]] = defaultdict(deque)
    for rank in range(world):
        owners[
            (rank % learner.tp, rank % learner.etp, (rank // learner.etp) % learner.ep)
        ].append(rank)
    placement = []
    for rank in range(world):
        needed = (
            (rank % engine.tp) // tp_ratio,
            (rank % engine.etp) // etp_ratio,
            ((rank // engine.etp) % engine.ep) // ep_ratio,
        )
        if not owners[needed]:
            return identity
        placement.append(owners[needed].popleft())
    return placement


def refit_metadata(core: torch.nn.Module) -> list[ParameterMetadata]:
    """Extract native logical shard coordinates for this rank's refit tensors."""
    # Megatron is optional outside policy and generation workers.
    from megatron.core.resharding.planner import _extract_module_metadata

    return _extract_module_metadata(
        core, torch.distributed.get_rank(), core.config.num_moe_experts, 0, {}
    )


def open_refit_handles(handles: dict[str, tuple]) -> dict[str, torch.Tensor]:
    """Engine side, once: map the learner's tensors into this process (views of the learner's memory, by name)."""
    return {name: rebuild(*args) for name, (rebuild, args) in handles.items()}


def bind_shared(
    src: dict[str, RefitSource],
    named_dst: Iterable[tuple[str, torch.Tensor]],
    leaf_ids: set[int],
) -> dict[str, float]:
    """Bind compatible inference parameters to learner CUDA IPC storage.

    Engine side, once, before the engine's first forward: point every engine tensor whose local shard equals the
    learner's (same shape and dtype, contiguous, and a module's own parameter, not a derived view) at the
    learner's memory. Returns the tensors / bytes bound. The engine may still re-home a bound tensor later (MCore
    concatenates expert weights into its own buffer on the first inference forward); ``copy_refit`` copies whatever
    no longer aliases its source, so a binding is a memory saving, never a correctness assumption.
    """
    dst = dict(named_dst)
    _check_names(src, dst)
    bound = 0
    for name, d in dst.items():
        s = src[name]
        if s.shape != d.shape:
            raise ValueError(
                f"co-resident refit: {name} has shape {tuple(s.shape)} (learner) vs {tuple(d.shape)} (engine)"
            )
        # A strided view is never bound: TE GEMMs read weights as dense (TE 2.15 is silently wrong on strided ones).
        # Persistent buffers can change during backward, before the optimizer pause.
        if (
            isinstance(s, torch.Tensor)
            and isinstance(d, torch.nn.Parameter)
            and s.dtype == d.dtype
            and s.is_contiguous()
            and id(d) in leaf_ids
        ):
            d.data = s
            bound += 1
    return {"tensors_bound": bound, "tensors": len(dst)}


def _check_names(src: dict[str, RefitSource], dst: dict[str, torch.Tensor]) -> None:
    if dst.keys() != src.keys():
        missing, extra = sorted(set(dst) - set(src)), sorted(set(src) - set(dst))
        raise ValueError(
            "co-resident refit: engine tensors differ by name from the planned local sources: "
            f"engine-only {missing[:8]} ({len(missing)}), learner-only {extra[:8]} ({len(extra)})"
        )


def _contiguous_expert_stack(weights: list[torch.Tensor]) -> Optional[torch.Tensor]:
    """View equally shaped experts as one stack only when their storage is already packed in order."""
    if not weights:
        return None
    first = weights[0]
    if (
        type(first) not in (torch.Tensor, torch.nn.Parameter)
        or first.ndim != 2
        or first.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return None
    storage = first.untyped_storage()
    for index, weight in enumerate(weights):
        if (
            type(weight) not in (torch.Tensor, torch.nn.Parameter)
            or weight.shape != first.shape
            or weight.dtype != first.dtype
            or weight.device != first.device
            or not weight.is_contiguous()
            or weight.untyped_storage().data_ptr() != storage.data_ptr()
            or weight.untyped_storage().nbytes() != storage.nbytes()
            or weight.storage_offset() != first.storage_offset() + index * first.numel()
        ):
            return None
    return first.as_strided(
        (len(weights), *first.shape), (first.numel(), *first.stride())
    )


def _reverse_expert_parameter_order(
    linear: torch.nn.Module, *, num_experts: int
) -> bool:
    """Keep expert names and objects while ordering weights for native DDP's reversed layout."""
    names = [f"weight{index}" for index in range(num_experts)]
    parameters = linear._parameters
    if not names or any(name not in parameters for name in names):
        return False
    weights = [parameters[name] for name in names]
    if any(
        type(weight) is not torch.nn.Parameter
        or not weight.requires_grad
        or weight.ndim != 2
        or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not weight.is_contiguous()
        or weight.shape != weights[0].shape
        or weight.dtype != weights[0].dtype
        for weight in weights
    ):
        return False
    weight_names = set(names)
    ordered = [
        (name, value) for name, value in parameters.items() if name not in weight_names
    ]
    ordered.extend((name, parameters[name]) for name in reversed(names))
    parameters.clear()
    parameters.update(ordered)
    return True


def configure_shared_expert_checkpoint(checkpoint_config: CheckpointConfig) -> None:
    """Use model-space optimizer checkpoints for the shared expert parameter layout.

    Reordered parameters cannot restore moments from positional optimizer buckets.
    Validate the saved format before model setup and require name-based saves for
    subsequent resumes, including resumes into a learner without weight sharing.
    """
    if (
        checkpoint_config.load_optim
        and not checkpoint_config.finetune
        and checkpoint_config.load is not None
    ):
        # Bridge and the distributed optimizer are optional outside this learner.
        from megatron.bridge.training.checkpointing import (
            _DIRECT_ITERATION_DIR_SENTINEL,
            _resolve_checkpoint_iteration,
        )
        from megatron.bridge.training.utils.checkpoint_utils import get_checkpoint_name
        from megatron.core import dist_checkpointing
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

        iteration, release = _resolve_checkpoint_iteration(
            checkpoint_config.load, checkpoint_config.ckpt_step
        )
        if not release and iteration != -1:
            checkpoint_dir = (
                checkpoint_config.load
                if iteration == _DIRECT_ITERATION_DIR_SENTINEL
                else get_checkpoint_name(
                    checkpoint_config.load, iteration, release=False
                )
            )
            common_state = dist_checkpointing.load_common_state_dict(checkpoint_dir)
            if "optimizer" in common_state:
                metadata = (
                    dist_checkpointing.load_content_metadata(
                        preloaded_state_dict=common_state
                    )
                    or {}
                )
                saved_format = metadata.get("distrib_optim_sharding_type")
                if (
                    saved_format
                    not in DistributedOptimizer.checkpoint_fully_reshardable_formats
                ):
                    raise ValueError(
                        "Shared expert weights require a model-space optimizer checkpoint; "
                        f"{checkpoint_dir!r} uses {saved_format!r}. Bucket-position optimizer state cannot "
                        "be restored after expert parameter reordering. Convert the checkpoint to "
                        "fully_reshardable format with its original parameter layout, or load weights "
                        "with a fresh optimizer."
                    )
    # Expert packing changes bucket positions; moment identity must follow model names.
    checkpoint_config.dist_ckpt_optim_fully_reshardable = True


def prepare_shared_expert_layout(
    models: list[torch.nn.Module],
) -> list[torch.nn.Module]:
    """Pre-DDP hook: pack ordinary expert weights in inference order without changing their kernels."""
    # Megatron's expert modules are optional outside the co-resident learner.
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    linears = 0
    for model in models:
        for module in model.modules():
            if isinstance(module, TEGroupedMLP):
                for linear in (module.linear_fc1, module.linear_fc2):
                    linears += _reverse_expert_parameter_order(
                        linear, num_experts=module.num_local_experts
                    )
    print(
        f"[sm_partition] expert projections prepared before DDP: {linears}", flush=True
    )
    return models


def bind_shared_expert_stacks(
    core: torch.nn.Module, src: dict[str, RefitSource]
) -> dict[str, float]:
    """Keep compatible BF16/FP16 expert stacks in learner storage through inference initialization.

    Both projections must already alias their corresponding source parameters and be packed in expert order.
    Other layouts retain Megatron's ordinary lazy packing and per-refit copies.
    """
    # The co-resident engine is the only consumer of optional Megatron inference modules.
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    weight_groups = (("linear_fc1", "_fc1_weight"), ("linear_fc2", "_fc2_weight"))
    layers = shared_bytes = 0
    for prefix, module in core.named_modules():
        if not isinstance(module, InferenceGroupedMLP):
            continue
        if module._concatenated_weights_built:
            raise RuntimeError(
                "shared expert stacks must be bound before the first inference forward"
            )
        stacks = {}
        for linear_name, buffer_name in weight_groups:
            linear = getattr(module, linear_name)
            weights = []
            for index in range(module.num_local_experts):
                name = f"{prefix + '.' if prefix else ''}{linear_name}.weight{index}"
                weight = getattr(linear, f"weight{index}")
                source = src.get(name)
                if (
                    not isinstance(source, torch.Tensor)
                    or not isinstance(weight, torch.nn.Parameter)
                    or not _aliases(weight, source)
                ):
                    break
                weights.append(source)
            if len(weights) != module.num_local_experts:
                break
            stack = _contiguous_expert_stack(weights)
            if stack is None:
                break
            stacks[buffer_name] = stack
        if len(stacks) != len(weight_groups):
            continue
        for buffer_name, stack in stacks.items():
            module.register_buffer(buffer_name, stack, persistent=False)
            shared_bytes += stack.numel() * stack.element_size()
        module._uses_mxfp8_weights = False
        module._concatenated_weights_built = True
        layers += 1
    return {"shared_expert_layers": layers, "shared_expert_bytes": shared_bytes}


def _aliases(d: torch.Tensor, s: torch.Tensor) -> bool:
    return (
        d.data_ptr() == s.data_ptr()
        and d.shape == s.shape
        and d.stride() == s.stride()
        and d.dtype == s.dtype
    )


def copy_refit(
    src: dict[str, RefitSource], named_dst: Iterable[tuple[str, torch.Tensor]]
) -> dict[str, float]:
    """Copy private inference tensors and account for shared storage.

    Engine side, per step (engine paused, learner idle): copy every learner tensor into the engine tensor of the
    same name unless that engine tensor IS the learner's memory. The engine's tensors are enumerated live, so a tensor
    the engine re-homed after setup is still the one written. Returns the bytes copied and the bytes shared.
    """
    dst = dict(named_dst)
    _check_names(src, dst)
    copied = shared = 0
    copy_groups = {}
    # The engine's tensors can be inference tensors (allocated lazily under inference mode); write them there too.
    with torch.inference_mode():
        for name, d in dst.items():
            s = src[name]
            if s.shape != d.shape:
                raise ValueError(
                    f"co-resident refit: {name} has shape {tuple(s.shape)} (learner) vs {tuple(d.shape)} (engine)"
                )
            if isinstance(s, RefitSlices):
                for source_view, target_slice in s.parts:
                    d[target_slice].copy_(source_view)
                    copied += source_view.numel() * source_view.element_size()
                continue
            nbytes = s.numel() * s.element_size()
            if _aliases(d, s):
                shared += nbytes
                continue
            if (
                nbytes <= _FOREACH_COPY_MAX_BYTES
                and s.is_contiguous()
                and d.is_contiguous()
            ):
                # Mixed dtype lists force foreach back to one copy per tensor.
                key = (s.device, d.device, s.dtype, d.dtype)
                group = copy_groups.get(key)
                if group is None:
                    group = ([], [])
                    copy_groups[key] = group
                group[0].append(s)
                group[1].append(d)
            else:
                # Enqueue large copies while the host prepares subsequent tensors.
                d.copy_(s)
            copied += nbytes
        for copy_src, copy_dst in copy_groups.values():
            torch._foreach_copy_(copy_dst, copy_src)
    torch.cuda.current_stream().synchronize()
    return {"bytes": float(copied), "bytes_shared": float(shared)}
