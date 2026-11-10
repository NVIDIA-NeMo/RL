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
"""Run Megatron training and inference concurrently on shared GPUs with CUDA MPS.

Co-resident generation: the Megatron engine and the learner on the same GPUs, as two processes per GPU, each on its
own share of the SMs. This module is the one owner of the mechanism.

The knob: ``policy.generation.colocated.engine_sm_fraction = f`` (with ``colocated.enabled = false`` and
``refit_transport = ipc``). The generation fleet is the stock non-colocated Megatron engine (its own processes, process
groups and CUDA graphs), placed on the training GPUs: one learner rank and one engine rank per GPU. The training loop is
stock async: the engine never stands down.

SM split (``sm_partition_backend``, see BACKENDS): every process on a node is a client of one MPS daemon started by the
job before Ray. Under the default ``mps_percentage`` only the learner is capped: it runs with
``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = 100 - round(100 f)``, fixed at CUDA initialization for every kernel, CUDA graph and
NCCL kernel of the process. The engine is not capped and may use all SMs while the learner waits; during concurrent
work MPS schedules both clients without reserving a minimum SM share for the engine. A cap on the engine only slows
it, including while the learner is idle (measured: capping the engine at 42 % costs x1.23-1.56 per decode step at
32-95 rows, and uncapping it gave -10.5 % s/step at K2, tw-cobn coresident-bottlenecks.md F1). The cap is not a disjoint
SM set, and the driver rounds it
(58 % -> 88 of 152 SMs); ``check_worker`` logs the allocated count.

Layouts: TP / PP / ETP equal on both sides; the engine's EP may be m x the learner's (TP = ETP = 1), with each engine rank
placed on the GPU whose learner rank holds its experts (``engine_rank_gpus``). Tensors are matched by Megatron's refit
metadata (``refit_names``: the resolved name, with global expert indices), and every engine rank checks at setup that
its TP / PP coordinates equal the learner rank's on its GPU.

Refit (``refit_transport = ipc``):
- Once, at setup: each learner rank exports CUDA IPC handles of its local refit tensors (parameters and persistent
  buffers; DDP-owned storage, stable for the run). The engine rank on the same GPU maps them.
- ``colocated.engine_weights = "copy"``: the engine keeps its own weights. Per step, inside the loop's refit: pause at a
  step boundary (no suspend), copy every local tensor device to device, refresh derived weights, bump the prefix-cache
  weight epoch, unpause. Cost per rank per step: the local shard's bytes over HBM.
- ``"shared"``: every eligible engine parameter whose local shard equals the learner's (same shape and dtype) is bound to the
    learner's memory before the engine's first forward (``bind_shared``). Persistent buffers stay private because
    training may update them before the optimizer pause. Each refit copies whatever does not alias its
  source (``copy_refit``), so it stays correct if the engine re-homes a tensor. Native DDP learners prepare ordinary
  expert parameter order before buffer allocation; compatible packed expert projections become shared [E, N, K]
  inference views. Other storage layouts retain Megatron's lazy expert packing and per-refit copies.
  These learners save optimizer state by model name. Optimizer resume requires a model-space checkpoint;
  bucket-position checkpoints must be converted first or loaded with a fresh optimizer.
  The engine pauses before the optimizer step writes the weights and stays paused through the
  loop's refit (parameter all-gather, copy, epoch bump), which resumes it: the refit stays the one point where a new
  version is published. Saves the shared bytes per GPU (logged per step); the engine idles for the optimizer step.

Per-rank cost: one extra process and CUDA context per GPU, plus the engine's weight copy ("copy") or only its non-shared
tensors ("shared"). Nothing grows with TP / PP / EP or the node count.

Not built yet: PP > 1, engine TP != learner TP, distillation teachers, cluster segments, or reference-policy weight
swapping with shared engine weights (setup rejects them).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional

import torch

if TYPE_CHECKING:
    from megatron.bridge.training.config import CheckpointConfig

Role = Literal["engine", "learner"]
MPS_PCT_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
_FOREACH_COPY_MAX_BYTES = 512 * 1024
_LAYOUT_KEYS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
)


def engine_sm_fraction(policy_cfg: Any) -> Optional[float]:
    """The configured engine SM fraction, or None (not co-resident)."""
    gen = (policy_cfg or {}).get("generation") or {}
    frac = (gen.get("colocated") or {}).get("engine_sm_fraction")
    return None if frac is None else float(frac)


def validate(
    policy_cfg: Any, *, env: Optional[dict] = None, unsupported: Optional[dict[str, bool]] = None
) -> None:
    """Reject configurations co-residence cannot serve (the driver calls it at setup).

    ``unsupported``: other features of the run, by name, that co-residence does not support yet (true = in use).
    """
    frac = engine_sm_fraction(policy_cfg)
    if frac is None:
        return
    env = os.environ if env is None else env
    if not 0.0 < frac < 1.0:
        raise ValueError(f"policy.generation.colocated.engine_sm_fraction must be in (0, 1), got {frac}")
    gen = policy_cfg["generation"]
    if gen.get("backend") != "megatron":
        raise ValueError("engine_sm_fraction (co-resident generation) needs the Megatron generation backend")
    if gen["colocated"].get("enabled", False):
        raise ValueError(
            "engine_sm_fraction runs the engine as its own processes on the training GPUs: set "
            "policy.generation.colocated.enabled=false (the in-process colocated engine stands down for training)"
        )
    if engine_weights_mode(policy_cfg) not in ("copy", "shared"):
        raise ValueError("policy.generation.colocated.engine_weights must be 'copy' or 'shared'")
    backend = partition_backend(policy_cfg)
    if backend not in BACKENDS:
        raise ValueError(f"policy.generation.colocated.sm_partition_backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "mps_static" and not env.get(MPS_PARTITION_MAP_ENV):
        raise ValueError(
            f"sm_partition_backend='mps_static' needs {MPS_PARTITION_MAP_ENV}: the per-node map of MPS static SM "
            "partitions the job created before Ray (driver R590+ / CUDA 13.1+, MPS daemon started with -S)"
        )
    if gen.get("refit_transport") != "ipc":
        raise ValueError("engine_sm_fraction needs policy.generation.refit_transport=ipc (same-GPU CUDA IPC refit)")
    if not env.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise ValueError(
            "engine_sm_fraction needs an MPS control daemon on every node, started before Ray, with "
            "CUDA_MPS_PIPE_DIRECTORY exported (without MPS, two processes on one GPU time-slice)"
        )
    learner = policy_cfg["megatron_cfg"]
    engine = gen.get("mcore_generation_config") or {}
    mismatch = {}
    for k in _LAYOUT_KEYS:
        lv = learner.get(k) or 1
        ev = engine.get(k) or lv
        if lv != ev:
            mismatch[k] = (lv, ev)
    ep_l, ep_e = mismatch.pop("expert_model_parallel_size", (None, None))
    # engine_rank_gpus places an EP-multiple engine for TP = ETP = 1 only.
    if ep_l is not None and (
        ep_e % ep_l != 0
        or (learner.get("expert_tensor_parallel_size") or 1) != 1
        or (learner.get("tensor_model_parallel_size") or 1) != 1
    ):
        mismatch["expert_model_parallel_size"] = (ep_l, ep_e)
    if mismatch:
        raise ValueError(
            "engine_sm_fraction: one engine rank and one learner rank per GPU need the same TP / PP / ETP, and an "
            "engine EP that differs from the learner's must be a multiple of it at TP = ETP = 1 (each engine rank's "
            f"experts are a block of its GPU's learner experts); other mixed layouts are not built yet. "
            f"(learner, engine): {mismatch}"
        )
    if (learner.get("pipeline_model_parallel_size") or 1) > 1:
        raise ValueError("engine_sm_fraction with pipeline parallelism is not built yet")
    for what, on in (unsupported or {}).items():
        if on:
            raise ValueError(f"engine_sm_fraction with {what} is not built yet")


#: The SM-partition backend, the one swap point (``policy.generation.colocated.sm_partition_backend``):
#:   "mps_percentage" (default): an MPS active-thread percentage on the learner only, a soft cap; the engine is
#:                     uncapped. Measured 11-09 on driver 580 / CUDA 13.2: capped SM sets are not disjoint (all 62 of a
#:                     42 %-capped engine's SMs also ran learner kernels).
#:   "mps_static":     MPS static SM partitions (driver R590+, CUDA 13.1+; daemon started with -S): the job creates two
#:                     partitions per GPU before Ray, in 8-SM chunks (engine round(f x chunks), learner the rest), and
#:                     writes their ids to the node-local JSON map named by NRL_MPS_SM_PARTITION_MAP
#:                     ({gpu index: {"engine": id, "learner": id, "engine_sms": n, "learner_sms": m}}); each worker
#:                     selects its partition (CUDA_MPS_SM_PARTITION) before CUDA init; ``check_worker`` refuses a
#:                     process whose SM count is not its partition's. Disjoint by construction. Untested on
#:                     hardware (needs R590+).
BACKENDS = ("mps_percentage", "mps_static")
MPS_PARTITION_MAP_ENV = "NRL_MPS_SM_PARTITION_MAP"


def partition_backend(policy_cfg: Any) -> str:
    gen = (policy_cfg or {}).get("generation") or {}
    return (gen.get("colocated") or {}).get("sm_partition_backend") or "mps_percentage"


def engine_weights_mode(policy_cfg: Any) -> str:
    """Return whether inference weights are copied or shared.

    ``colocated.engine_weights``: "copy" (the engine keeps its own weights, refit = per-step copy) or "shared" (the
    engine maps the learner's weights wherever the local shards match; see bind_shared).
    """
    gen = (policy_cfg or {}).get("generation") or {}
    return (gen.get("colocated") or {}).get("engine_weights") or "copy"


def mps_percentage(frac: float, role: Role) -> Optional[int]:
    """The process's MPS active-thread percentage: the learner 100 - round(100 f) (1..99); the engine None (uncapped)."""
    if role == "engine":
        return None
    return 100 - max(1, min(99, round(100 * frac)))


def actor_env(policy_cfg: Any, role: Role, *, env: Optional[dict] = None) -> dict[str, str]:
    """Environment of a co-resident worker of ``role`` (empty when not co-resident)."""
    frac = engine_sm_fraction(policy_cfg)
    if frac is None:
        return {}
    env = os.environ if env is None else env
    out = {}
    pct = mps_percentage(frac, role) if partition_backend(policy_cfg) == "mps_percentage" else None
    if pct is not None:
        out[MPS_PCT_ENV] = str(pct)
    for k in ("CUDA_MPS_PIPE_DIRECTORY", MPS_PARTITION_MAP_ENV):
        if env.get(k):
            out[k] = env[k]
    return out


def select_static_partition(policy_cfg: Any, role: Role, gpu_index: int) -> Optional[str]:
    """Select the worker static MPS partition before CUDA initializes.

    ``mps_static``: in a co-resident worker BEFORE its first CUDA call, pick this GPU's partition for ``role``
    (``CUDA_MPS_SM_PARTITION``). No-op for the other backends.
    """
    if engine_sm_fraction(policy_cfg) is None or partition_backend(policy_cfg) != "mps_static":
        return None
    import json

    with open(os.environ[MPS_PARTITION_MAP_ENV]) as f:
        part = json.load(f)[str(gpu_index)]
    os.environ["CUDA_MPS_SM_PARTITION"] = str(part[role])
    os.environ["NRL_MPS_SM_PARTITION_SMS"] = str(part[f"{role}_sms"])
    return os.environ["CUDA_MPS_SM_PARTITION"]


def check_worker(policy_cfg: Any, role: Role, *, rank: int = 0) -> Optional[dict[str, Any]]:
    """In a co-resident worker after CUDA init: its SM share must be in effect. Logs and returns what it sees.

    Proof (capped learner): the process sees fewer SMs than the GPU has (NVML's count is unaffected by MPS). Without an
    MPS daemon the percentage is silently ignored and the two processes time-slice the whole GPU. The uncapped engine
    sees the whole GPU; the learner's check on the same GPU proves the daemon.
    """
    if engine_sm_fraction(policy_cfg) is None:
        return None
    seen = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if partition_backend(policy_cfg) == "mps_static":
        want = int(os.environ.get("NRL_MPS_SM_PARTITION_SMS", "0"))
        info = {"role": role, "mps_sm_partition": os.environ.get("CUDA_MPS_SM_PARTITION"), "sms_seen": seen,
                "sms_partition": want}
        if seen != want:
            raise RuntimeError(f"co-resident {role} worker: the MPS static partition was not applied: {info}")
        print(f"[sm_partition] rank={rank} {info}", flush=True)
        return info
    pct = os.environ.get(MPS_PCT_ENV)
    expected_pct = mps_percentage(engine_sm_fraction(policy_cfg), role)
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
    return _check_limit(role, pct, seen, _device_sm_count(), rank)


def _check_limit(role: Role, pct: Optional[str], seen: int, full: int, rank: int) -> dict[str, Any]:
    info = {"role": role, "mps_pct": pct, "sms_seen": seen, "sms_device": full}
    if pct is None or not seen < full:
        raise RuntimeError(f"co-resident {role} worker is not running under its MPS limit (no MPS daemon?): {info}")
    print(f"[sm_partition] rank={rank} {info}", flush=True)
    return info


def _device_sm_count() -> int:
    """Query the physical GPU SM count outside the current MPS client cap.

    The GPU's full SM count, unaffected by this process's MPS limit: a short CUDA probe in a child process started
    without the percentage (under MPS the CUDA attribute shows the limit; NVML's attribute query is MIG-only and raises
    NotSupported on a full GB200).
    """
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != MPS_PCT_ENV}
    out = subprocess.run(
        [sys.executable, "-c", "import torch; "
         f"print(torch.cuda.get_device_properties({torch.cuda.current_device()}).multi_processor_count)"],
        env=env, capture_output=True, text=True, check=True, timeout=180,
    )
    return int(out.stdout.strip().splitlines()[-1])


# ---- refit by CUDA IPC (same GPU) ----


def refit_core(model: Any) -> torch.nn.Module:
    """The module whose ``named_refit_tensors`` a refit walks (the unwrapped model)."""
    from megatron.core.utils import unwrap_model

    return unwrap_model(model[0] if isinstance(model, (list, tuple)) else model)


def export_refit_handles(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, tuple]:
    """Learner side, once: picklable CUDA IPC handles of its local refit tensors, by name."""
    from torch.multiprocessing.reductions import reduce_tensor

    return {name: reduce_tensor(t.detach()) for name, t in named}


def storage_fingerprint(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Host-only hash of (name, data_ptr, numel) of the refit tensors: changes iff any of them moved."""
    import hashlib

    h = hashlib.sha1()
    for name, t in named:
        h.update(f"{name}:{t.data_ptr()}:{t.numel()};".encode())
    return h.hexdigest()


def engine_rank_gpus(world: int, learner_ep: int, engine_ep: int) -> list[int]:
    """Map engine ranks to the GPUs that hold their expert weights.

    GPU of each engine rank when the engine's EP is m x the learner's: engine rank r (EP rank ep = r % engine_ep,
    expert-DP rank d = r // engine_ep) goes to the GPU whose learner rank holds its experts (learner EP rank
    ep // m) as a contiguous block, so its experts are views of that GPU's learner tensors. Identity when m == 1.
    Learner rank g is on GPU g; EP ranks are the fastest-varying expert-group coordinate (TP = ETP = 1).
    """
    m = engine_ep // learner_ep
    gpus = [learner_ep * (m * (r // engine_ep) + (r % engine_ep) % m) + (r % engine_ep) // m for r in range(world)]
    assert sorted(gpus) == list(range(world)), gpus
    return gpus


def refit_names(core: torch.nn.Module) -> dict[str, str]:
    """Resolve local refit tensor names through Megatron layout metadata.

    {local refit tensor name: Megatron's resolved name} of this process's model: the name Megatron's own refit
    planner matches tensors by across layouts (per-expert tensors carry their global expert index).
    """
    from megatron.core.resharding.planner import _extract_module_metadata

    metadata = _extract_module_metadata(core, torch.distributed.get_rank(), core.config.num_moe_experts, 0, {})
    return {m.name: m.resolved_name for m in metadata}


def rank_coordinates() -> dict[str, tuple[int, int]]:
    """(rank, size) of this process in its TP and PP groups."""
    from megatron.core import parallel_state as ps

    return {
        "tp": (ps.get_tensor_model_parallel_rank(), ps.get_tensor_model_parallel_world_size()),
        "pp": (ps.get_pipeline_model_parallel_rank(), ps.get_pipeline_model_parallel_world_size()),
    }


def map_sources(
    src: dict[str, torch.Tensor],
    src_coords: dict[str, tuple[int, int]],
    dst_names: dict[str, str],
    dst_coords: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Map inference tensors to compatible learner shards on the same GPU.

    The learner tensor behind each engine tensor. ``src``: the learner's tensors on this GPU by resolved name;
    ``dst_names``: the engine's {local name: resolved name}. The two ranks must hold the same TP / PP shard (else a
    dense tensor of equal name and shape would be another rank's shard), and every engine tensor, each expert by its
    global index, must be on this GPU's learner rank.
    """
    if src_coords != dst_coords:
        raise ValueError(
            f"co-resident refit: the engine rank on this GPU is at {dst_coords} but the learner rank at {src_coords} "
            "(TP / PP (rank, size)): engine ranks are misplaced"
        )
    missing = sorted(r for r in dst_names.values() if r not in src)
    if missing:
        raise ValueError(
            f"co-resident refit: {len(missing)} engine tensors are not on this GPU's learner rank, e.g. {missing[:8]} "
            "(engine ranks not placed by engine_rank_gpus, or the layouts differ)"
        )
    return {name: src[resolved] for name, resolved in dst_names.items()}


def open_refit_handles(handles: dict[str, tuple]) -> dict[str, torch.Tensor]:
    """Engine side, once: map the learner's tensors into this process (views of the learner's memory, by name)."""
    return {name: rebuild(*args) for name, (rebuild, args) in handles.items()}


def bind_shared(
    src: dict[str, torch.Tensor], named_dst: Iterable[tuple[str, torch.Tensor]], leaf_ids: set[int]
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
            raise ValueError(f"co-resident refit: {name} has shape {tuple(s.shape)} (learner) vs {tuple(d.shape)} (engine)")
        # A strided view is never bound: TE GEMMs read weights as dense (TE 2.15 is silently wrong on strided ones).
        # Persistent buffers can change during backward, before the optimizer pause.
        if isinstance(d, torch.nn.Parameter) and s.dtype == d.dtype and s.is_contiguous() and id(d) in leaf_ids:
            d.data = s
            bound += 1
    return {"tensors_bound": bound, "tensors": len(dst)}


def _check_names(src: dict[str, torch.Tensor], dst: dict[str, torch.Tensor]) -> None:
    if dst.keys() != src.keys():
        missing, extra = sorted(set(dst) - set(src)), sorted(set(src) - set(dst))
        raise ValueError(
            "co-resident refit: engine and learner refit tensors differ by name (layouts must match): "
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
    return first.as_strided((len(weights), *first.shape), (first.numel(), *first.stride()))


def _reverse_expert_parameter_order(linear: torch.nn.Module, *, num_experts: int) -> bool:
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
    ordered = [(name, value) for name, value in parameters.items() if name not in weight_names]
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
    if checkpoint_config.load_optim and not checkpoint_config.finetune and checkpoint_config.load is not None:
        # Bridge and the distributed optimizer are optional outside this learner.
        from megatron.bridge.training.checkpointing import (
            _DIRECT_ITERATION_DIR_SENTINEL,
            _resolve_checkpoint_iteration,
        )
        from megatron.bridge.training.utils.checkpoint_utils import get_checkpoint_name
        from megatron.core import dist_checkpointing
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

        iteration, release = _resolve_checkpoint_iteration(checkpoint_config.load, checkpoint_config.ckpt_step)
        if not release and iteration != -1:
            checkpoint_dir = (
                checkpoint_config.load
                if iteration == _DIRECT_ITERATION_DIR_SENTINEL
                else get_checkpoint_name(checkpoint_config.load, iteration, release=False)
            )
            common_state = dist_checkpointing.load_common_state_dict(checkpoint_dir)
            if "optimizer" in common_state:
                metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common_state) or {}
                saved_format = metadata.get("distrib_optim_sharding_type")
                if saved_format not in DistributedOptimizer.checkpoint_fully_reshardable_formats:
                    raise ValueError(
                        "Shared expert weights require a model-space optimizer checkpoint; "
                        f"{checkpoint_dir!r} uses {saved_format!r}. Bucket-position optimizer state cannot "
                        "be restored after expert parameter reordering. Convert the checkpoint to "
                        "fully_reshardable format with its original parameter layout, or load weights "
                        "with a fresh optimizer."
                    )
    # Expert packing changes bucket positions; moment identity must follow model names.
    checkpoint_config.dist_ckpt_optim_fully_reshardable = True


def prepare_shared_expert_layout(models: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Pre-DDP hook: pack ordinary expert weights in inference order without changing their kernels."""
    # Megatron's expert modules are optional outside the co-resident learner.
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    linears = 0
    for model in models:
        for module in model.modules():
            if isinstance(module, TEGroupedMLP):
                for linear in (module.linear_fc1, module.linear_fc2):
                    linears += _reverse_expert_parameter_order(linear, num_experts=module.num_local_experts)
    print(f"[sm_partition] expert projections prepared before DDP: {linears}", flush=True)
    return models


def bind_shared_expert_stacks(core: torch.nn.Module, src: dict[str, torch.Tensor]) -> dict[str, float]:
    """Keep compatible BF16/FP16 expert stacks in learner storage through inference initialization.

    Both projections must already alias their corresponding source parameters and be packed in expert order.
    Other layouts retain Megatron's ordinary lazy packing and per-refit copies.
    """
    # The co-resident engine is the only consumer of optional Megatron inference modules.
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    layers = shared_bytes = 0
    for prefix, module in core.named_modules():
        if not isinstance(module, InferenceGroupedMLP):
            continue
        if module._concatenated_weights_built:
            raise RuntimeError("shared expert stacks must be bound before the first inference forward")
        stacks = {}
        for linear_name, buffer_name in module._EXPERT_WEIGHT_GROUPS:
            linear = getattr(module, linear_name)
            weights = []
            for index in range(module.num_local_experts):
                name = f"{prefix + '.' if prefix else ''}{linear_name}.weight{index}"
                weight = getattr(linear, f"weight{index}")
                source = src[name]
                if not isinstance(weight, torch.nn.Parameter) or not _aliases(weight, source):
                    break
                weights.append(source)
            if len(weights) != module.num_local_experts:
                break
            stack = _contiguous_expert_stack(weights)
            if stack is None:
                break
            stacks[buffer_name] = stack
        if len(stacks) != len(module._EXPERT_WEIGHT_GROUPS):
            continue
        for buffer_name, stack in stacks.items():
            module.register_buffer(buffer_name, stack, persistent=False)
            shared_bytes += stack.numel() * stack.element_size()
        module._uses_mxfp8_weights = False
        module._concatenated_weights_built = True
        layers += 1
    return {"shared_expert_layers": layers, "shared_expert_bytes": shared_bytes}


def _aliases(d: torch.Tensor, s: torch.Tensor) -> bool:
    return d.data_ptr() == s.data_ptr() and d.shape == s.shape and d.stride() == s.stride() and d.dtype == s.dtype


def copy_refit(src: dict[str, torch.Tensor], named_dst: Iterable[tuple[str, torch.Tensor]]) -> dict[str, float]:
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
            nbytes = s.numel() * s.element_size()
            if _aliases(d, s):
                shared += nbytes
                continue
            if nbytes <= _FOREACH_COPY_MAX_BYTES and s.is_contiguous() and d.is_contiguous():
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


# ---- per-version weight patches (measurement of the K > 1 single-copy path) ----


class WeightPatchStore:
    """Record a bounded history of sparse per-step weight patches on the GPU.

    Sparse per-step weight deltas of the learner's local refit tensors, newest last, at most ``keep`` versions, on
    the GPU.

    One shared weight copy holds only the newest version. An engine whose rows keep the version they started under
    needs, per version, what changed: patch v = {tensor: (flat int32 indices, values before the step)}, so version
    v - k = current with the last k patches undone. Recorded around the optimizer step into a snapshot buffer allocated
    once (one extra copy of the local weights while recording); ``stats`` gives the patch sizes per step and over
    k-step gaps.
    """

    def __init__(self, keep: int) -> None:
        self.keep = keep
        self.patches: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
        self._before: dict[str, torch.Tensor] = {}

    def before_write(self, named: Iterable[tuple[str, torch.Tensor]]) -> None:
        for n, t in named:
            if n not in self._before:
                self._before[n] = torch.empty_like(t)
            self._before[n].copy_(t.detach())

    def after_write(self, named: Iterable[tuple[str, torch.Tensor]]) -> None:
        patch = {}
        for n, t in named:
            old = self._before[n].reshape(-1)
            idx = (t.detach().reshape(-1) != old).nonzero().squeeze(1)
            if idx.numel():
                patch[n] = (idx.to(torch.int32), old[idx])
        self.patches = (self.patches + [patch])[-self.keep :]

    def stats(self, numel: int, element_size: int) -> dict[str, float]:
        """Summarize changed weights and sparse patch sizes across retained versions.

        Changed elements of the newest step and of the union over the last k steps (k = 1..keep), as a fraction of
        the local weights and in bytes of an (int32 index, value) patch.
        """
        out = {}
        for k in range(1, len(self.patches) + 1):
            names = set().union(*(p.keys() for p in self.patches[-k:]))
            nnz = sum(
                int(torch.unique(torch.cat([p[n][0] for p in self.patches[-k:] if n in p])).numel()) for n in names
            )
            out[f"delta_frac_k{k}"] = nnz / max(1, numel)
            out[f"delta_gib_k{k}"] = nnz * (4 + element_size) / 2**30
        return out
