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

import gc
import os
import traceback
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, Iterable, Optional

import requests
import torch
import torch.distributed as dist
import zmq
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTextToWaveform,
)

# Try to import nemo_automodel classes, fallback to None if not available
try:
    from nemo_automodel._transformers.auto_model import (
        NeMoAutoModelForCausalLM,
        NeMoAutoModelForImageTextToText,
        NeMoAutoModelForTextToWaveform,
    )

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    # nemo_automodel is not installed, classes will be None
    NeMoAutoModelForCausalLM = None  # type: ignore
    NeMoAutoModelForImageTextToText = None  # type: ignore
    NeMoAutoModelForTextToWaveform = None  # type: ignore
    NEMO_AUTOMODEL_AVAILABLE = False

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches

# an automodel factory for loading the huggingface models from correct class

AUTOMODEL_FACTORY: Dict[str, Any] = {
    "qwen2_5_vl": AutoModelForImageTextToText,
    "qwen2_vl": AutoModelForImageTextToText,
    "qwen2_5_omni": AutoModelForTextToWaveform,
    "llava": AutoModelForImageTextToText,
    "internvl": AutoModelForImageTextToText,
    "gemma3": AutoModelForImageTextToText,
    "smolvlm": AutoModelForImageTextToText,
    "mistral3": AutoModelForImageTextToText,
    "llama4": AutoModelForImageTextToText,
}

if NEMO_AUTOMODEL_AVAILABLE:
    AUTOMODEL_FACTORY = {
        "qwen2_5_vl": NeMoAutoModelForImageTextToText,
        "qwen2_vl": NeMoAutoModelForImageTextToText,
        "qwen2_5_omni": NeMoAutoModelForTextToWaveform,
        "llava": NeMoAutoModelForImageTextToText,
        "internvl": NeMoAutoModelForImageTextToText,
        "gemma3": NeMoAutoModelForImageTextToText,
        "smolvlm": NeMoAutoModelForImageTextToText,
        "mistral3": NeMoAutoModelForImageTextToText,
        "llama4": NeMoAutoModelForImageTextToText,
    }


class IPCProtocol(Enum):
    """IPC protocol constants for ZMQ weight streaming."""

    COMPLETE = "complete"
    ACK = "ack"


def resolve_model_class(model_name: str) -> Any:
    """Resolve the appropriate model class for a given model name."""
    if NEMO_AUTOMODEL_AVAILABLE:
        return AUTOMODEL_FACTORY.get(model_name.lower(), NeMoAutoModelForCausalLM)
    return AUTOMODEL_FACTORY.get(model_name.lower(), AutoModelForCausalLM)


def is_vllm_v1_engine_enabled() -> bool:
    """Check if vLLM V1 engine is enabled.

    Returns:
        bool: True if V1 engine is enabled, False otherwise (defaults to True if not set)
    """
    return os.environ.get("NRL_VLLM_USE_V1", "1") == "1"


def get_gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    """Return information about the GPU being used by this worker."""
    import torch

    # Get distributed training info
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get device info from CUDA
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # in MB

    # Try to get the real global device ID (not the local one)
    # In distributed training, each process only sees its assigned GPU as device 0
    local_device_id = device
    global_device_id = local_device_id

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    # Get a parameter from the model to verify CUDA device placement
    # This confirms tensors are actually on the appropriate device
    param_info = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and param.requires_grad:
                full_name = f"{module_name}.{param_name}"
                param_info[full_name] = {
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
                # Just grab one parameter for verification
                break
        if param_info:
            break

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        "parameter_sample": param_info,
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        },
    }


def configure_dynamo_cache() -> None:
    """Disable dynamo autotune_local_cache.

    Dynamo may fail at cached_autotune when there's already a cache with different order of node_bundles.
    Disable autotune_local_cache as a workaround.
    See https://github.com/pytorch/pytorch/issues/153791 for more details.
    """
    torch._inductor.config.autotune_local_cache = False


def get_runtime_env_for_policy_worker(policy_worker_name: str) -> dict[str, Any]:
    """Get runtime environment configuration for policy workers.

    Note: expandable_segments configuration is handled directly in the worker init methods
    to ensure proper GPU detection after CUDA initialization.
    """
    runtime_env = {
        **get_nsight_config_if_pattern_matches(policy_worker_name),
    }

    return runtime_env


def get_megatron_checkpoint_dir() -> str:
    """Gets the default megatron checkpoint directory for initial HF -> Mcore conversion.

    Megatron initial checkpoint should be saved to a path available on all nodes. The directory used will take this order of precendence:
    1. $NRL_MEGATRON_CHECKPOINT_DIR (if set)
    2. $HF_HOME/nemo_rl (if HF_HOME is set)
    3. ~/.cache/huggingface/nemo_rl

    HF_HOME is preferred since many users will also have that path mounted and it means one less directory
    to mount into your runtime environment.
    """
    nrl_checkpoint_dir = os.environ.get("NRL_MEGATRON_CHECKPOINT_DIR")
    if nrl_checkpoint_dir is not None and nrl_checkpoint_dir.strip():
        checkpoint_dir = nrl_checkpoint_dir
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is not None and hf_home.strip():
            checkpoint_dir = os.path.join(hf_home, "nemo_rl")
        else:
            checkpoint_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "nemo_rl"
            )
    print(f"Using default megatron checkpoint dir: {checkpoint_dir}")
    return checkpoint_dir


def get_handle_from_tensor(tensor: torch.Tensor) -> tuple[Any]:
    """Get IPC handle from a tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    # skip serializing the function for better refit performance
    return reduce_tensor(tensor.detach())[1:]


def calculate_aligned_size(size_bytes: int, alignment: int = 512) -> int:
    """Calculate aligned size for memory alignment.

    Args:
        size_bytes(int): Size in bytes to align
        alignment(int): Alignment boundary in bytes (default 512)

    Returns:
        Aligned size in bytes(int).
    """
    return int(((size_bytes + alignment - 1) // alignment) * alignment)


def stream_weights_via_ipc_zmq_impl(
    params_generator, buffer_size_bytes: int, zmq_socket, rank: int, worker_name: str
) -> None:
    """Shared implementation for streaming weights via IPC ZMQ with improved memory management.

    Uses ping-pong double buffering to enable overlapping communication while reusing buffers
    to reduce memory allocation overhead and improve stability.

    Args:
        params_generator: Generator yielding (name, tensor) pairs
        buffer_size_bytes: total size of buffer in bytes for batching parameters
        zmq_socket: ZMQ socket for communication
        rank: Worker rank for logging
        worker_name: Name of the worker for logging
    """
    # Divide total buffer size by 2 because we use two individual buffers (ping-pong) for overlapping communication.
    buffer_size_bytes = buffer_size_bytes // 2

    def send_buffer_group_overlap(buffer, param_names, used_bytes, await_recv) -> bool:
        """Send a group of parameters and return new pending_recv state."""
        # Synchronize before getting IPC handle to ensure data is ready
        torch.cuda.current_stream().synchronize()
        cuda_ipc_handle = get_handle_from_tensor(buffer)

        if await_recv:
            zmq_socket.recv()

        # Payload tuple: (cuda_ipc_handle, param_names, used_bytes)
        payload = (cuda_ipc_handle, param_names, used_bytes)
        zmq_socket.send_pyobj(payload)
        return True  # pending_recv = True

    def allocate_buffer(device):
        """Allocate a new aligned buffer with proper memory alignment."""
        aligned_size = calculate_aligned_size(buffer_size_bytes)
        return torch.empty(
            aligned_size,
            device=device,
            dtype=torch.uint8,
            requires_grad=False,
        )

    def pack_tensor(buffer, tensor, used_bytes) -> int:
        """Pack tensor into buffer and return new used_bytes."""
        tensor_bytes = tensor.nbytes
        buffer[used_bytes : used_bytes + tensor_bytes].data.copy_(
            tensor.data.view(-1).view(dtype=torch.uint8), non_blocking=True
        )
        return used_bytes + calculate_aligned_size(tensor_bytes)

    # Initialize ping-pong double buffering
    buffer_a: torch.Tensor | None = None
    buffer_b: torch.Tensor | None = None
    current_buffer: torch.Tensor | None = None

    used_bytes = 0
    param_names = []
    await_recv = False
    count_of_groups = 0

    try:
        for name, tensor in params_generator:
            # Initialize device and buffers on first tensor
            if buffer_a is None:
                buffer_a = allocate_buffer(tensor.device)
                buffer_b = allocate_buffer(tensor.device)
                current_buffer = buffer_a

            aligned_size = calculate_aligned_size(tensor.nbytes)
            assert aligned_size <= buffer_size_bytes, (
                f"Parameter {name} too large for buffer: {aligned_size} > {buffer_size_bytes}"
            )

            # Check if we need to send current buffer and switch to the other one
            if used_bytes + aligned_size > buffer_size_bytes:
                await_recv = send_buffer_group_overlap(
                    current_buffer, param_names, used_bytes, await_recv
                )
                count_of_groups += 1

                # Switch buffers for ping-pong double buffering
                current_buffer = buffer_b if current_buffer is buffer_a else buffer_a
                used_bytes, param_names = 0, []

            # Pack tensor into current buffer
            param_names.append(name)
            used_bytes = pack_tensor(current_buffer, tensor, used_bytes)

        # Send remaining tensors
        if param_names:
            await_recv = send_buffer_group_overlap(
                current_buffer, param_names, used_bytes, await_recv
            )
            count_of_groups += 1

        # Complete transmission
        if await_recv:
            zmq_socket.recv()

        # Final synchronization and completion signal
        torch.cuda.current_stream().synchronize()
        zmq_socket.send_pyobj(IPCProtocol.COMPLETE)
        zmq_socket.recv()

        if rank == 0:
            print(
                f"{worker_name}: Packed {count_of_groups} groups of tensors", flush=True
            )

    except zmq.Again:
        timeout_ms = zmq_socket.getsockopt(zmq.RCVTIMEO)
        raise TimeoutError(
            f"{worker_name} (rank {rank}): ZMQ communication timeout after {timeout_ms}ms in policy worker side. "
            f"The generation worker may be dead or unresponsive. "
            f"This typically indicates the generation worker has crashed or is not responding to weight streaming."
        ) from None
    except zmq.ZMQError as e:
        raise RuntimeError(
            f"{worker_name} (rank {rank}): ZMQ error during weight streaming: {e} (errno: {e.errno}). "
            f"Error details: {e.strerror}. "
            f"This may indicate network issues or the peer process has terminated unexpectedly.\n"
            f"{traceback.format_exc()}"
        ) from e

    finally:
        # Clean up buffers in finally block to ensure cleanup even on exceptions
        if buffer_a is not None:
            del buffer_a
        if buffer_b is not None:
            del buffer_b

        # Force garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()


def rebuild_cuda_tensor_from_ipc(
    cuda_ipc_handle: tuple, device_id: int
) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC handle."""
    func = rebuild_cuda_tensor
    args = cuda_ipc_handle[0]
    list_args = list(args)
    list_args[6] = device_id
    return func(*list_args)


def _derive_engine_gpu_offsets(engine_gpu_counts: list[int]) -> list[int]:
    """Cumulative-sum offsets for a dense engine layout."""
    offsets: list[int] = []
    cursor = 0
    for c in engine_gpu_counts:
        offsets.append(cursor)
        cursor += c
    return offsets


def connect_colocate_topology(
    *,
    engine_gpu_counts: list[int],
    engine_gpu_offsets: Optional[list[int]] = None,
    worker_state: dict,
    monkey_patch_fn=None,
) -> None:
    """Generalized colocate rollout-engine connect for FSDP and Megatron.

    Builds a Gloo gather subgroup for each engine's GPU rank range and stashes
    rank-only routing state into ``worker_state``:

    - ``worker_state["_ipc_gather_group"]``: ``ProcessGroup`` covering this
      trainer rank's engine, or ``None`` if the rank is a placeholder /
      not covered by any engine.
    - ``worker_state["_ipc_gather_src"]``: the source rank inside the gather
      group (the first GPU index of the covering engine), or ``None``.
    - ``worker_state["_ipc_engine_index"]``: index into the caller's engine
      list, or ``None``. The caller is responsible for resolving the actor
      handle / URL at call time so post-recover actor swaps are picked up.
    - ``worker_state["_ipc_layout_key"]``: cached topology signature so
      subsequent connects with the same layout are no-ops.

    All trainer ranks must enter this function collectively (each call to
    ``dist.new_group`` is collective). When the layout changes (e.g. a
    recovered engine resizes the topology) the cached subgroup is destroyed
    and rebuilt for the new layout.
    """
    if not engine_gpu_counts:
        raise ValueError("engine_gpu_counts must be non-empty")
    if engine_gpu_offsets is None:
        engine_gpu_offsets = _derive_engine_gpu_offsets(engine_gpu_counts)
    elif len(engine_gpu_offsets) != len(engine_gpu_counts):
        raise ValueError(
            "engine_gpu_offsets and engine_gpu_counts must have the same length, "
            f"got {len(engine_gpu_offsets)} vs {len(engine_gpu_counts)}"
        )

    layout_key = (tuple(engine_gpu_counts), tuple(engine_gpu_offsets))
    if worker_state.get("_ipc_layout_key") == layout_key:
        return

    if monkey_patch_fn is not None and not worker_state.get("_ipc_monkey_patched"):
        monkey_patch_fn()
        worker_state["_ipc_monkey_patched"] = True

    old_group = worker_state.get("_ipc_gather_group")
    if old_group is not None:
        try:
            dist.destroy_process_group(old_group)
        except Exception:
            # Some torch builds raise when the group has no peers; safe to
            # ignore — the new group below replaces it.
            pass

    my_rank = dist.get_rank()
    new_group = None
    new_src: Optional[int] = None
    new_engine_idx: Optional[int] = None
    for i, (offset, count) in enumerate(
        zip(engine_gpu_offsets, engine_gpu_counts, strict=True)
    ):
        group_ranks = list(range(offset, offset + count))
        grp = dist.new_group(ranks=group_ranks, backend="gloo")
        if my_rank in group_ranks:
            new_group = grp
            new_src = offset
            new_engine_idx = i

    worker_state["_ipc_gather_group"] = new_group
    worker_state["_ipc_gather_src"] = new_src
    worker_state["_ipc_engine_index"] = new_engine_idx
    worker_state["_ipc_layout_key"] = layout_key
    worker_state.setdefault("weight_version", 0)


def _flush_bucket(
    named_tensors,
    gather_src: int,
    gather_group,
    engine_url: str,
    weight_version: int,
    flattened_tensor_bucket_cls,
    multiprocessing_serializer_cls,
) -> None:
    """Flatten ``named_tensors`` per dtype, gather to ``gather_src``, and POST to the engine."""
    # Wait on any async DTensor redistributes.
    named_tensors = [
        (n, (t.wait() if hasattr(t, "wait") else t)) for n, t in named_tensors
    ]

    by_dtype: dict = {}
    for n, t in named_tensors:
        by_dtype.setdefault(t.dtype, []).append((n, t))

    serialized: list[str] = []
    for _dtype, tensors in by_dtype.items():
        bkt = flattened_tensor_bucket_cls(named_tensors=tensors)
        payload = {
            "flattened_tensor": bkt.get_flattened_tensor(),
            "metadata": bkt.get_metadata(),
        }
        serialized.append(
            multiprocessing_serializer_cls.serialize(payload, output_str=True)
        )

    my_rank = dist.get_rank()
    group_world = dist.get_world_size(gather_group)
    gathered = [None] * group_world if my_rank == gather_src else None
    dist.gather_object(
        serialized,
        object_gather_list=gathered,
        dst=gather_src,
        group=gather_group,
    )

    if my_rank != gather_src:
        return

    num_dtypes = len(gathered[0])
    assert num_dtypes > 0
    for i in range(num_dtypes):
        body = {
            "serialized_named_tensors": [g[i] for g in gathered],
            "load_format": "flattened_bucket",
            "flush_cache": False,
            "weight_version": str(weight_version),
        }
        response = requests.post(f"{engine_url}/update_weights_from_tensor", json=body)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        result = response.json()
        success = result.get("success", True)
        error_msg = result.get("error_message") or result.get(
            "message", "unknown error"
        )
        if not success:
            raise RuntimeError(
                f"Weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )


def stream_weights_via_http_impl(
    params_generator: Iterable[tuple[str, torch.Tensor]],
    rollout_engine_urls: Iterable[str],
    num_gpus_per_engine: int,
    rank: int,
    world_size: int,
    worker_name: str,
    buffer_size_bytes: int,
    worker_state: dict,
    *,
    engine_gpu_counts: Optional[list[int]] = None,
    engine_gpu_offsets: Optional[list[int]] = None,
) -> None:
    """Stream FSDP weights to colocated SGLang engines via CUDA IPC over HTTP.

    Args:
        params_generator: Iterable yielding ``(name, tensor)`` pairs to stream.
            Caller is responsible for any pre-processing (LoRA merge, HF
            adaptation, dtype cast).
        rollout_engine_urls: ``http://host:port`` base URLs of each engine's
            ``node_rank=0`` SGLang HTTP server. One entry per engine, in TP
            rank-range order: engine ``i`` owns global ranks
            ``[i * num_gpus_per_engine, (i + 1) * num_gpus_per_engine)``.
        num_gpus_per_engine: TP size per SGLang engine.
        rank: Global FSDP rank.
        world_size: Global FSDP world size.
        worker_name: Human label for logs.
        buffer_size_bytes: Max bucket size in bytes.
        worker_state: Mutable dict on the worker used to cache topology and
            weight version across refits.
    """
    from nemo_rl.models.policy.torch_reductions_utils import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
        monkey_patch_torch_reductions,
    )

    rollout_engine_urls = list(rollout_engine_urls)

    if engine_gpu_counts is None:
        engine_gpu_counts = [num_gpus_per_engine] * len(rollout_engine_urls)
    if engine_gpu_offsets is None:
        engine_gpu_offsets = _derive_engine_gpu_offsets(engine_gpu_counts)

    connect_colocate_topology(
        engine_gpu_counts=engine_gpu_counts,
        engine_gpu_offsets=engine_gpu_offsets,
        worker_state=worker_state,
        monkey_patch_fn=monkey_patch_torch_reductions,
    )

    worker_state["weight_version"] = worker_state.get("weight_version", 0) + 1
    weight_version = worker_state["weight_version"]
    gather_src = worker_state["_ipc_gather_src"]
    gather_group = worker_state["_ipc_gather_group"]
    engine_idx = worker_state["_ipc_engine_index"]
    engine_url = (
        rollout_engine_urls[engine_idx] if engine_idx is not None else None
    )

    if gather_group is None:
        # Placeholder rank not covered by any engine: drain quietly.
        return

    try:
        bucket: list = []
        bucket_size = 0
        for name, param in params_generator:
            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= buffer_size_bytes:
                _flush_bucket(
                    bucket,
                    gather_src=gather_src,
                    gather_group=gather_group,
                    engine_url=engine_url,
                    weight_version=weight_version,
                    flattened_tensor_bucket_cls=FlattenedTensorBucket,
                    multiprocessing_serializer_cls=MultiprocessingSerializer,
                )
                bucket = []
                bucket_size = 0

            param = param.cuda()
            bucket.append((name, param))
            bucket_size += param_size

        if bucket:
            _flush_bucket(
                bucket,
                gather_src=gather_src,
                gather_group=gather_group,
                engine_url=engine_url,
                weight_version=weight_version,
                flattened_tensor_bucket_cls=FlattenedTensorBucket,
                multiprocessing_serializer_cls=MultiprocessingSerializer,
            )

        if dist.get_rank() == gather_src:
            # Mirror SGLangGenerationWorker.flush_cache: the endpoint returns
            # non-200 while requests are still pending, so retry up to 60s.
            import time

            for _ in range(60):
                try:
                    response = requests.get(f"{engine_url}/flush_cache")
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Timeout while flushing cache at {engine_url}.")

    except Exception as e:
        print(
            f"{worker_name} (rank {rank}): Error during HTTP weight streaming: {e}.\n"
            f"{traceback.format_exc()}"
        )
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _check_weight_sync_results(results: list) -> None:
    from collections.abc import Mapping

    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = (
                result.get("error_message") or result.get("error") or "unknown error"
            )
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"SGLang weight sync failed on rollout engine: {error_msg}. "
                "Check SGLang version compatibility."
            )


def send_hf_buckets_via_ipc_actor_impl(
    *,
    bucket_iterator: Iterable[list[tuple[str, torch.Tensor]]],
    rollout_engines: list,
    worker_state: dict,
    weight_version: Optional[int] = None,
) -> None:
    """Send finalized HF tensor buckets to colocated SGLang engines via Ray IPC.

    Per bucket: group by dtype, serialize a ``FlattenedTensorBucket`` per
    dtype, ``dist.gather_object`` to the gather source rank, then on the
    source rank call ``ipc_engine.update_weights_from_tensor.remote(...)``
    once per dtype, **block on ``ray.get(refs)`` per chunk**, validate
    engine return values, then drop the trainer-side
    ``flattened_tensor`` references before moving on.

    The trainer-side topology (``_ipc_gather_group`` / ``_ipc_gather_src`` /
    ``_ipc_engine_index``) must already have been set up by
    :func:`connect_colocate_topology`. Placeholder ranks (no covering engine)
    return immediately — they must not call ``gather_object``. Non-source
    trainer ranks participate only in the gather collective; they don't
    issue Ray RPCs and don't ``ray.get``.

    Returns ``None``. Raises ``RuntimeError`` if any chunk fails on the
    engine side.
    """
    import ray

    from nemo_rl.models.policy.torch_reductions_utils import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
    )

    gather_group = worker_state.get("_ipc_gather_group")
    gather_src = worker_state.get("_ipc_gather_src")
    engine_idx = worker_state.get("_ipc_engine_index")

    if gather_group is None or gather_src is None or engine_idx is None:
        # Placeholder rank: must not participate in the per-engine gather.
        return None

    if weight_version is None:
        worker_state["weight_version"] = worker_state.get("weight_version", 0) + 1
        weight_version = worker_state["weight_version"]

    ipc_engine = rollout_engines[engine_idx]
    my_rank = dist.get_rank()

    try:
        for bucket in bucket_iterator:
            if not bucket:
                continue

            # No async-collective ``.wait()`` here — Megatron's AutoBridge
            # yields plain ``torch.Tensor``, no DTensor wrapping.

            if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
                by_dtype: dict = {"dtype": list(bucket)}
            else:
                by_dtype = {}
                for name, tensor in bucket:
                    by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            serialized: list[str] = []
            long_lived_tensors: list[dict] = []
            for _dtype, named_tensors in by_dtype.items():
                bkt = FlattenedTensorBucket(named_tensors=named_tensors)
                payload = {
                    "flattened_tensor": bkt.get_flattened_tensor(),
                    "metadata": bkt.get_metadata(),
                }
                long_lived_tensors.append(payload)
                serialized.append(
                    MultiprocessingSerializer.serialize(payload, output_str=True)
                )

            group_world = dist.get_world_size(gather_group)
            gathered = [None] * group_world if my_rank == gather_src else None
            dist.gather_object(
                serialized,
                object_gather_list=gathered,
                dst=gather_src,
                group=gather_group,
            )

            # Only the gather-source rank queues the engine RPC; non-source
            # ranks have nothing to await but must still hold their
            # long_lived_tensors past the engine's copy. Mirroring miles, we
            # use a single uniform "ray.get → check → del" tail across all
            # ranks: refs is empty on non-source so ray.get is a no-op there.
            refs: list = []
            if my_rank == gather_src:
                num_dtypes = len(gathered[0])
                for i in range(num_dtypes):
                    refs.append(
                        ipc_engine.update_weights_from_tensor.remote(
                            serialized_named_tensors=[g[i] for g in gathered],
                            load_format="flattened_bucket",
                            weight_version=str(weight_version),
                        )
                    )

            # Block until the engine has copied this chunk through the IPC
            # handles, surface any per-chunk failure, then release the
            # trainer-side GPU tensors before the next chunk allocates.
            results = ray.get(refs)
            _check_weight_sync_results(results)
            del long_lived_tensors, refs, results
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return None


def find_free_port() -> int:
    """Return a currently-free TCP port on the local node."""
    import socket

    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def init_process_group(
    backend: "str | dist.Backend | None" = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: "Optional[dist.Store]" = None,
    group_name: Optional[str] = None,
    pg_options: Any = None,
) -> "torch.distributed.ProcessGroup":
    """Create a side-by-side ``ProcessGroup`` without touching the default world.

    ``torch.distributed.init_process_group`` initializes the *default* world
    process group. Once the Megatron trainer has stood up its own world during
    Policy construction, calling it again to talk to SGLang either errors with
    "trying to initialize the default process group twice" or — depending on
    torch version — silently hangs in rendezvous against a peer that has
    already finished its own custom-group setup.

    Same approach as SGLang's ``sglang.srt.utils.common.init_custom_process_group``:
    replay the public API's wiring (rendezvous → ``PrefixStore`` →
    ``_new_process_group_helper``) but skip the "set as default PG" step, so
    multiple independent groups can coexist in the same process.

    Only one of ``init_method`` and ``store`` may be set; otherwise the
    rendezvous source is ambiguous.
    """
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend) if backend else Backend("undefined")
    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(
            init_method, rank, world_size, timeout=timeout
        )
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        # PrefixStore so multiple co-tenant groups don't trample each other's keys.
        store = PrefixStore(group_name or "", store)

    # ``pg_options`` was renamed to ``backend_options`` in PyTorch 2.6:
    #   https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # Use numeric tuple compare — string compare ``"2.10" >= "2.6"`` returns
    # False because ``"1"`` sorts before ``"6"`` lexicographically.
    _torch_mm = tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    )
    pg_options_kw = "backend_options" if _torch_mm >= (2, 6) else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_kw: pg_options},
        timeout=timeout,
    )

    # Map identity ranks so collective ops can resolve member ranks for ``pg``.
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


def connect_rollout_engines_from_distributed(
    *,
    group_name: str,
    rollout_engines: list,
    engine_gpu_counts: list[int],
) -> "torch.distributed.ProcessGroup":
    """Set up the SGLang NCCL weight-update group with trainer rank 0 as rank 0.

    Only trainer rank 0 broadcasts because the AutoBridge path restores
    full HF weights, not per-PP slices.

    The caller (a trainer) must invoke this only on rank 0; other ranks must
    not call it.
    """
    import ray

    master_address = ray._private.services.get_node_ip_address()
    master_port = find_free_port()
    world_size = 1 + sum(engine_gpu_counts)

    refs = []
    rank_cursor = 1
    for engine, gpu_count in zip(rollout_engines, engine_gpu_counts, strict=True):
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                rank_cursor,
                world_size,
                group_name,
                "nccl",
            )
        )
        rank_cursor += gpu_count

    group = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return group


def disconnect_rollout_engines_from_distributed(
    *,
    group_name: str,
    model_update_group: "torch.distributed.ProcessGroup",
    rollout_engines: list,
) -> None:
    """Tear down trainer-side and engine-side NCCL state for ``group_name``."""
    import ray

    refs = [
        engine.destroy_weights_update_group.remote(group_name)
        for engine in rollout_engines
    ]
    try:
        dist.destroy_process_group(model_update_group)
    except Exception:
        pass
    try:
        ray.get(refs)
    except Exception:
        pass


def get_sglang_quantization_cfg(policy_generation: Any) -> dict:
    """Read the active SGLang quantization block from the generation handle.

    Returns an empty dict when no quantization config is set, so callers can
    treat the result as a stable mapping without ``None`` checks.
    """
    return dict(
        policy_generation.sglang_cfg["sglang_cfg"].get("quantization") or {}
    )


def fetch_updatable_engines_with_recover(policy_generation: Any) -> tuple:
    """Run the design-mandated weight-update prelude.

    1. If ``sglang_cfg.use_fault_tolerance`` is enabled, call
       ``rollout_manager.recover_updatable_engines`` which internally pauses
       health monitoring, restarts dead engines, and runs
       release/resume_memory_occupation on every recovered node-0 engine.
    2. Read the current updatable-engine state via
       ``get_updatable_engines_and_lock``.

    Both calls are idempotent — recover is a no-op when no engines have died.
    """
    use_ft = bool(
        policy_generation.sglang_cfg["sglang_cfg"].get("use_fault_tolerance", False)
    )
    if use_ft:
        policy_generation.recover_updatable_engines()
    return policy_generation.get_updatable_engines_and_lock()


def broadcast_hf_buckets_via_distributed_impl(
    *,
    bucket_iterator: Iterable[list[tuple[str, torch.Tensor]]],
    rollout_engines: list,
    rollout_engine_lock,
    group_name: str,
    model_update_group: "torch.distributed.ProcessGroup",
    weight_version: int,
) -> None:
    """Broadcast finalized HF tensor buckets to SGLang via NCCL (rank 0 only).

    Per-bucket protocol: trainer rank 0 sends per-tensor metadata to every
    engine via Ray (``update_weights_from_distributed``), then issues one
    ``dist.broadcast`` per tensor over the NCCL group, then waits for the Ray
    refs to confirm engines finished loading the bucket.

    The rollout-engine lock wraps each bucket's broadcast so concurrent SGLang
    NCCL operations (e.g. health-check pings) cannot collide with the
    weight-update broadcast.
    """
    import time as _time

    import ray

    bucket_idx = 0
    for bucket in bucket_iterator:
        if not bucket:
            continue

        bucket_idx += 1
        # No async-collective ``.wait()`` here — AutoBridge yields plain
        # ``torch.Tensor`` for the Megatron path (no DTensor wrapping).

        names = [name for name, _ in bucket]
        dtypes = [tensor.dtype for _, tensor in bucket]
        shapes = [tensor.shape for _, tensor in bucket]
        devices = [str(tensor.device) for _, tensor in bucket]
        total_bytes = sum(t.numel() * t.element_size() for _, t in bucket)
        print(
            f"[BCAST bucket={bucket_idx}] n={len(bucket)} bytes={total_bytes} "
            f"first={names[0]} last={names[-1]} devs={set(devices)} dtypes={set(dtypes)}",
            flush=True,
        )

        print(f"[BCAST bucket={bucket_idx}] acquiring rollout_engine_lock...", flush=True)
        while not ray.get(rollout_engine_lock.acquire.remote()):
            _time.sleep(0.1)
        print(f"[BCAST bucket={bucket_idx}] lock acquired", flush=True)
        try:
            print(
                f"[BCAST bucket={bucket_idx}] kicking engine.update_weights_from_distributed.remote() RPCs...",
                flush=True,
            )
            refs = [
                engine.update_weights_from_distributed.remote(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=group_name,
                    weight_version=str(weight_version),
                )
                for engine in rollout_engines
            ]
            print(
                f"[BCAST bucket={bucket_idx}] issuing {len(bucket)} dist.broadcast async_op calls...",
                flush=True,
            )
            handles = []
            for i, (_, tensor) in enumerate(bucket):
                handles.append(
                    dist.broadcast(
                        tensor.data, 0, group=model_update_group, async_op=True
                    )
                )
            print(
                f"[BCAST bucket={bucket_idx}] all {len(handles)} broadcasts launched; waiting...",
                flush=True,
            )
            for i, handle in enumerate(handles):
                handle.wait()
                if i == 0 or (i + 1) == len(handles):
                    print(
                        f"[BCAST bucket={bucket_idx}] handle.wait() done {i + 1}/{len(handles)}",
                        flush=True,
                    )
            print(
                f"[BCAST bucket={bucket_idx}] all broadcasts complete; ray.get(refs)...",
                flush=True,
            )
            ray.get(refs)
            print(
                f"[BCAST bucket={bucket_idx}] engine RPCs returned (engine done loading)",
                flush=True,
            )
        finally:
            ray.get(rollout_engine_lock.release.remote())
            print(f"[BCAST bucket={bucket_idx}] lock released", flush=True)
