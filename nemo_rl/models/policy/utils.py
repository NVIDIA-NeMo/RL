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
import warnings
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, cast

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

    # Side-effect import: installs the resolver hook that routes FP8-native
    # Mistral 3.5 configs to Mistral3FP8VLM. Without it, HF's stock FP8Linear
    # path runs and produces 0-d weight_scale_inv params that FSDP2 rejects.
    try:
        import nemo_automodel.components.models.mistral3_vlm  # noqa: F401
    except ImportError:
        pass

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    # nemo_automodel is not installed, classes will be None
    NeMoAutoModelForCausalLM = None  # type: ignore
    NeMoAutoModelForImageTextToText = None  # type: ignore
    NeMoAutoModelForTextToWaveform = None  # type: ignore
    NEMO_AUTOMODEL_AVAILABLE = False

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches

# Plain Hugging Face classes remain separate from the NeMo AutoModel wrappers so
# callers that manage distribution can request them when NeMo AutoModel is installed.
# Add an entry here whenever a model's architecture isn't loadable via
# AutoModelForCausalLM (e.g. VLMs using ForConditionalGeneration /
# ForImageTextToText). Unlike AUTOMODEL_FACTORY below, this dict is also read on
# the ``use_nemo_automodel=False`` path (DTensor V1), where no NeMo custom impl
# intercepts from_pretrained -- so a model that has a custom NeMo automodel impl
# still needs an entry here when its parent AutoModel class is not
# AutoModelForCausalLM. Check MODEL_ARCH_MAPPING in the NeMo automodel registry
# to see which architectures have custom impls:
# https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/registry.py#L32-L146
HF_AUTOMODEL_FACTORY: Dict[str, Any] = {
    "qwen2_5_vl": AutoModelForImageTextToText,
    "qwen2_vl": AutoModelForImageTextToText,
    "qwen2_5_omni": AutoModelForTextToWaveform,
    "qwen3_5": AutoModelForImageTextToText,
    "llava": AutoModelForImageTextToText,
    "internvl": AutoModelForImageTextToText,
    "gemma3": AutoModelForImageTextToText,
    "gemma4": AutoModelForImageTextToText,
    "gemma4_unified": AutoModelForImageTextToText,
    "smolvlm": AutoModelForImageTextToText,
    "mistral3": AutoModelForImageTextToText,
    "llama4": AutoModelForImageTextToText,
}

AUTOMODEL_FACTORY: Dict[str, Any] = HF_AUTOMODEL_FACTORY
DENSE_TEACHER_IPC_FLAT_LAYOUT = "flat_valid_prefix_v1"

if NEMO_AUTOMODEL_AVAILABLE:
    AUTOMODEL_FACTORY = {
        # NeMo wrappers — keep in sync with the vanilla HF dict above.
        # See comment above for when to add entries.
        "qwen2_5_vl": NeMoAutoModelForImageTextToText,
        "qwen2_vl": NeMoAutoModelForImageTextToText,
        "qwen2_5_omni": NeMoAutoModelForTextToWaveform,
        "qwen3_5": NeMoAutoModelForImageTextToText,
        "llava": NeMoAutoModelForImageTextToText,
        "internvl": NeMoAutoModelForImageTextToText,
        "gemma3": NeMoAutoModelForImageTextToText,
        "gemma4": NeMoAutoModelForImageTextToText,
        "gemma4_unified": NeMoAutoModelForImageTextToText,
        "smolvlm": NeMoAutoModelForImageTextToText,
        "mistral3": NeMoAutoModelForImageTextToText,
        "llama4": NeMoAutoModelForImageTextToText,
    }


class IPCProtocol(Enum):
    """IPC protocol constants for ZMQ weight streaming."""

    COMPLETE = "complete"
    ACK = "ack"


# TODO: Replace this hard-coded map with a generic plugin-registration
# hook on ``Policy`` (e.g. a ``worker_cls_overrides`` registry populated by
# ``nemo_rl.modelopt`` on import) so core has no knowledge of ModelOpt-specific
# worker classes.
POLICY_WORKER_OVERRIDES = {
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker.MegatronQuantPolicyWorker",
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker.DTensorQuantPolicyWorker",
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker_v2.DTensorQuantPolicyWorkerV2",
}


def resolve_policy_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized policy worker FQN if ``quant_cfg`` is set, else ``default_cls``.

    Safe to call even when ModelOpt is not installed — returns ``default_cls``
    unchanged whenever ``quant_cfg`` is ``None``, so the core policy path stays
    import-free of ModelOpt.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return POLICY_WORKER_OVERRIDES.get(default_cls, default_cls)


def resolve_model_class(
    model_name: str,
    *,
    use_nemo_automodel: bool = True,
) -> Any:
    """Resolve the model class for a model type.

    Args:
        model_name: Model type to resolve.
        use_nemo_automodel: Whether to prefer NeMo AutoModel wrappers when they
            are available.
    """
    if use_nemo_automodel and NEMO_AUTOMODEL_AVAILABLE:
        return AUTOMODEL_FACTORY.get(model_name.lower(), NeMoAutoModelForCausalLM)
    return HF_AUTOMODEL_FACTORY.get(model_name.lower(), AutoModelForCausalLM)


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


def ensure_teacher_ipc_buffer(
    storage: Optional[torch.Tensor],
    handle: Optional[tuple[Any, ...]],
    num_microbatches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    """Lazy-alloc / grow ``[N_mb, B, T, V]`` teacher-logits IPC storage.

    Returns the (possibly reallocated) ``(storage, handle)``. Reallocates and
    re-exports the IPC handle whenever any dim of the requested shape exceeds
    the current storage, or dtype/device changed; otherwise the existing
    storage and cached handle are returned unchanged.
    """
    needs_realloc = (
        storage is None
        or storage.shape[0] < num_microbatches
        or storage.shape[1] < batch_size
        or storage.shape[2] < seq_len
        or storage.shape[3] < vocab_size
        or storage.dtype != dtype
        or storage.device != device
    )
    if needs_realloc:
        storage = torch.empty(
            (num_microbatches, batch_size, seq_len, vocab_size),
            dtype=dtype,
            device=device,
        )
        handle = get_handle_from_tensor(storage)
    assert storage is not None and handle is not None
    return storage, handle


def ensure_teacher_ipc_row_buffer(
    storage: Optional[torch.Tensor],
    total_rows: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Lazy-allocate a compact ``[sum(B_mb), T, V]`` teacher IPC slab.

    Only the row capacity may be reused at a larger size. Sequence/vocabulary
    geometry, dtype, and device must match exactly so every exported bin view is
    contiguous and preserves the existing ``[1, B_mb, T, V]`` IPC contract.
    """
    if total_rows <= 0 or seq_len <= 0 or vocab_size <= 0:
        raise ValueError(
            "Dense teacher IPC dimensions must be positive, got "
            f"rows={total_rows}, seq_len={seq_len}, vocab_size={vocab_size}."
        )
    needs_realloc = (
        storage is None
        or storage.ndim != 3
        or storage.shape[0] < total_rows
        or tuple(storage.shape[1:]) != (seq_len, vocab_size)
        or storage.dtype != dtype
        or storage.device != device
    )
    if needs_realloc:
        storage = torch.empty(
            (total_rows, seq_len, vocab_size),
            dtype=dtype,
            device=device,
        )
    return storage


def partition_teacher_ipc_row_buffer(
    storage: torch.Tensor, batch_sizes: Iterable[int]
) -> list[tuple[int, torch.Tensor]]:
    """Return compact row offsets and legacy-compatible 4-D bin views.

    Each view has shape ``[1, B_mb, T, V]`` and addresses a disjoint contiguous
    range of the shared row slab. Exporting the views through CUDA IPC therefore
    keeps existing consumers' ``buf_idx=0`` / bin-local sample indexing and
    zero-copy fast path while avoiding ``N_mb * max(B_mb)`` padding rows.
    """
    if storage.ndim != 3:
        raise ValueError(
            "Dense teacher IPC row storage must be three-dimensional, got "
            f"shape={tuple(storage.shape)}."
        )
    normalized_batch_sizes = [int(batch_size) for batch_size in batch_sizes]
    if any(batch_size <= 0 for batch_size in normalized_batch_sizes):
        raise ValueError(
            "Dense teacher IPC microbatch cardinalities must be positive, got "
            f"{normalized_batch_sizes}."
        )
    total_rows = sum(normalized_batch_sizes)
    if total_rows > storage.shape[0]:
        raise ValueError(
            "Dense teacher IPC row partitions exceed storage capacity: "
            f"rows={total_rows}, capacity={storage.shape[0]}."
        )

    partitions: list[tuple[int, torch.Tensor]] = []
    row_offset = 0
    for batch_size in normalized_batch_sizes:
        next_offset = row_offset + batch_size
        partitions.append((row_offset, storage[row_offset:next_offset].unsqueeze(0)))
        row_offset = next_offset
    return partitions


def extract_teacher_ipc_valid_lengths(
    data: Mapping[str, Any],
    expected_batch_size: int,
    *,
    full_seq_len: int,
    compact_padding: bool,
) -> list[int]:
    """Return the logical valid prefix length for every teacher row.

    Packed :class:`FullLogitsPostProcessor` uses ``input_lengths`` as its
    authoritative raw boundary. It either zero-fills the remainder of a logical
    dense row or streams only the valid prefix; the compact consumer reconstructs
    the omitted suffix as zero in both cases. The unpacked path does not promise
    zero logits at padding positions, so it deliberately retains the complete
    logical width.
    """
    if expected_batch_size <= 0 or full_seq_len <= 0:
        raise ValueError(
            "Dense teacher IPC row geometry must be positive, got "
            f"rows={expected_batch_size}, full_seq_len={full_seq_len}."
        )
    if not compact_padding:
        return [full_seq_len] * expected_batch_size

    values = data.get("input_lengths")
    if values is None:
        raise ValueError(
            "Packed dense teacher IPC requires input_lengths on every logical "
            "microbatch row."
        )
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(
                "input_lengths must be a one-dimensional tensor, got shape "
                f"{tuple(values.shape)}."
            )
        if torch.is_floating_point(values) or torch.is_complex(values):
            raise TypeError(
                "input_lengths must use an integer dtype for dense teacher IPC, "
                f"got {values.dtype}."
            )
        normalized = [int(value) for value in values.detach().cpu().tolist()]
    elif isinstance(values, (list, tuple)):
        normalized = [int(value) for value in values]
    else:
        raise TypeError(
            "input_lengths must be a one-dimensional tensor, list, or tuple, "
            f"got {type(values).__name__}."
        )
    if len(normalized) != expected_batch_size:
        raise ValueError(
            "input_lengths cardinality does not match the logical microbatch: "
            f"lengths={len(normalized)}, rows={expected_batch_size}."
        )
    invalid = [value for value in normalized if value < 0 or value > full_seq_len]
    if invalid:
        raise ValueError(
            "Dense teacher IPC input_lengths must lie within the logical "
            f"sequence width [0, {full_seq_len}], got {invalid}."
        )
    return normalized


def localize_teacher_ipc_valid_lengths(
    valid_lengths: Iterable[int],
    *,
    full_seq_len: int,
    cp_rank: int,
    cp_size: int,
) -> list[int]:
    """Intersect global valid prefixes with one teacher CP rectangle."""
    if cp_size <= 0 or cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(
            f"Invalid context-parallel coordinates: rank={cp_rank}, size={cp_size}."
        )
    if full_seq_len <= 0 or full_seq_len % cp_size != 0:
        raise ValueError(
            f"Logical dense teacher width {full_seq_len} must be positive and "
            f"divisible by context-parallel size {cp_size}."
        )
    local_seq_len = full_seq_len // cp_size
    global_seq_start = cp_rank * local_seq_len
    global_seq_end = global_seq_start + local_seq_len
    localized: list[int] = []
    for value in valid_lengths:
        valid_seq_len = int(value)
        if valid_seq_len < 0 or valid_seq_len > full_seq_len:
            raise ValueError(
                "Dense teacher IPC valid length must lie within the logical "
                f"sequence width [0, {full_seq_len}], got {valid_seq_len}."
            )
        localized.append(max(0, min(valid_seq_len, global_seq_end) - global_seq_start))
    return localized


def can_reuse_teacher_ipc_token_buffer(
    storage: Optional[torch.Tensor],
    *,
    total_tokens: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return whether ``storage`` can hold one compact teacher IPC payload."""
    if total_tokens < 0 or vocab_size <= 0:
        raise ValueError(
            "Dense teacher IPC token geometry is invalid, got "
            f"tokens={total_tokens}, vocab_size={vocab_size}."
        )
    return (
        storage is not None
        and storage.ndim == 2
        and storage.shape[0] >= total_tokens
        and storage.shape[1] == vocab_size
        and storage.dtype == dtype
        and storage.device == device
    )


def ensure_teacher_ipc_token_buffer(
    storage: Optional[torch.Tensor],
    *,
    total_tokens: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Lazy-allocate a flat ``[sum(valid_local_T), V]`` teacher IPC slab."""
    if not can_reuse_teacher_ipc_token_buffer(
        storage,
        total_tokens=total_tokens,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device,
    ):
        storage = torch.empty(
            (total_tokens, vocab_size),
            dtype=dtype,
            device=device,
        )
    assert storage is not None
    return storage


def partition_teacher_ipc_token_buffer(
    storage: torch.Tensor,
    valid_lengths_by_microbatch: Iterable[Iterable[int]],
) -> list[list[tuple[int, torch.Tensor]]]:
    """Return token offsets and allocation-free row views into a flat slab."""
    if storage.ndim != 2:
        raise ValueError(
            "Dense teacher IPC token storage must be two-dimensional, got "
            f"shape={tuple(storage.shape)}."
        )
    normalized = [
        [int(valid_length) for valid_length in microbatch_lengths]
        for microbatch_lengths in valid_lengths_by_microbatch
    ]
    if any(valid_length < 0 for lengths in normalized for valid_length in lengths):
        raise ValueError(
            "Dense teacher IPC local valid lengths must be non-negative, got "
            f"{normalized}."
        )
    total_tokens = sum(sum(lengths) for lengths in normalized)
    if total_tokens > storage.shape[0]:
        raise ValueError(
            "Dense teacher IPC token partitions exceed storage capacity: "
            f"tokens={total_tokens}, capacity={storage.shape[0]}."
        )

    partitions: list[list[tuple[int, torch.Tensor]]] = []
    token_offset = 0
    for microbatch_lengths in normalized:
        microbatch_partitions: list[tuple[int, torch.Tensor]] = []
        for valid_length in microbatch_lengths:
            next_offset = token_offset + valid_length
            microbatch_partitions.append(
                (token_offset, storage[token_offset:next_offset])
            )
            token_offset = next_offset
        partitions.append(microbatch_partitions)
    return partitions


def validate_compact_teacher_ipc_handle(
    handle: Mapping[str, Any],
) -> Optional[tuple[int, int, int, int]]:
    """Validate compact dense-IPC metadata and return its physical geometry.

    Legacy rectangular handles have no ``ipc_layout`` field and return
    ``None``.  Compact handles return ``(token_offset, stored_seq_len,
    used_tokens, local_vocab_size)``.  Unknown or partial schemas fail loudly
    so a malformed handle cannot silently read another row's logits.
    """
    layout = handle.get("ipc_layout")
    if layout is None:
        return None
    if layout != DENSE_TEACHER_IPC_FLAT_LAYOUT:
        raise ValueError(f"Unsupported dense teacher IPC layout {layout!r}.")

    required_fields = {
        "actual_shape",
        "dtype",
        "full_seq_len",
        "global_seq_start",
        "payload_ipc",
        "storage_capacity_tokens",
        "storage_shape",
        "storage_token_offset",
        "storage_used_tokens",
        "stored_seq_len",
        "valid_seq_len",
    }
    missing_fields = required_fields - handle.keys()
    if missing_fields:
        raise ValueError(
            "Compact dense teacher IPC handle is missing metadata "
            f"{sorted(missing_fields)}."
        )

    actual_shape = handle["actual_shape"]
    storage_shape = handle["storage_shape"]
    if (
        not isinstance(actual_shape, (list, tuple, torch.Size))
        or len(actual_shape) != 2
    ):
        raise ValueError(
            "Compact dense teacher IPC actual_shape must be two-dimensional, "
            f"got {actual_shape!r}."
        )
    if (
        not isinstance(storage_shape, (list, tuple, torch.Size))
        or len(storage_shape) != 2
    ):
        raise ValueError(
            "Compact dense teacher IPC storage_shape must be two-dimensional, "
            f"got {storage_shape!r}."
        )

    local_seq_len, local_vocab_size = (int(value) for value in actual_shape)
    capacity_tokens, storage_vocab_size = (int(value) for value in storage_shape)
    full_seq_len = int(handle["full_seq_len"])
    global_seq_start = int(handle["global_seq_start"])
    valid_seq_len = int(handle["valid_seq_len"])
    stored_seq_len = int(handle["stored_seq_len"])
    token_offset = int(handle["storage_token_offset"])
    used_tokens = int(handle["storage_used_tokens"])
    declared_capacity = int(handle["storage_capacity_tokens"])

    if (
        local_seq_len <= 0
        or local_vocab_size <= 0
        or full_seq_len <= 0
        or global_seq_start < 0
        or global_seq_start + local_seq_len > full_seq_len
    ):
        raise ValueError(
            "Compact dense teacher IPC has invalid logical sequence/vocabulary "
            f"geometry: actual_shape={tuple(actual_shape)}, global_seq_start="
            f"{global_seq_start}, full_seq_len={full_seq_len}."
        )
    if storage_vocab_size != local_vocab_size or capacity_tokens < 0:
        raise ValueError(
            "Compact dense teacher IPC storage shape disagrees with the logical "
            f"vocabulary shard: storage_shape={tuple(storage_shape)}, "
            f"actual_shape={tuple(actual_shape)}."
        )
    if declared_capacity != capacity_tokens:
        raise ValueError(
            "Compact dense teacher IPC storage capacity metadata disagrees with "
            f"storage_shape: declared={declared_capacity}, shape={capacity_tokens}."
        )
    if valid_seq_len < 0 or valid_seq_len > full_seq_len:
        raise ValueError(
            "Compact dense teacher IPC valid_seq_len lies outside the logical "
            f"sequence: valid={valid_seq_len}, full={full_seq_len}."
        )
    expected_stored_seq_len = max(
        0,
        min(valid_seq_len, global_seq_start + local_seq_len) - global_seq_start,
    )
    if stored_seq_len != expected_stored_seq_len:
        raise ValueError(
            "Compact dense teacher IPC stored_seq_len disagrees with the valid "
            f"CP overlap: stored={stored_seq_len}, expected="
            f"{expected_stored_seq_len}."
        )
    if (
        used_tokens < 0
        or used_tokens > capacity_tokens
        or token_offset < 0
        or token_offset > used_tokens
        or token_offset + stored_seq_len > used_tokens
    ):
        raise ValueError(
            "Compact dense teacher IPC token range is outside storage bounds: "
            f"offset={token_offset}, stored={stored_seq_len}, used={used_tokens}, "
            f"capacity={capacity_tokens}."
        )
    if stored_seq_len > 0 and handle["payload_ipc"] is None:
        raise ValueError(
            "Compact dense teacher IPC omitted payload_ipc for a non-empty row."
        )
    return token_offset, stored_seq_len, used_tokens, local_vocab_size


def extract_batch_item_ids(
    data: Mapping[str, Any],
    expected_batch_size: int,
    *,
    required: bool,
) -> list[int] | None:
    """Read occurrence identities from one logical microbatch.

    Packed teacher IPC must be routed by occurrence identity rather than by a
    bin-local position. The helper accepts the tensor or list representation
    used by :class:`BatchedDataDict`, validates cardinality, and deliberately
    does not synthesize identities when the controller did not provide them.
    """
    values = data.get("batch_item_id")
    if values is None:
        if required:
            raise ValueError(
                "Packed dense teacher IPC requires batch_item_id on every "
                "logical microbatch row."
            )
        return None
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(
                "batch_item_id must be a one-dimensional tensor, got shape "
                f"{tuple(values.shape)}."
            )
        normalized = [int(value) for value in values.detach().cpu().tolist()]
    else:
        normalized = [int(value) for value in values]
    if len(normalized) != expected_batch_size:
        raise ValueError(
            "batch_item_id cardinality does not match the logical microbatch: "
            f"ids={len(normalized)}, rows={expected_batch_size}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"batch_item_id values must be unique within a microbatch, got {normalized}."
        )
    return normalized


def get_dense_ipc_sequence_layout(
    data: Mapping[str, Any], *, cp_rank: int, cp_size: int
) -> tuple[int, int, int]:
    """Return logical dense-IPC ``(full, local, start)`` sequence geometry.

    Packed backend schedulers operate on a physical bin width, which can be
    much larger than the original per-sample token rectangle restored by the
    full-logit postprocessor. IPC metadata must describe that logical rectangle
    rather than the packed scheduler width so teacher masks and alignment axes
    agree with reconstruction.
    """
    if cp_size <= 0 or cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(
            f"Invalid context-parallel coordinates: rank={cp_rank}, size={cp_size}."
        )
    input_ids = data.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        raise ValueError(
            "Dense teacher IPC requires input_ids with a logical sequence axis."
        )
    full_seq_len = int(input_ids.shape[1])
    if full_seq_len % cp_size != 0:
        raise ValueError(
            f"Logical dense teacher width {full_seq_len} must be divisible by "
            f"context-parallel size {cp_size}."
        )
    local_seq_len = full_seq_len // cp_size
    return full_seq_len, local_seq_len, cp_rank * local_seq_len


def _validate_dense_ipc_shard_coverage(
    batch_item_id: int, teacher_shards: list[dict[str, Any]]
) -> None:
    """Validate that one sample has a complete, non-overlapping TP/CP grid."""
    required_fields = {
        "actual_shape",
        "cp_rank",
        "cp_size",
        "full_seq_len",
        "full_vocab_size",
        "global_seq_start",
        "tp_rank",
        "tp_size",
        "vocab_end_index",
        "vocab_start_index",
    }
    first = teacher_shards[0]
    missing_fields = required_fields - first.keys()
    if missing_fields:
        raise ValueError(
            f"Dense teacher IPC shard for batch_item_id={batch_item_id} is "
            f"missing coverage metadata {sorted(missing_fields)}."
        )

    tp_size = int(first["tp_size"])
    cp_size = int(first["cp_size"])
    full_seq_len = int(first["full_seq_len"])
    full_vocab_size = int(first["full_vocab_size"])
    if tp_size <= 0 or cp_size <= 0:
        raise ValueError(
            f"Dense teacher IPC for batch_item_id={batch_item_id} has invalid "
            f"parallel sizes tp_size={tp_size}, cp_size={cp_size}."
        )
    if full_seq_len <= 0 or full_vocab_size <= 0:
        raise ValueError(
            f"Dense teacher IPC for batch_item_id={batch_item_id} has invalid "
            f"logical shape ({full_seq_len}, {full_vocab_size})."
        )

    coordinates: set[tuple[int, int]] = set()
    sequence_ranges: dict[int, tuple[int, int]] = {}
    vocab_ranges: dict[int, tuple[int, int]] = {}
    layouts: set[Optional[str]] = set()
    valid_seq_lengths: set[int] = set()
    compact_geometry_by_cp: dict[int, tuple[int, int, int, int]] = {}
    for shard in teacher_shards:
        missing_fields = required_fields - shard.keys()
        if missing_fields:
            raise ValueError(
                f"Dense teacher IPC shard for batch_item_id={batch_item_id} is "
                f"missing coverage metadata {sorted(missing_fields)}."
            )
        shard_tp_size = int(shard["tp_size"])
        shard_cp_size = int(shard["cp_size"])
        shard_full_seq_len = int(shard["full_seq_len"])
        shard_full_vocab_size = int(shard["full_vocab_size"])
        if (
            shard_tp_size != tp_size
            or shard_cp_size != cp_size
            or shard_full_seq_len != full_seq_len
            or shard_full_vocab_size != full_vocab_size
        ):
            raise ValueError(
                "Dense teacher IPC shards disagree on topology or logical shape "
                f"for batch_item_id={batch_item_id}."
            )

        tp_rank = int(shard["tp_rank"])
        cp_rank = int(shard["cp_rank"])
        if tp_rank < 0 or tp_rank >= tp_size or cp_rank < 0 or cp_rank >= cp_size:
            raise ValueError(
                f"Dense teacher IPC shard for batch_item_id={batch_item_id} has "
                f"out-of-range coordinate (tp={tp_rank}, cp={cp_rank}) for "
                f"topology ({tp_size}, {cp_size})."
            )
        coordinate = (tp_rank, cp_rank)
        if coordinate in coordinates:
            raise ValueError(
                f"Dense teacher IPC has duplicate shard coordinate {coordinate} "
                f"for batch_item_id={batch_item_id}."
            )
        coordinates.add(coordinate)
        layouts.add(shard.get("ipc_layout"))

        actual_shape = shard["actual_shape"]
        if (
            not isinstance(actual_shape, (list, tuple, torch.Size))
            or len(actual_shape) != 2
        ):
            raise ValueError(
                f"Dense teacher IPC shard for batch_item_id={batch_item_id} must "
                f"have two-dimensional actual_shape, got {actual_shape!r}."
            )
        local_seq_len, local_vocab_size = (int(value) for value in actual_shape)
        seq_start = int(shard["global_seq_start"])
        seq_end = seq_start + local_seq_len
        vocab_start = int(shard["vocab_start_index"])
        vocab_end = int(shard["vocab_end_index"])
        if (
            local_seq_len <= 0
            or local_vocab_size <= 0
            or seq_start < 0
            or seq_end > full_seq_len
            or vocab_start < 0
            or vocab_end > full_vocab_size
            or vocab_end - vocab_start != local_vocab_size
        ):
            raise ValueError(
                f"Dense teacher IPC shard {coordinate} for "
                f"batch_item_id={batch_item_id} has invalid rectangle "
                f"seq=[{seq_start}, {seq_end}), vocab=[{vocab_start}, {vocab_end}), "
                f"actual_shape={tuple(actual_shape)}, full_shape="
                f"({full_seq_len}, {full_vocab_size})."
            )

        sequence_range = (seq_start, seq_end)
        prior_sequence_range = sequence_ranges.setdefault(cp_rank, sequence_range)
        if prior_sequence_range != sequence_range:
            raise ValueError(
                f"Dense teacher IPC shards for cp_rank={cp_rank} disagree on "
                f"sequence range for batch_item_id={batch_item_id}."
            )
        vocab_range = (vocab_start, vocab_end)
        prior_vocab_range = vocab_ranges.setdefault(tp_rank, vocab_range)
        if prior_vocab_range != vocab_range:
            raise ValueError(
                f"Dense teacher IPC shards for tp_rank={tp_rank} disagree on "
                f"vocabulary range for batch_item_id={batch_item_id}."
            )

        compact_geometry = validate_compact_teacher_ipc_handle(shard)
        if compact_geometry is not None:
            token_offset, stored_seq_len, used_tokens, _ = compact_geometry
            valid_seq_len = int(shard["valid_seq_len"])
            valid_seq_lengths.add(valid_seq_len)
            storage_capacity_tokens = int(shard["storage_capacity_tokens"])
            geometry = (
                token_offset,
                stored_seq_len,
                used_tokens,
                storage_capacity_tokens,
            )
            prior_geometry = compact_geometry_by_cp.setdefault(cp_rank, geometry)
            if prior_geometry != geometry:
                raise ValueError(
                    "Compact dense teacher IPC TP shards disagree on physical "
                    f"storage geometry for cp_rank={cp_rank}, "
                    f"batch_item_id={batch_item_id}."
                )

    if len(layouts) != 1:
        raise ValueError(
            "Dense teacher IPC shards mix incompatible storage layouts for "
            f"batch_item_id={batch_item_id}: {sorted(map(str, layouts))}."
        )
    if layouts != {None} and len(valid_seq_lengths) != 1:
        raise ValueError(
            "Compact dense teacher IPC shards disagree on valid_seq_len for "
            f"batch_item_id={batch_item_id}: {sorted(valid_seq_lengths)}."
        )

    expected_coordinates = {
        (tp_rank, cp_rank) for tp_rank in range(tp_size) for cp_rank in range(cp_size)
    }
    if coordinates != expected_coordinates:
        raise ValueError(
            "Dense teacher IPC has incomplete TP/CP shard coverage for "
            f"batch_item_id={batch_item_id}: missing="
            f"{sorted(expected_coordinates - coordinates)}, unexpected="
            f"{sorted(coordinates - expected_coordinates)}."
        )

    sequence_cursor = 0
    for cp_rank in range(cp_size):
        seq_start, seq_end = sequence_ranges[cp_rank]
        if seq_start != sequence_cursor:
            raise ValueError(
                "Dense teacher IPC sequence shards contain a gap or overlap for "
                f"batch_item_id={batch_item_id} before cp_rank={cp_rank}: "
                f"expected start {sequence_cursor}, got {seq_start}."
            )
        sequence_cursor = seq_end
    if sequence_cursor != full_seq_len:
        raise ValueError(
            "Dense teacher IPC sequence shards do not cover the logical axis for "
            f"batch_item_id={batch_item_id}: covered={sequence_cursor}, "
            f"expected={full_seq_len}."
        )

    vocab_cursor = 0
    for tp_rank in range(tp_size):
        vocab_start, vocab_end = vocab_ranges[tp_rank]
        if vocab_start != vocab_cursor:
            raise ValueError(
                "Dense teacher IPC vocabulary shards contain a gap or overlap for "
                f"batch_item_id={batch_item_id} before tp_rank={tp_rank}: "
                f"expected start {vocab_cursor}, got {vocab_start}."
            )
        vocab_cursor = vocab_end
    if vocab_cursor != full_vocab_size:
        raise ValueError(
            "Dense teacher IPC vocabulary shards do not cover the logical axis for "
            f"batch_item_id={batch_item_id}: covered={vocab_cursor}, "
            f"expected={full_vocab_size}."
        )


def aggregate_per_sample_handles(
    worker_results: list[dict[str, Any]],
    canonical_batch_item_ids: Optional[Iterable[int]] = None,
) -> list[dict[str, Any]]:
    """Flatten teacher per-sample IPC handles into a global-batch-ordered list.

    Each worker returns ``{"dp_rank": int, "per_sample_handles": list}`` where
    the handle list is in local sample order; the several workers sharing a
    ``dp_rank`` are TP/CP/PP replicas that each contribute one shard per sample.
    Concatenating samples in ``sorted(dp_rank)`` order reproduces the original
    global sample order (rank 0 holds the first ``gbs/dp`` samples, rank 1 the
    next, ...), so the result is a length-``gbs`` list independent of the
    teacher's DP degree. Element ``i`` is ``{"teacher_shards": [shard, ...]}``
    holding all TP×CP shards of global sample ``i``.

    Workers that contribute no handles are skipped. Under pipeline parallelism
    only the LAST stage holds logits, so every earlier stage returns an empty
    list; without this filter the per-``dp_rank`` length check below would fire
    on those empty contributions.

    When ``canonical_batch_item_ids`` is supplied, positional ordering is not
    used: every record must carry ``batch_item_id``, TP/CP replicas on one DP
    rank must report identical identity sets, and every sample must have a
    complete, non-overlapping TP×CP shard grid covering its full logical
    sequence/vocabulary rectangle. The aggregate is returned in exact canonical
    order. Omitting the IDs retains the legacy unpacked behavior.
    """
    nonempty_results = [
        worker_result
        for worker_result in worker_results
        if worker_result["per_sample_handles"]
    ]
    assert nonempty_results, (
        "No teacher worker returned full-logits IPC handles. Under pipeline "
        "parallelism only the last stage produces them, so this means the "
        "last-stage results were not collected."
    )
    if canonical_batch_item_ids is not None:
        canonical_ids = tuple(int(value) for value in canonical_batch_item_ids)
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError(
                f"canonical_batch_item_ids must be unique, got {canonical_ids}."
            )
        shards_by_item: dict[int, list[dict[str, Any]]] = {}
        owner_dp_by_item: dict[int, int] = {}
        identity_sets_by_dp: dict[int, list[set[int]]] = {}
        for worker_result in nonempty_results:
            dp_rank = int(worker_result["dp_rank"])
            seen_by_worker: set[int] = set()
            for handle in worker_result["per_sample_handles"]:
                if "batch_item_id" not in handle:
                    raise ValueError(
                        "Identity-aware dense teacher IPC requires every handle "
                        "record to carry batch_item_id."
                    )
                batch_item_id = int(handle["batch_item_id"])
                if batch_item_id in seen_by_worker:
                    raise ValueError(
                        "A teacher worker returned duplicate handles for "
                        f"batch_item_id={batch_item_id}."
                    )
                seen_by_worker.add(batch_item_id)
                owner_dp = owner_dp_by_item.setdefault(batch_item_id, dp_rank)
                if owner_dp != dp_rank:
                    raise ValueError(
                        f"batch_item_id={batch_item_id} was returned by multiple "
                        f"data-parallel ranks ({owner_dp} and {dp_rank})."
                    )
                shards_by_item.setdefault(batch_item_id, []).append(handle)
            identity_sets_by_dp.setdefault(dp_rank, []).append(seen_by_worker)
        for dp_rank, worker_identity_sets in identity_sets_by_dp.items():
            expected = worker_identity_sets[0]
            if any(
                identity_set != expected for identity_set in worker_identity_sets[1:]
            ):
                raise ValueError(
                    f"Teacher shard replicas for dp_rank={dp_rank} returned "
                    f"inconsistent batch_item_id sets: {worker_identity_sets}."
                )
        actual_ids = set(shards_by_item)
        canonical_id_set = set(canonical_ids)
        if actual_ids != canonical_id_set:
            missing = sorted(canonical_id_set - actual_ids)
            unexpected = sorted(actual_ids - canonical_id_set)
            raise ValueError(
                "Dense teacher IPC identities do not cover the canonical batch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        for batch_item_id in canonical_ids:
            _validate_dense_ipc_shard_coverage(
                batch_item_id, shards_by_item[batch_item_id]
            )
        return [
            {
                "batch_item_id": batch_item_id,
                "teacher_shards": shards_by_item[batch_item_id],
            }
            for batch_item_id in canonical_ids
        ]

    handles_by_dp_rank: dict[int, list[list[dict[str, Any]]]] = {}
    for worker_result in nonempty_results:
        per_sample_handles = worker_result["per_sample_handles"]
        dp_rank = worker_result["dp_rank"]
        handles_by_dp_rank.setdefault(dp_rank, []).append(per_sample_handles)
    aggregated: list[dict[str, Any]] = []
    for dp_rank in sorted(handles_by_dp_rank.keys()):
        worker_handles_in_dp = handles_by_dp_rank[dp_rank]
        num_samples = len(worker_handles_in_dp[0])
        for worker_handles in worker_handles_in_dp:
            assert len(worker_handles) == num_samples, (
                f"dp={dp_rank}: per_sample_handles length mismatch "
                f"{[len(h) for h in worker_handles_in_dp]}"
            )
        for sample_idx in range(num_samples):
            aggregated.append(
                {
                    "teacher_shards": [
                        worker_handles[sample_idx]
                        for worker_handles in worker_handles_in_dp
                    ]
                }
            )
    return aggregated


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
        # reshape(-1) (not view(-1)): the params iterator may yield
        # non-contiguous tensors and view would raise on incompatible stride.
        buffer[used_bytes : used_bytes + tensor_bytes].data.copy_(
            tensor.data.reshape(-1).view(dtype=torch.uint8), non_blocking=True
        )
        return used_bytes + calculate_aligned_size(tensor_bytes)

    # Initialize ping-pong double buffering
    buffer_a: torch.Tensor | None = None
    buffer_b: torch.Tensor | None = None
    current_buffer: torch.Tensor | None = None

    def release_staging_buffers() -> None:
        """Release acyclic IPC buffers without scanning the worker object graph."""
        nonlocal buffer_a, buffer_b, current_buffer

        had_buffers = buffer_a is not None or buffer_b is not None
        current_buffer = None
        buffer_a = None
        buffer_b = None
        if had_buffers:
            torch.cuda.empty_cache()

    used_bytes = 0
    param_names = []
    await_recv = False
    count_of_groups = 0

    try:
        for name, tensor in params_generator:
            # Initialize device and buffers on first tensor
            if buffer_a is None:
                buffer_device = tensor.device
                if buffer_device.type == "cpu" and torch.cuda.is_available():
                    buffer_device = torch.device("cuda", torch.cuda.current_device())
                buffer_a = allocate_buffer(buffer_device)
                buffer_b = allocate_buffer(buffer_device)
                current_buffer = buffer_a

            aligned_size = calculate_aligned_size(tensor.nbytes)

            # A parameter larger than a single staging buffer cannot be packed
            # at all. Ship it on its own in a buffer sized to fit rather than
            # failing the refit. The staging buffers are sized from *free
            # memory* (NRL_REFIT_BUFFER_MEMORY_RATIO, default 0.3, halved again
            # for ping-pong) with no floor at the largest parameter, so a big
            # embedding can exceed one: DeepSeek-V3's model.embed_tokens.weight
            # is 1.73 GiB against a 1.65 GiB buffer. This mirrors the HTTP
            # streaming path, which already gives an oversized parameter a
            # bucket of its own instead of raising.
            if aligned_size > buffer_size_bytes:
                if param_names:
                    await_recv = send_buffer_group_overlap(
                        current_buffer, param_names, used_bytes, await_recv
                    )
                    count_of_groups += 1
                    current_buffer = (
                        buffer_b if current_buffer is buffer_a else buffer_a
                    )
                    used_bytes, param_names = 0, []

                oversized_buffer = torch.empty(
                    aligned_size,
                    device=current_buffer.device,
                    dtype=torch.uint8,
                    requires_grad=False,
                )
                try:
                    packed_bytes = pack_tensor(oversized_buffer, tensor, 0)
                    send_buffer_group_overlap(
                        oversized_buffer, [name], packed_bytes, await_recv
                    )
                    count_of_groups += 1
                    # Unlike the ping-pong pair, this buffer is not kept alive
                    # across the next send, so its ACK must be consumed here
                    # before it is freed.
                    zmq_socket.recv()
                    await_recv = False
                finally:
                    del oversized_buffer
                    torch.cuda.empty_cache()
                continue

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

        # The receiver synchronizes and drops every IPC view before ACKing a
        # group, so the final data ACK is the staging buffers' safe lifetime
        # boundary. Reclaim them before asking the receiver to run its final
        # post-load conversion, which can otherwise retain both large buffers
        # for the whole conversion and amplify a single-rank tail.
        torch.cuda.current_stream().synchronize()
        release_staging_buffers()
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
        # Tensor references are acyclic and deterministic; a full gc.collect()
        # scans the entire model object graph and can become a multi-second
        # rank straggler without releasing anything that refcounting cannot.
        release_staging_buffers()


def rebuild_cuda_tensor_from_ipc(
    cuda_ipc_handle: tuple, device_id: int
) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC handle."""
    func = rebuild_cuda_tensor
    args = cuda_ipc_handle[0]
    list_args = list(args)
    list_args[6] = device_id
    return func(*list_args)


# ---------------------------------------------------------------------------
# SGLang weight-update plumbing (colocate IPC gather + disaggregate broadcast)
# ---------------------------------------------------------------------------
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
) -> None:
    """Generalized colocate rollout-engine connect for FSDP and Megatron.

    Builds a Gloo gather subgroup for each engine's GPU rank range and stashes
    rank-only routing state into ``worker_state``:

    - ``worker_state["_ipc_gather_group"]``: ``ProcessGroup`` covering this
      trainer rank's engine, or ``None`` if the rank is a placeholder /
      not covered by any engine.
    - ``worker_state["_ipc_gather_groups"]``: all subgroup handles created for
      this layout, retained so a rebuild can destroy every live group.
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

    old_groups = worker_state.get("_ipc_gather_groups")
    if old_groups is None:
        old_group = worker_state.get("_ipc_gather_group")
        old_groups = [old_group] if old_group is not None else []
    for old_group in old_groups:
        if not isinstance(old_group, dist.ProcessGroup):
            continue
        try:
            dist.destroy_process_group(old_group)
        except Exception:
            # Some torch builds raise when the group has no peers; safe to
            # ignore — the new group below replaces it.
            pass

    my_rank = dist.get_rank()
    new_group = None
    new_groups = []
    new_src: Optional[int] = None
    new_engine_idx: Optional[int] = None
    for i, (offset, count) in enumerate(
        zip(engine_gpu_offsets, engine_gpu_counts, strict=True)
    ):
        group_ranks = list(range(offset, offset + count))
        grp = dist.new_group(ranks=group_ranks, backend="gloo")
        new_groups.append(grp)
        if my_rank in group_ranks:
            new_group = grp
            new_src = offset
            new_engine_idx = i

    worker_state["_ipc_gather_group"] = new_group
    worker_state["_ipc_gather_groups"] = new_groups
    worker_state["_ipc_gather_src"] = new_src
    worker_state["_ipc_engine_index"] = new_engine_idx
    worker_state["_ipc_layout_key"] = layout_key
    worker_state.setdefault("weight_version", 0)


def _check_weight_sync_results(results: list) -> None:
    from collections.abc import Mapping

    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = (
                result.get("error_message")
                or result.get("error")
                or result.get("message")
                or "unknown error"
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


def iter_named_tensor_buckets(
    params_generator: Iterable[tuple[str, torch.Tensor]],
    buffer_size_bytes: int,
) -> "Iterable[list[tuple[str, torch.Tensor]]]":
    """Group ``(name, tensor)`` pairs into buckets of at most ``buffer_size_bytes``.

    Waits on async DTensor redistributes (``.wait()``) before sizing, so the
    yielded tensors are always materialized and safe to serialize.
    """
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")

    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_size = 0
    for name, tensor in params_generator:
        if hasattr(tensor, "wait"):
            tensor = tensor.wait()
        tensor_size = tensor.numel() * tensor.element_size()
        if bucket and bucket_size + tensor_size > buffer_size_bytes:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.append((name, tensor))
        bucket_size += tensor_size

    if bucket:
        yield bucket


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
    engine return values, synchronize all trainer ranks, then drop the
    trainer-side ``flattened_tensor`` references before moving on.

    The trainer-side topology (``_ipc_gather_group`` / ``_ipc_gather_src`` /
    ``_ipc_engine_index``) must already have been set up by
    :func:`connect_colocate_topology`. Placeholder ranks (no covering engine)
    return immediately — they must not call ``gather_object``. Non-source
    trainer ranks participate in the gather and completion broadcast; they
    don't issue Ray RPCs and don't ``ray.get``.

    Returns ``None``. Raises ``RuntimeError`` if any chunk fails on the
    engine side.
    """
    import ray

    from nemo_rl.models.generation.sglang.utils.train_utils import (
        FlattenedTensorBucket,
        MultiprocessingSerializer,
    )

    gather_group = worker_state.get("_ipc_gather_group")
    gather_src = worker_state.get("_ipc_gather_src")
    engine_idx = worker_state.get("_ipc_engine_index")

    if gather_group is None or gather_src is None or engine_idx is None:
        # Placeholder rank: skips the gather, but must still drain the iterator.
        # It lazily drives export_hf_weights, whose PP broadcast and TP gather
        # are collective over the whole trainer world.
        for _bucket in bucket_iterator:
            del _bucket
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
                serialized_payload = MultiprocessingSerializer.serialize(
                    payload, output_str=True
                )
                if not isinstance(serialized_payload, str):
                    raise TypeError("SGLang IPC serialization did not return text")
                serialized.append(serialized_payload)

            group_world = dist.get_world_size(gather_group)
            gathered = cast(
                Optional[list[Optional[list[str]]]],
                [None] * group_world if my_rank == gather_src else None,
            )
            dist.gather_object(
                serialized,
                object_gather_list=gathered,
                dst=gather_src,
                group=gather_group,
            )

            refs: list = []
            if my_rank == gather_src:
                if gathered is None or any(payload is None for payload in gathered):
                    raise RuntimeError("SGLang IPC gather returned incomplete payloads")
                gathered_payloads = cast(list[list[str]], gathered)
                num_dtypes = len(gathered_payloads[0])
                for i in range(num_dtypes):
                    refs.append(
                        ipc_engine.update_weights_from_tensor.remote(
                            serialized_named_tensors=[
                                payload[i] for payload in gathered_payloads
                            ],
                            load_format="flattened_bucket",
                            weight_version=str(weight_version),
                        )
                    )

            # The serialized IPC handles gathered on the source may point at
            # flattened tensors owned by non-source trainer ranks. Keep every
            # rank's tensors alive until the source finishes the engine RPCs.
            sync_error: Optional[str] = None
            source_exc: Optional[BaseException] = None
            if my_rank == gather_src:
                try:
                    results = ray.get(refs)
                    _check_weight_sync_results(results)
                except BaseException as exc:
                    source_exc = exc
                    sync_error = repr(exc)

            sync_state = [sync_error]
            dist.broadcast_object_list(sync_state, src=gather_src, group=gather_group)
            del long_lived_tensors, refs

            if source_exc is not None:
                raise source_exc
            if sync_state[0] is not None:
                raise RuntimeError(
                    f"SGLang IPC weight update failed on gather src rank "
                    f"{gather_src}: {sync_state[0]}"
                )
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return None


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
        GroupName,
        PrefixStore,
        _get_default_group,
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
        assert init_method is not None
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        # PrefixStore so multiple co-tenant groups don't trample each other's keys.
        store = PrefixStore(group_name or "", store)

    group_name = GroupName(group_name or "")

    # ``pg_options`` was renamed to ``backend_options`` in PyTorch 2.6:
    #   https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # Use numeric tuple compare — string compare ``"2.10" >= "2.6"`` returns
    # False because ``"1"`` sorts before ``"6"`` lexicographically.
    _torch_mm = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    pg_options_kw = "backend_options" if _torch_mm >= (2, 6) else "pg_options"

    # Disable the ncclCommSplit path (see docstring). Safe to mutate here:
    # nothing else reads ``bound_device_id`` during group construction, and
    # refit setup does not create process groups from other threads.
    default_pg = _get_default_group() if dist.is_initialized() else None
    saved_bound_device_id = getattr(default_pg, "bound_device_id", None)
    if saved_bound_device_id is not None:
        default_pg.bound_device_id = None
    try:
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
    finally:
        if saved_bound_device_id is not None:
            default_pg.bound_device_id = saved_bound_device_id

    if not isinstance(pg, dist.ProcessGroup):
        raise RuntimeError("Torch returned an invalid custom process group")

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

    from nemo_rl.distributed.virtual_cluster import _get_free_port_local

    master_address = ray._private.services.get_node_ip_address()
    master_port = _get_free_port_local()
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
    from ray.exceptions import RayError

    refs = [
        engine.destroy_weights_update_group.remote(group_name)
        for engine in rollout_engines
    ]
    try:
        dist.destroy_process_group(model_update_group)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Failed to destroy the trainer side of SGLang weight-update group "
            f"{group_name!r}: {exc!r}",
            stacklevel=2,
        )
    try:
        ray.get(refs)
    except (RayError, RuntimeError) as exc:
        warnings.warn(
            f"Failed to destroy the engine side of SGLang weight-update group "
            f"{group_name!r}: {exc!r}",
            stacklevel=2,
        )


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

    for bucket in bucket_iterator:
        if not bucket:
            continue

        # No async-collective ``.wait()`` here — AutoBridge yields plain
        # ``torch.Tensor`` for the Megatron path (no DTensor wrapping).

        names = [name for name, _ in bucket]
        dtypes = [tensor.dtype for _, tensor in bucket]
        shapes = [tensor.shape for _, tensor in bucket]

        lock_timeout_s = 300
        lock_deadline = _time.monotonic() + lock_timeout_s
        while not ray.get(rollout_engine_lock.acquire.remote()):
            if _time.monotonic() >= lock_deadline:
                raise TimeoutError(
                    "Timed out after 300 seconds waiting for the SGLang "
                    "rollout-engine lock."
                )
            _time.sleep(0.1)
        try:
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
            # broadcast needs contiguous buffers; AutoBridge yields views. Held
            # in a list because with async_op=True they must outlive wait().
            send_buffers = [
                tensor.data if tensor.data.is_contiguous() else tensor.data.contiguous()
                for _, tensor in bucket
            ]
            handles = [
                dist.broadcast(buffer, 0, group=model_update_group, async_op=True)
                for buffer in send_buffers
            ]
            for handle in handles:
                handle.wait()
            del send_buffers
            ray.get(refs)
        finally:
            ray.get(rollout_engine_lock.release.remote())
