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

"""Runtime Mamba zero-KL patches for current Megatron-LM (public ssm_prefill/decode API)."""

import importlib
from types import ModuleType
from typing import Any, Callable, Mapping, Optional

import torch

_MAMBA_MIXER_MODULE: Optional[ModuleType] = None
_ORIGINAL_GET_MAMBA_VERSION: Optional[Callable[..., Any]] = None
_MAMBA_UTILS_MODULE: Optional[ModuleType] = None
_BID_MODULE: Optional[ModuleType] = None
_ORIGINAL_BIK_DECODE_SCAN: Optional[Callable[..., torch.Tensor]] = None
_ORIGINAL_BIK_DECODE_SEED: Optional[Callable[..., None]] = None

_MAMBA_MODEL_NAME_MARKERS = (
    "nemotron-nano",
    "nemotron_nano",
    "nemotron-3-nano",
    "nemotron_3_nano",
    "nemotron-3-super",
    "nemotron_3_super",
    "falconh1",
    "falcon-h1",
    "falcon_h1",
)


def mamba_mixer_module() -> ModuleType:
    global _MAMBA_MIXER_MODULE
    if _MAMBA_MIXER_MODULE is None:
        _MAMBA_MIXER_MODULE = importlib.import_module("megatron.core.ssm.mamba_mixer")
    return _MAMBA_MIXER_MODULE


def set_mamba_mixer_module(module: ModuleType) -> None:
    global _MAMBA_MIXER_MODULE
    _MAMBA_MIXER_MODULE = module


def clear_mamba_mixer_module() -> None:
    global _MAMBA_MIXER_MODULE
    _MAMBA_MIXER_MODULE = None


def train_kernel_buffered_scan(
    buf: Any,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    z: Optional[torch.Tensor],
    batch_indices: Optional[torch.Tensor],
    ssm_state: torch.Tensor,
    *,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    flatten_output: bool,
) -> torch.Tensor:
    """Run batched buffered decode through ``mamba_chunk_scan_combined`` (training kernel)."""
    module = mamba_mixer_module()
    rearrange = module.rearrange
    batch_size, sequence_length, nheads, head_dim = x.shape
    if sequence_length != 1:
        raise NotImplementedError(
            "BIK Mamba decode supports one token per request and no speculative decoding"
        )

    slots = (
        batch_indices.to(torch.long)
        if batch_indices is not None
        else torch.arange(batch_size, device=x.device, dtype=torch.long)
    )
    safe_slots = slots.clamp(min=0)
    slots_list = slots.tolist()
    counts_list = buf._bik_chunk_count.index_select(0, safe_slots).tolist()
    zero_state_list = buf._bik_state_is_zero.index_select(0, safe_slots).tolist()

    active_rows = [index for index, slot in enumerate(slots_list) if slot >= 0]
    y = torch.zeros(batch_size, 1, nheads, head_dim, device=x.device, dtype=x.dtype)
    if not active_rows:
        return rearrange(y, "b s h p -> b s (h p)") if flatten_output else y

    slot_indices = torch.tensor(
        [slots_list[index] for index in active_rows], device=x.device, dtype=torch.long
    )
    count_indices = torch.tensor(
        [counts_list[index] for index in active_rows], device=x.device, dtype=torch.long
    )
    row_indices = torch.tensor(active_rows, device=x.device, dtype=torch.long)
    buf._bik_chunk_x_buf[slot_indices, count_indices] = x[row_indices, 0]
    buf._bik_chunk_dt_buf[slot_indices, count_indices] = dt[row_indices, 0]
    buf._bik_chunk_B_buf[slot_indices, count_indices] = B[row_indices, 0]
    buf._bik_chunk_C_buf[slot_indices, count_indices] = C[row_indices, 0]
    if z is not None:
        buf._bik_chunk_z_buf[slot_indices, count_indices] = z[row_indices, 0]

    groups: dict[tuple[int, bool], list[int]] = {}
    for row in active_rows:
        key = (int(counts_list[row]), bool(zero_state_list[row]))
        groups.setdefault(key, []).append(row)

    chunk_scan = module.mamba_chunk_scan_combined
    if chunk_scan is None:
        raise RuntimeError("mamba_ssm is required for BIK Mamba decode")
    chunk_size = buf._bik_chunk_x_buf.shape[1]
    for (count, initial_state_is_zero), group_rows in groups.items():
        group_slots = torch.tensor(
            [slots_list[row] for row in group_rows], device=x.device, dtype=torch.long
        )
        length = count + 1
        x_buffer = buf._bik_chunk_x_buf.index_select(0, group_slots)[:, :length].contiguous()
        dt_buffer = buf._bik_chunk_dt_buf.index_select(0, group_slots)[:, :length].contiguous()
        B_buffer = buf._bik_chunk_B_buf.index_select(0, group_slots)[:, :length].contiguous()
        C_buffer = buf._bik_chunk_C_buf.index_select(0, group_slots)[:, :length].contiguous()
        z_buffer = (
            buf._bik_chunk_z_buf.index_select(0, group_slots)[:, :length].contiguous()
            if z is not None
            else None
        )
        initial_state = (
            None
            if initial_state_is_zero
            else ssm_state.index_select(0, group_slots).contiguous()
        )
        y_run, new_state = chunk_scan(
            x_buffer,
            dt_buffer,
            A,
            B_buffer,
            C_buffer,
            chunk_size,
            D=D,
            z=z_buffer,
            dt_bias=dt_bias,
            dt_softplus=True,
            initial_states=initial_state,
            return_final_states=True,
        )
        group_row_indices = torch.tensor(group_rows, device=x.device, dtype=torch.long)
        y[group_row_indices] = y_run[:, -1:].to(y.dtype)
        if length == chunk_size:
            ssm_state.index_copy_(0, group_slots, new_state.to(ssm_state.dtype))
            buf._bik_chunk_count.index_fill_(0, group_slots, 0)
            buf._bik_state_is_zero.index_fill_(0, group_slots, False)
        else:
            buf._bik_chunk_count.index_fill_(0, group_slots, length)

    return rearrange(y, "b s h p -> b s (h p)") if flatten_output else y


def policy_uses_mamba_layers(config: Mapping[str, Any]) -> bool:
    """Return True when the policy model includes MambaMixer layers."""
    model_name = str(config.get("model_name", "")).lower()
    if any(marker in model_name for marker in _MAMBA_MODEL_NAME_MARKERS):
        return True
    if "mamba" in model_name and "qwen" not in model_name:
        return True

    megatron_cfg = config.get("megatron_cfg") or {}
    if megatron_cfg.get("is_hybrid_model"):
        return True

    hybrid_layer_pattern = megatron_cfg.get("hybrid_layer_pattern")
    if hybrid_layer_pattern is not None and "M" in str(hybrid_layer_pattern).upper():
        return True

    return False


def _nrl_get_mamba_version() -> Any:
    """Read ``mamba_ssm`` version from package metadata without importing it."""
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    from packaging.version import Version as PkgVersion

    if _MAMBA_UTILS_MODULE is None:
        raise RuntimeError("Mamba version metadata patch has not been installed")
    cached = getattr(_MAMBA_UTILS_MODULE, "_mamba_ssm_version", None)
    if cached is not None:
        return cached
    try:
        ver_str = pkg_version("mamba_ssm")
    except PackageNotFoundError:
        raise ImportError("mamba_ssm is not installed") from None
    ver = PkgVersion(ver_str)
    _MAMBA_UTILS_MODULE._mamba_ssm_version = ver
    return ver


def _apply_mamba_version_metadata_patch() -> None:
    global _MAMBA_UTILS_MODULE, _ORIGINAL_GET_MAMBA_VERSION
    if _ORIGINAL_GET_MAMBA_VERSION is not None:
        return
    try:
        mcore_utils = importlib.import_module("megatron.core.utils")
    except ImportError:
        return
    _MAMBA_UTILS_MODULE = mcore_utils
    _ORIGINAL_GET_MAMBA_VERSION = mcore_utils.get_mamba_version
    mcore_utils.get_mamba_version = _nrl_get_mamba_version
    mcore_utils._mamba_ssm_version = None
    print(
        "[zero_train_gen_mismatch] patched get_mamba_version to use "
        "importlib.metadata (skip mamba_ssm/tvm import)",
        flush=True,
    )


def _restore_mamba_version_metadata_patch() -> None:
    global _MAMBA_UTILS_MODULE, _ORIGINAL_GET_MAMBA_VERSION
    if _ORIGINAL_GET_MAMBA_VERSION is None or _MAMBA_UTILS_MODULE is None:
        return
    _MAMBA_UTILS_MODULE.get_mamba_version = _ORIGINAL_GET_MAMBA_VERSION
    _MAMBA_UTILS_MODULE._mamba_ssm_version = None
    _ORIGINAL_GET_MAMBA_VERSION = None
    _MAMBA_UTILS_MODULE = None


def _megatron_uses_public_ssm_api(mixer_class: type) -> bool:
    return hasattr(mixer_class, "ssm_prefill") and not hasattr(
        mixer_class, "_ssm_prefill"
    )


def _ensure_bik_buffer_aliases(buffers: Any) -> None:
    """Map upstream ``BatchInvariantDecodeBuffers`` fields to train-kernel names."""
    buffers._bik_chunk_x_buf = buffers.x
    buffers._bik_chunk_dt_buf = buffers.dt
    buffers._bik_chunk_B_buf = buffers.B
    buffers._bik_chunk_C_buf = buffers.C
    buffers._bik_chunk_z_buf = buffers.z
    buffers._bik_chunk_count = buffers.num_buffered
    max_requests = buffers.num_buffered.shape[0]
    if (
        not hasattr(buffers, "_bik_state_is_zero")
        or buffers._bik_state_is_zero.shape[0] != max_requests
    ):
        buffers._bik_state_is_zero = torch.zeros(
            max_requests, device=buffers.num_buffered.device, dtype=torch.bool
        )


def _nrl_batch_invariant_decode_seed(
    self: Any,
    x: torch.Tensor,
    z: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_indices: torch.Tensor,
) -> None:
    if _ORIGINAL_BIK_DECODE_SEED is None:
        raise RuntimeError("Mamba decode seed patch is not installed")
    _ORIGINAL_BIK_DECODE_SEED(self, x, z, dt, B, C, cu_seqlens, batch_indices)
    _ensure_bik_buffer_aliases(self)
    chunk_size = self.x.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    prefill_lens = cu_seqlens[1:].to(torch.long) - cu_seqlens[:-1].to(torch.long)
    self._bik_state_is_zero.zero_()
    self._bik_state_is_zero[batch_indices[:num_seqs]] = prefill_lens < chunk_size


def _nrl_batch_invariant_decode_buffered_scan(
    buffers: Any,
    x: torch.Tensor,
    z: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_indices: torch.Tensor,
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    _ensure_bik_buffer_aliases(buffers)
    decode_batch_size = x.shape[0]
    y = train_kernel_buffered_scan(
        buffers,
        x,
        dt,
        B,
        C,
        None,
        batch_indices,
        ssm_state,
        A=A,
        D=D,
        dt_bias=dt_bias,
        flatten_output=False,
    )
    buffers.out[:decode_batch_size] = y[:, 0]
    return buffers.out[:decode_batch_size].unsqueeze(1)


def _apply_new_api_train_kernel_decode_patch() -> None:
    global _BID_MODULE, _ORIGINAL_BIK_DECODE_SCAN, _ORIGINAL_BIK_DECODE_SEED
    if _ORIGINAL_BIK_DECODE_SCAN is not None:
        return

    bid_mod = importlib.import_module("megatron.core.ssm.ops.batch_invariant_decode")
    _BID_MODULE = bid_mod
    _ORIGINAL_BIK_DECODE_SCAN = bid_mod.batch_invariant_decode_buffered_scan
    _ORIGINAL_BIK_DECODE_SEED = bid_mod.BatchInvariantDecodeBuffers.seed
    bid_mod.batch_invariant_decode_buffered_scan = _nrl_batch_invariant_decode_buffered_scan
    bid_mod.BatchInvariantDecodeBuffers.seed = _nrl_batch_invariant_decode_seed
    print(
        "[zero_train_gen_mismatch] patched batch_invariant_decode to reuse "
        "mamba_chunk_scan_combined (train kernel) for decode",
        flush=True,
    )


def _restore_new_api_train_kernel_decode_patch() -> None:
    global _BID_MODULE, _ORIGINAL_BIK_DECODE_SCAN, _ORIGINAL_BIK_DECODE_SEED
    if _ORIGINAL_BIK_DECODE_SCAN is None or _BID_MODULE is None:
        return
    _BID_MODULE.batch_invariant_decode_buffered_scan = _ORIGINAL_BIK_DECODE_SCAN
    _BID_MODULE.BatchInvariantDecodeBuffers.seed = _ORIGINAL_BIK_DECODE_SEED
    _ORIGINAL_BIK_DECODE_SCAN = None
    _ORIGINAL_BIK_DECODE_SEED = None
    _BID_MODULE = None


def apply_mamba_alignment_patch(*, required: bool = True) -> None:
    """Install zero-KL Mamba patches once per process."""
    _apply_mamba_version_metadata_patch()
    if _ORIGINAL_BIK_DECODE_SCAN is not None:
        return

    from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.mamba_zero_kl_patches_legacy import (
        apply_legacy_mamba_alignment_patch,
        is_legacy_patch_applied,
    )

    if is_legacy_patch_applied():
        return

    module = importlib.import_module("megatron.core.ssm.mamba_mixer")
    if _megatron_uses_public_ssm_api(module.MambaMixer):
        _apply_new_api_train_kernel_decode_patch()
        return

    apply_legacy_mamba_alignment_patch(module, required=required)


def restore_mamba_alignment_patch() -> None:
    """Restore patched Megatron modules (for unit tests)."""
    _restore_mamba_version_metadata_patch()
    _restore_new_api_train_kernel_decode_patch()
    from nemo_rl.models.generation.megatron.zero_train_gen_kl_patches.mamba_zero_kl_patches_legacy import (
        restore_legacy_mamba_alignment_patch,
    )

    restore_legacy_mamba_alignment_patch()
