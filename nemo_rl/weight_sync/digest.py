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

"""Deterministic per-tensor digests for refit weight-transfer verification.

Each int64 lane is combined with its position and passed through a nonlinear
64-bit mixer before reduction. Integer addition and multiplication use
hardware wraparound (i.e. mod 2**64), so the result is bit-identical
regardless of reduction order, chunking, or device. The tensor's dtype and
shape are folded into the result as well as its bytes so stale refit metadata
cannot reinterpret an otherwise identical byte stream undetected.
"""

import hashlib

import torch

_U64 = 1 << 64
# SplitMix64 constants. Position-salting before the bijective finalizer avoids
# the deterministic cancellation that a linear polynomial has for paired
# high-bit flips in separate int64 lanes.
_POSITION_SALT = 0x9E3779B97F4A7C15
_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB
# 1 Mi int64 lanes (8 MiB) hashed per chunk.
_CHUNK_LANES = 1 << 20


def _as_i64(value: int) -> int:
    """Map an unsigned 64-bit value onto the torch.int64 two's-complement range."""
    value &= _U64 - 1
    return value - _U64 if value >= (1 << 63) else value


def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Shift an int64 tensor right as unsigned values."""
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def _mix64(value: torch.Tensor) -> torch.Tensor:
    """Apply the SplitMix64 finalizer with int64 wraparound arithmetic."""
    value = (value ^ _logical_right_shift(value, 30)) * _as_i64(_MIX_MULTIPLIER_1)
    value = (value ^ _logical_right_shift(value, 27)) * _as_i64(_MIX_MULTIPLIER_2)
    return value ^ _logical_right_shift(value, 31)


def _metadata_seed(tensor: torch.Tensor) -> int:
    """Stable 64-bit seed covering the tensor's dtype, shape, and byte length."""
    metadata = hashlib.blake2b(digest_size=8, person=b"NeMoRLrefit-v2")
    dtype_name = str(tensor.dtype).encode("ascii")
    metadata.update(len(dtype_name).to_bytes(2, byteorder="little"))
    metadata.update(dtype_name)
    metadata.update(tensor.ndim.to_bytes(8, byteorder="little"))
    for dimension in tensor.shape:
        metadata.update(dimension.to_bytes(8, byteorder="little"))
    metadata.update(tensor.nbytes.to_bytes(8, byteorder="little"))
    return int.from_bytes(metadata.digest(), byteorder="little")


def tensor_digest(tensor: torch.Tensor) -> torch.Tensor:
    """Digest a tensor's logical byte stream into a 0-dim int64 tensor.

    The hash covers the tensor's dtype and shape followed by exactly
    ``tensor.nbytes`` bytes in flattened logical element order.

    The result stays on ``tensor.device`` so callers can batch hashing on
    the active CUDA stream without a device sync per tensor; convert with
    :func:`digests_to_ints` only after the producing stream is synchronized.
    """
    data = tensor.detach().reshape(-1)
    if not data.is_contiguous():
        data = data.contiguous()
    raw = data.view(torch.uint8)
    num_bytes = raw.numel()
    pad = (-num_bytes) % 8
    if pad or (raw.storage_offset() % 8):
        # int64 lanes need 8-byte-aligned storage and a multiple-of-8 length;
        # a bf16 slice at an odd element offset satisfies neither.
        padded = raw.new_zeros(num_bytes + pad)
        padded[:num_bytes] = raw
        raw = padded
    lanes = raw.view(torch.int64)

    digest = torch.full(
        (), _as_i64(_metadata_seed(tensor)), dtype=torch.int64, device=lanes.device
    )
    lane_offset = 0
    for chunk in lanes.split(_CHUNK_LANES):
        lanes_in_chunk = chunk.numel()
        positions = torch.arange(
            lane_offset + 1,
            lane_offset + lanes_in_chunk + 1,
            dtype=torch.int64,
            device=lanes.device,
        )
        salted = chunk + positions * _as_i64(_POSITION_SALT)
        digest = digest + _mix64(salted).sum(dtype=torch.int64)
        lane_offset += lanes_in_chunk
    return _mix64(digest)


def digests_to_ints(digests: dict[str, torch.Tensor]) -> dict[str, int]:
    """Normalize digest tensors to unsigned Python ints (forces a device sync)."""
    if not digests:
        return {}

    device_batches: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for name, digest in digests.items():
        device_batches.setdefault(digest.device, []).append((name, digest))

    normalized: dict[str, int] = {}
    for batch in device_batches.values():
        stacked = torch.stack([digest for _, digest in batch]).cpu()
        for (name, _), value in zip(batch, stacked.tolist()):
            normalized[name] = int(value) & (_U64 - 1)
    return {name: normalized[name] for name in digests}


def compare_digests(sender: dict[str, int], receiver: dict[str, int]) -> list[str]:
    """Parameter names whose digests disagree or are missing on either side."""
    return sorted(
        name
        for name in set(sender) | set(receiver)
        if sender.get(name) != receiver.get(name)
    )
