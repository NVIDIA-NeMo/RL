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

Each int64 lane is bound to its position through a nonlinear mix and reduced
with wraparound addition on two independently-parameterized 64-bit channels
(128 bits of state total). Wraparound addition is associative and
commutative, so the result is bit-identical regardless of reduction order,
chunking, or device. A single such channel is not enough: for any bijective
per-lane transform, a corruption that permutes the transformed lane values
leaves a commutative sum unchanged and can be constructed in closed form.
With two channels the same corrupted lanes must simultaneously preserve both
nonlinearly-coupled sums, for which no closed-form construction exists. The
tensor's dtype and shape seed both channels so equal-size metadata drift is
also detected.
"""

import hashlib

import torch

_U64 = 1 << 64
# SplitMix64 finalizer constants.
_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB
# Independent per-channel position salts and the second channel's lane offset.
# A *linear* position salt (lane + pos * salt) is absorbable: corrupted lanes
# can soak up the salt difference between two positions and permute the salted
# values, which a commutative sum cannot see. Positions are therefore injected
# through the nonlinear mixer (lane ^ _mix64(pos * salt)).
_CHANNEL_1_POSITION_SALT = 0x9E3779B97F4A7C15
_CHANNEL_2_POSITION_SALT = 0xC2B2AE3D27D4EB4F
_CHANNEL_2_LANE_OFFSET = 0x452821E638D01377
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


def _metadata_seeds(tensor: torch.Tensor) -> tuple[int, int]:
    """Stable per-channel 64-bit seeds covering dtype, shape, and byte length."""
    metadata = hashlib.blake2b(digest_size=16, person=b"NeMoRLrefit-v3")
    dtype_name = str(tensor.dtype).encode("ascii")
    metadata.update(len(dtype_name).to_bytes(2, byteorder="little"))
    metadata.update(dtype_name)
    metadata.update(tensor.ndim.to_bytes(8, byteorder="little"))
    for dimension in tensor.shape:
        metadata.update(dimension.to_bytes(8, byteorder="little"))
    metadata.update(tensor.nbytes.to_bytes(8, byteorder="little"))
    raw = metadata.digest()
    return (
        int.from_bytes(raw[:8], byteorder="little"),
        int.from_bytes(raw[8:], byteorder="little"),
    )


def tensor_digest(tensor: torch.Tensor) -> torch.Tensor:
    """Digest a tensor's logical byte stream into a 2-element int64 tensor.

    The 128-bit digest covers the tensor's dtype and shape followed by
    exactly ``tensor.nbytes`` bytes in flattened logical element order.

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

    seed_1, seed_2 = _metadata_seeds(tensor)
    channel_1 = torch.full((), _as_i64(seed_1), dtype=torch.int64, device=lanes.device)
    channel_2 = torch.full((), _as_i64(seed_2), dtype=torch.int64, device=lanes.device)
    lane_offset = 0
    for chunk in lanes.split(_CHUNK_LANES):
        lanes_in_chunk = chunk.numel()
        positions = torch.arange(
            lane_offset + 1,
            lane_offset + lanes_in_chunk + 1,
            dtype=torch.int64,
            device=lanes.device,
        )
        position_mix_1 = _mix64(positions * _as_i64(_CHANNEL_1_POSITION_SALT))
        position_mix_2 = _mix64(positions * _as_i64(_CHANNEL_2_POSITION_SALT))
        channel_1 = channel_1 + _mix64(chunk ^ position_mix_1).sum(dtype=torch.int64)
        channel_2 = channel_2 + _mix64(
            (chunk + _as_i64(_CHANNEL_2_LANE_OFFSET)) ^ position_mix_2
        ).sum(dtype=torch.int64)
        lane_offset += lanes_in_chunk
    return torch.stack([_mix64(channel_1), _mix64(channel_2)])


def digests_to_ints(digests: dict[str, torch.Tensor]) -> dict[str, str]:
    """Normalize digest tensors to 32-hex strings (forces a device sync)."""
    if not digests:
        return {}

    device_batches: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for name, digest in digests.items():
        device_batches.setdefault(digest.device, []).append((name, digest))

    normalized: dict[str, str] = {}
    for batch in device_batches.values():
        stacked = torch.stack([digest for _, digest in batch]).cpu()
        for (name, _), (first, second) in zip(batch, stacked.tolist()):
            normalized[name] = (
                f"{int(first) & (_U64 - 1):016x}{int(second) & (_U64 - 1):016x}"
            )
    return {name: normalized[name] for name in digests}


def compare_digests(sender: dict[str, str], receiver: dict[str, str]) -> list[str]:
    """Parameter names whose digests disagree or are missing on either side."""
    return sorted(
        name
        for name in set(sender) | set(receiver)
        if sender.get(name) != receiver.get(name)
    )
