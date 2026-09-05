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

Int64 lanes are folded through an ordered binary tree whose pairwise merge
is non-commutative (left and right children enter through different odd
multipliers before a nonlinear mix), on two independently-parameterized
64-bit channels. Every reduction step is elementwise over disjoint pairs, so
the fold parallelizes on GPU and is bit-identical across devices; a lane's
position is bound to its path through the tree rather than to its value.

Commutative reductions are structurally unsafe here no matter how many
channels they use: k channels impose k multiset constraints while n lanes
provide n degrees of freedom, so lane permutations satisfying all channels
simultaneously exist and have been constructed for one- and two-channel
variants of this module. An ordered tree has no such permutation freedom.
The tensor's dtype and shape seed both channels so equal-size metadata
drift is also detected.
"""

import hashlib

import torch

_U64 = 1 << 64
# SplitMix64 finalizer constants.
_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB
# Per-channel (left tweak, right tweak, multiplier) of the tree merge.
# Both children pass through the mixer (with distinct tweaks) before the
# linear combination: a bare linear combination leaves a structural channel
# open, because two odd multipliers sum to an even number and
# 2^63 * even == 0 mod 2^64, so flipping the top bit of both children
# cancels. Distinct tweaks keep the merge non-commutative and remove the
# all-zero fixed point of the mixer.
_CHANNEL_1_LEFT_TWEAK = 0x9E3779B97F4A7C15
_CHANNEL_1_RIGHT_TWEAK = 0xC2B2AE3D27D4EB4F
_CHANNEL_1_MULTIPLIER = 0xFF51AFD7ED558CCD
_CHANNEL_2_LEFT_TWEAK = 0x452821E638D01377
_CHANNEL_2_RIGHT_TWEAK = 0xD1B54A32D192ED03
_CHANNEL_2_MULTIPLIER = 0xC4CEB9FE1A85EC53
# Chunk-chain multiplier binding the order of per-chunk tree roots.
_CHAIN_MULTIPLIER = 0x2545F4914F6CDD1D
# Lanes folded per tree (an algorithm constant, not a tuning knob: changing
# it changes chunk boundaries and therefore the digest).
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


def _tree_fold(
    lanes: torch.Tensor, left_tweak: int, right_tweak: int, multiplier: int
) -> torch.Tensor:
    """Fold a power-of-two lane vector through an ordered binary tree.

    Each level merges disjoint (left, right) pairs elementwise as
    ``mix64(mix64(left ^ lt) + mix64(right ^ rt) * mult)``. Mixing both
    children before the combination closes the linear cancellation channel;
    distinct tweaks make the merge order-sensitive, so a lane's position is
    encoded by its path through the tree. Returns a 0-dim int64 tensor.
    """
    value = lanes
    lt = _as_i64(left_tweak)
    rt = _as_i64(right_tweak)
    mult = _as_i64(multiplier)
    while value.numel() > 1:
        value = _mix64(_mix64(value[0::2] ^ lt) + _mix64(value[1::2] ^ rt) * mult)
    return value.reshape(())


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
    chain = _as_i64(_CHAIN_MULTIPLIER)
    for chunk in lanes.split(_CHUNK_LANES):
        lanes_in_chunk = chunk.numel()
        tree_width = 1 << max(1, (lanes_in_chunk - 1).bit_length())
        if tree_width != lanes_in_chunk:
            padded_chunk = chunk.new_zeros(tree_width)
            padded_chunk[:lanes_in_chunk] = chunk
            chunk = padded_chunk
        root_1 = _tree_fold(
            chunk, _CHANNEL_1_LEFT_TWEAK, _CHANNEL_1_RIGHT_TWEAK, _CHANNEL_1_MULTIPLIER
        )
        root_2 = _tree_fold(
            chunk, _CHANNEL_2_LEFT_TWEAK, _CHANNEL_2_RIGHT_TWEAK, _CHANNEL_2_MULTIPLIER
        )
        # Chunk roots are chained in order, so chunk position is bound too.
        channel_1 = _mix64(channel_1 * chain + root_1)
        channel_2 = _mix64(channel_2 * chain + root_2)
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
