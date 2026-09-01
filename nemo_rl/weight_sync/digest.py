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

The digest is a polynomial rolling hash over a tensor's flattened byte
stream, computed in int64 with hardware wraparound (i.e. mod 2**64).
Integer modular addition and multiplication are associative and
commutative, so the result is bit-identical regardless of reduction
order, chunking, or device -- unlike any floating-point checksum, which
would itself be subject to the nondeterminism this module is meant to
detect. Both ends of a transfer can therefore hash "the bytes they
sent" and "the bytes they received" independently and compare exact
integers.
"""

import torch

_FNV_PRIME = 0x100000001B3
_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_U64 = 1 << 64
# 1 Mi int64 lanes (8 MiB) hashed per chunk; bounds the cached power table.
_CHUNK_LANES = 1 << 20

_POW_CACHE: dict[str, torch.Tensor] = {}


def _as_i64(value: int) -> int:
    """Map an unsigned 64-bit value onto the torch.int64 two's-complement range."""
    value &= _U64 - 1
    return value - _U64 if value >= (1 << 63) else value


def _descending_powers(device: torch.device) -> torch.Tensor:
    """[R^(_CHUNK_LANES-1), ..., R^1, R^0] as int64, mod 2**64."""
    key = str(device)
    cached = _POW_CACHE.get(key)
    if cached is None:
        base = torch.full(
            (_CHUNK_LANES,), _as_i64(_FNV_PRIME), dtype=torch.int64, device=device
        )
        base[0] = 1
        cached = torch.cumprod(base, dim=0).flip(0)
        _POW_CACHE[key] = cached
    return cached


def tensor_digest(tensor: torch.Tensor) -> torch.Tensor:
    """Digest a tensor's logical byte stream into a 0-dim int64 tensor.

    The hash covers exactly ``tensor.nbytes`` bytes in flattened logical
    element order, with the byte length folded in at the end, so tensors
    of different lengths (or trailing zero padding) cannot collide with
    each other's prefixes.

    The result stays on ``tensor.device`` so callers can batch hashing on
    the active CUDA stream without a device sync per tensor; convert with
    :func:`digest_to_int` only after the producing stream is synchronized.
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
        (), _as_i64(_FNV_OFFSET_BASIS), dtype=torch.int64, device=lanes.device
    )
    powers = _descending_powers(lanes.device)
    for chunk in lanes.split(_CHUNK_LANES):
        lanes_in_chunk = chunk.numel()
        # H(prefix + chunk) = H(prefix) * R^len(chunk) + H(chunk), mod 2**64.
        digest = digest * _as_i64(pow(_FNV_PRIME, lanes_in_chunk, _U64)) + (
            chunk * powers[-lanes_in_chunk:]
        ).sum(dtype=torch.int64)
    return digest * _as_i64(_FNV_PRIME) + num_bytes


def digests_to_ints(digests: dict[str, torch.Tensor]) -> dict[str, int]:
    """Normalize digest tensors to unsigned Python ints (forces a device sync)."""
    if not digests:
        return {}
    stacked = torch.stack(list(digests.values())).cpu()
    return {
        name: int(value) & (_U64 - 1)
        for name, value in zip(digests.keys(), stacked.tolist())
    }


def compare_digests(sender: dict[str, int], receiver: dict[str, int]) -> list[str]:
    """Parameter names whose digests disagree or are missing on either side."""
    return sorted(
        name
        for name in set(sender) | set(receiver)
        if sender.get(name) != receiver.get(name)
    )
