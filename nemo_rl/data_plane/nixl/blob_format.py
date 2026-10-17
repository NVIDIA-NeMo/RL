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
"""Blob codec: many values → one self-describing byte blob, no per-tensor serialization.

Tensors and numpy arrays are stored as their raw contiguous bytes. What the
bytes mean (kind, dtype, shape) lives in two places that we own: the
per-entry location meta TransferQueue hands back on every get, and the
blob's trailing index so a unit can read its own blobs without the
controller (scan, checkpoint shards). Only non-tensor objects are pickled.

Layout::

    [ entry_0 | pad | entry_1 | pad | ... | index (msgpack) | footer 16 B ]

Entries are 64-byte aligned.
"""

from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass
from typing import Any, Sequence

import msgspec
import numpy as np
import torch

ALIGN = 64
MAGIC = 0x4E564450  # 'NVDP'
VERSION = 3
# v3 adds a 16-byte blob tag (the blob id) so a reader holding a cached
# (unit, offset) location can verify, from the same READ, that the bytes there
# are still this blob -- the basis of the RPC-free read path. v2 blobs (older
# checkpoints) still decode; they just carry no tag.
_FOOTER_FMT = "<IHHII16s"  # magic, version, flags, index_off, index_len, blob_tag
FOOTER_SIZE = struct.calcsize(_FOOTER_FMT)
_FOOTER_V2_FMT = "<IHHII"
FOOTER_V2_SIZE = struct.calcsize(_FOOTER_V2_FMT)
NO_TAG = bytes(16)

KIND_TENSOR = "t"
KIND_NUMPY = "n"
KIND_PICKLE = "p"


def _align_up(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass(frozen=True)
class Entry:
    key: str
    off: int
    len: int
    kind: str
    dtype: str | None  # "float32" / "<f4" / None for pickle
    shape: tuple[int, ...] | None

    def meta(self) -> dict[str, Any]:
        """Location meta stored by TQ as ``custom_backend_meta`` (blob id added by the client)."""
        return {
            "o": self.off,
            "n": self.len,
            "k": self.kind,
            "d": self.dtype,
            "s": list(self.shape or ()),
        }

    @staticmethod
    def from_meta(key: str, m: dict[str, Any]) -> "Entry":
        return Entry(
            key, int(m["o"]), int(m["n"]), m["k"], m.get("d"), tuple(m.get("s") or ())
        )


@dataclass
class BlobPlan:
    keys: list[str]
    buffers: list[memoryview]  # raw bytes per value; keepalive holds their owners
    keepalive: list[Any]
    entries: list[Entry]
    index_bytes: bytes
    nbytes: int
    tag: bytes = NO_TAG

    @property
    def index_off(self) -> int:
        return self.nbytes - FOOTER_SIZE - len(self.index_bytes)


# ----------------------------------------------------------------------------- encode
def _tensor_bytes(t: torch.Tensor) -> tuple[memoryview, Any]:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return memoryview(b""), t
    raw = flat.view(torch.uint8) if t.dtype != torch.uint8 else flat
    arr = raw.numpy()
    return memoryview(arr), t  # arr keeps the storage alive


def _numpy_bytes(a: np.ndarray) -> tuple[memoryview, Any]:
    a = np.ascontiguousarray(a)
    return memoryview(a.view(np.uint8).ravel()), a


def encode_value(
    key: str, v: Any
) -> tuple[memoryview, Any, str, str | None, tuple[int, ...] | None]:
    if isinstance(v, torch.Tensor) and not v.is_nested and not v.is_sparse:
        mv, keep = _tensor_bytes(v)
        return (
            mv,
            keep,
            KIND_TENSOR,
            str(v.dtype).removeprefix("torch."),
            tuple(v.shape),
        )
    if isinstance(v, np.ndarray) and v.dtype != object:
        mv, keep = _numpy_bytes(v)
        return mv, keep, KIND_NUMPY, v.dtype.str, tuple(v.shape)
    b = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
    return memoryview(b), b, KIND_PICKLE, None, None


def plan(keys: Sequence[str], values: Sequence[Any], tag: bytes = NO_TAG) -> BlobPlan:
    if len(keys) != len(values):
        raise ValueError("keys and values differ in length")
    buffers: list[memoryview] = []
    keepalive: list[Any] = []
    entries: list[Entry] = []
    off = 0
    for k, v in zip(keys, values):
        mv, keep, kind, dtype, shape = encode_value(str(k), v)
        buffers.append(mv)
        keepalive.append(keep)
        entries.append(Entry(str(k), off, mv.nbytes, kind, dtype, shape))
        off = _align_up(off + mv.nbytes)
    index_bytes = msgspec.msgpack.encode(
        [
            [
                e.key,
                e.off,
                e.len,
                e.kind,
                e.dtype,
                list(e.shape) if e.shape is not None else None,
            ]
            for e in entries
        ]
    )
    nbytes = off + len(index_bytes) + FOOTER_SIZE
    return BlobPlan(
        list(map(str, keys)),
        buffers,
        keepalive,
        entries,
        index_bytes,
        nbytes,
        bytes(tag)[:16].ljust(16, b"\0"),
    )


def write_tail(p: BlobPlan, out: memoryview) -> None:
    """Write only the index and footer (``len(p.index_bytes) + FOOTER_SIZE`` bytes).

    For a blob whose values are sent separately. ``out`` maps to blob offset
    ``p.index_off``.
    """
    out = out.cast("B")
    n = len(p.index_bytes)
    out[:n] = p.index_bytes
    struct.pack_into(_FOOTER_FMT, out, n, MAGIC, VERSION, 0, p.index_off, n, p.tag)


def write(p: BlobPlan, out: memoryview) -> None:
    out = out.cast("B")
    if out.nbytes < p.nbytes:
        raise ValueError(f"buffer has {out.nbytes} B, blob needs {p.nbytes} B")
    for e, mv in zip(p.entries, p.buffers):
        if e.len:
            out[e.off : e.off + e.len] = mv.cast("B")
    io = p.index_off
    out[io : io + len(p.index_bytes)] = p.index_bytes
    struct.pack_into(
        _FOOTER_FMT,
        out,
        p.nbytes - FOOTER_SIZE,
        MAGIC,
        VERSION,
        0,
        io,
        len(p.index_bytes),
        p.tag,
    )


# ----------------------------------------------------------------------------- decode
def _torch_dtype(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r}")
    return dt


def decode_entry(view: memoryview, e: Entry) -> Any:
    """Reconstruct a value from its bytes. Tensors and arrays are zero-copy views."""
    if e.kind == KIND_TENSOR:
        dt = _torch_dtype(e.dtype or "float32")
        shape = tuple(e.shape or ())
        if e.len == 0:
            return torch.empty(shape, dtype=dt)
        return torch.frombuffer(view, dtype=dt).view(shape)
    if e.kind == KIND_NUMPY:
        return np.frombuffer(view, dtype=np.dtype(e.dtype)).reshape(
            tuple(e.shape or ())
        )
    if e.kind == KIND_PICKLE:
        return pickle.loads(view)
    raise ValueError(f"unknown entry kind {e.kind!r}")


def materialize(v: Any) -> Any:
    """Copy a decoded zero-copy view out of the transfer buffer."""
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, np.ndarray):
        return v.copy()
    return v


def _footer(blob: memoryview) -> tuple[int, int, bytes] | None:
    """``(index_off, index_len, tag)`` from a v3 or v2 footer at the end of ``blob``."""
    blob = blob.cast("B")
    if blob.nbytes >= FOOTER_SIZE:
        magic, version, _f, io, il, tag = struct.unpack_from(
            _FOOTER_FMT, blob, blob.nbytes - FOOTER_SIZE
        )
        if magic == MAGIC and version == VERSION:
            return io, il, tag
    if blob.nbytes >= FOOTER_V2_SIZE:
        magic, version, _f, io, il = struct.unpack_from(
            _FOOTER_V2_FMT, blob, blob.nbytes - FOOTER_V2_SIZE
        )
        if magic == MAGIC and version == 2:
            return io, il, NO_TAG
    return None


def footer_tag(tail: memoryview) -> bytes | None:
    """Blob tag from a v3 footer (exactly the last ``FOOTER_SIZE`` bytes), else ``None``."""
    f = _footer(tail)
    return None if f is None or f[2] == NO_TAG else f[2]


def read_index(blob: memoryview) -> list[Entry]:
    blob = blob.cast("B")
    f = _footer(blob)
    if f is None:
        if blob.nbytes < FOOTER_V2_SIZE:
            raise ValueError("blob too small for a footer")
        raise ValueError("bad blob magic or unsupported blob version")
    io, il, _tag = f
    raw = msgspec.msgpack.decode(bytes(blob[io : io + il]))
    return [
        Entry(k, o, n, kind, d, tuple(s) if s is not None else None)
        for k, o, n, kind, d, s in raw
    ]


def footer_is_valid(blob_tail: memoryview) -> bool:
    return _footer(blob_tail) is not None
