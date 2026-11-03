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
"""Parallel reads for torch_dist checkpoint loads.

PyTorch's ``FileSystemReader.read_data`` reads a rank's items one at a time from one
stream, and Megatron builds its reader with no load-side thread count, so this patches
``read_data`` to read the same byte ranges with N threads (16 MB requests, at most
``WINDOW_BYTES`` ahead) while the calling thread loads the items in order with
PyTorch's own code. Peak host memory per rank: ``WINDOW_BYTES`` plus ~2x the largest item.
"""

import collections
import hashlib
import inspect
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from torch.distributed.checkpoint import filesystem as dcp_fs
from torch.distributed.checkpoint.planner import LoadItemType

CHUNK_BYTES = 16 << 20
WINDOW_BYTES = 1 << 30
# sha256 of inspect.getsource(FileSystemReader.read_data) in torch 2.11-2.13 (same code).
READ_DATA_SHA256 = "86ad890ac621f4f563ee5f66a2059ebb3a3ba022a58d91d4e23d757405c691be"  # pragma: allowlist secret
_original_read_data = dcp_fs.FileSystemReader.read_data


def install(threads: int) -> None:
    """Patch ``FileSystemReader.read_data`` to read with ``threads`` threads (if > 1)."""
    if threads <= 1:
        return
    source = inspect.getsource(_original_read_data)
    if hashlib.sha256(source.encode()).hexdigest() != READ_DATA_SHA256:
        raise RuntimeError(
            f"torch {torch.__version__} changed FileSystemReader.read_data, which "
            "parallel_read_threads replaces; set it to 1, or port the change into "
            "nemo_rl/models/megatron/ckpt_parallel_read.py and update READ_DATA_SHA256."
        )

    def read_data(self: Any, plan: Any, planner: Any) -> Any:
        if not isinstance(self.fs, dcp_fs.FileSystem):  # e.g. fsspec readers
            return _original_read_data(self, plan, planner)
        per_file: dict[str, list[Any]] = {}
        for req in plan.items:
            per_file.setdefault(
                self.storage_data[req.storage_index].relative_path, []
            ).append(req)
        for relative_path, reqs in per_file.items():
            path = os.fspath(self.fs.concat_path(self.path, relative_path))
            fd = os.open(path, os.O_RDONLY)
            pool = ThreadPoolExecutor(threads, thread_name_prefix="nrl-ckpt-read")
            try:
                _read_file(self, planner, fd, pool, reqs)
            except EOFError as e:
                raise EOFError(f"{path}: {e}") from e
            finally:
                # No read may outlive the file descriptor.
                pool.shutdown(wait=True, cancel_futures=True)
                os.close(fd)
        fut: torch.futures.Future = torch.futures.Future()
        fut.set_result(None)
        return fut

    dcp_fs.FileSystemReader.read_data = read_data


def _read_file(reader: Any, planner: Any, fd: int, pool: Any, reqs: list) -> None:
    pending: collections.deque = collections.deque()
    in_flight = 0
    for req in sorted(reqs, key=lambda r: reader.storage_data[r.storage_index].offset):
        info = reader.storage_data[req.storage_index]
        buf = bytearray(info.length)
        view = memoryview(buf)
        futures = [
            pool.submit(_pread_into, fd, view[o : o + CHUNK_BYTES], info.offset + o)
            for o in range(0, info.length, CHUNK_BYTES)
        ]
        pending.append((req, buf, futures))
        in_flight += info.length
        while pending and in_flight > WINDOW_BYTES:
            in_flight -= _load_oldest(reader, planner, pending)
    while pending:
        _load_oldest(reader, planner, pending)


def _load_oldest(reader: Any, planner: Any, pending: collections.deque) -> int:
    req, buf, futures = pending.popleft()
    for f in futures:
        f.result()
    _load_item(reader, planner, req, buf)
    return len(buf)


def _pread_into(fd: int, view: memoryview, offset: int) -> None:
    done = 0
    while done < len(view):
        n = os.preadv(fd, [view[done:]], offset + done)
        if n == 0:
            raise EOFError(f"checkpoint file ends at byte {offset + done}")
        done += n


def _load_item(reader: Any, planner: Any, req: Any, buf: bytearray) -> None:
    # The body of PyTorch's read_data loop, with the file slice replaced by the
    # bytes already read.
    info = reader.storage_data[req.storage_index]
    stream = reader.transforms.transform_load_stream(
        req, info.transform_descriptors or (), io.BytesIO(buf)
    )
    if req.type == LoadItemType.BYTE_IO:
        planner.load_bytes(req, io.BytesIO(stream.read(-1)))
        return
    if not stream.seekable():
        stream = io.BytesIO(stream.read(-1))
    tensor = torch.load(stream, map_location="cpu", weights_only=True)
    tensor = dcp_fs.narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
    target = planner.resolve_tensor(req).detach()
    if target.size() != tensor.size():
        raise AssertionError(
            f"req {req.storage_index} mismatch sizes {target.size()} vs {tensor.size()}"
        )
    target.copy_(tensor)
    planner.commit_tensor(req, target)
