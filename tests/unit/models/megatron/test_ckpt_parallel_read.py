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
import io
import os

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.filesystem import FileSystemReader

from nemo_rl.models.megatron import ckpt_parallel_read


@pytest.fixture(autouse=True)
def _restore_reader():
    original = FileSystemReader.read_data
    yield
    FileSystemReader.read_data = original


def _state(offset: float) -> dict:
    # Every element differs, so a read from a wrong offset changes the result.
    return {
        "big": torch.arange(5_000_000, dtype=torch.float32) + offset,  # 20 MB
        "small": torch.arange(21, dtype=torch.bfloat16).reshape(3, 7) + offset,
        "blob": io.BytesIO(b"bytes item" * (1 + int(offset))),
    }


def _load(path) -> dict:
    state = _state(0.0)
    dcp.load(state, storage_reader=FileSystemReader(path), no_dist=True)
    return state


def _assert_same(actual: dict, expected: dict) -> None:
    for key in ("big", "small"):
        assert torch.equal(actual[key], expected[key])
    assert actual["blob"].getvalue() == expected["blob"].getvalue()


def test_parallel_read_matches_pytorch_reader(tmp_path, monkeypatch):
    dcp.save(_state(2.0), checkpoint_id=tmp_path, no_dist=True)
    expected = _load(tmp_path)

    monkeypatch.setattr(ckpt_parallel_read, "CHUNK_BYTES", (1 << 20) + 7)
    monkeypatch.setattr(ckpt_parallel_read, "WINDOW_BYTES", 4 << 20)
    ckpt_parallel_read.install(4)
    assert FileSystemReader.read_data is not ckpt_parallel_read._original_read_data
    _assert_same(_load(tmp_path), expected)


@pytest.mark.timeout(60)
def test_parallel_read_short_reads(tmp_path, monkeypatch):
    dcp.save(_state(2.0), checkpoint_id=tmp_path, no_dist=True)
    expected = _load(tmp_path)

    real_preadv = os.preadv
    # At most 4097 bytes per call, so the retry loop runs.
    monkeypatch.setattr(
        os, "preadv", lambda fd, bufs, off: real_preadv(fd, [bufs[0][:4097]], off)
    )
    ckpt_parallel_read.install(4)
    _assert_same(_load(tmp_path), expected)


@pytest.mark.timeout(60)
def test_parallel_read_truncated_file_raises(tmp_path):
    dcp.save(_state(2.0), checkpoint_id=tmp_path, no_dist=True)
    (data_file,) = tmp_path.glob("*.distcp")
    os.truncate(data_file, data_file.stat().st_size // 2)

    ckpt_parallel_read.install(4)
    # CheckpointException subclasses BaseException, not Exception.
    with pytest.raises(CheckpointException, match="checkpoint file ends at byte"):
        _load(tmp_path)


def test_one_thread_keeps_pytorch_reader():
    ckpt_parallel_read.install(1)
    assert FileSystemReader.read_data is ckpt_parallel_read._original_read_data


def test_changed_pytorch_reader_raises(monkeypatch):
    def read_data(self, plan, planner):  # stands in for a changed upstream version
        return None

    monkeypatch.setattr(ckpt_parallel_read, "_original_read_data", read_data)
    with pytest.raises(RuntimeError, match="changed FileSystemReader.read_data"):
        ckpt_parallel_read.install(4)
