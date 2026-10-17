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

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader

from nemo_rl.models.megatron import ckpt_parallel_read


@pytest.fixture(autouse=True)
def _restore_reader():
    original = FileSystemReader.read_data
    yield
    FileSystemReader.read_data = original


def _state(fill: float) -> dict:
    return {
        "big": torch.full((5_000_000,), fill),  # 20 MB: several read requests
        "small": torch.full((3, 7), fill, dtype=torch.bfloat16),
        "blob": io.BytesIO(b"bytes item" * (1 + int(fill))),
    }


def _load(path) -> dict:
    state = _state(0.0)
    dcp.load(state, storage_reader=FileSystemReader(path), no_dist=True)
    return state


def test_parallel_read_matches_pytorch_reader(tmp_path, monkeypatch):
    dcp.save(_state(2.0), checkpoint_id=tmp_path, no_dist=True)
    expected = _load(tmp_path)

    monkeypatch.setenv("NRL_CKPT_READ_THREADS", "4")
    monkeypatch.setattr(ckpt_parallel_read, "CHUNK_BYTES", (1 << 20) + 7)
    monkeypatch.setattr(ckpt_parallel_read, "WINDOW_BYTES", 4 << 20)
    ckpt_parallel_read.install()
    assert FileSystemReader.read_data is not ckpt_parallel_read._original_read_data
    actual = _load(tmp_path)

    for key in ("big", "small"):
        assert torch.equal(actual[key], expected[key])
    assert actual["blob"].getvalue() == expected["blob"].getvalue()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("NRL_CKPT_READ_THREADS", raising=False)
    ckpt_parallel_read.install()
    assert FileSystemReader.read_data is ckpt_parallel_read._original_read_data


def test_unsupported_torch_raises(monkeypatch):
    monkeypatch.setenv("NRL_CKPT_READ_THREADS", "4")
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(RuntimeError, match="supports torch 2.11-2.13"):
        ckpt_parallel_read.install()
