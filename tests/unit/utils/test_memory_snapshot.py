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
"""Opt-in OOM memory snapshots."""

import os

import pytest
import torch

from nemo_rl.utils import memory_snapshot as ms


def _fake_snapshot():
    gib = 2**30
    return {
        "segments": [
            {
                "blocks": [
                    {"state": "active_allocated", "size": 9 * gib},
                    {"state": "inactive", "size": 3 * gib},
                    {"state": "active_allocated", "size": gib // 3},
                ]
            },
            {"blocks": [{"state": "active_allocated", "size": gib // 3}]},
        ]
    }


def test_summarize_live_blocks_counts_only_active_allocated():
    summary = ms.summarize_live_blocks(_fake_snapshot(), top_k=2)
    gib = 2**30
    assert summary["live_blocks"] == 3
    assert summary["live_total_bytes"] == 9 * gib + 2 * (gib // 3)
    assert summary["top_block_bytes"] == [9 * gib, gib // 3]


def test_enable_memory_history_is_a_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ms.enable_memory_history() is False


class _Worker:
    def __init__(self, cfg):
        self._memory_snapshot_cfg = (
            ms.MemorySnapshotConfig.model_validate(cfg)
            if isinstance(cfg, dict)
            else cfg
        )
        self.calls = 0

    @ms.snapshot_on_oom("train")
    def train(self, fail: bool):
        self.calls += 1
        if fail:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return "ok"


def test_decorator_dumps_snapshot_then_reraises(monkeypatch, tmp_path, capsys):
    dumped: list[str] = []
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", _fake_snapshot)
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", lambda p: dumped.append(p))

    worker = _Worker({"enabled": True, "directory": str(tmp_path)})
    assert worker.train(fail=False) == "ok"
    with pytest.raises(torch.OutOfMemoryError):
        worker.train(fail=True)

    assert len(dumped) == 1
    assert os.path.dirname(dumped[0]) == str(tmp_path)
    assert os.path.basename(dumped[0]).startswith(f"oom_train_rank0_pid{os.getpid()}_")
    out = capsys.readouterr().out
    assert "live_blocks=3" in out and "top_blocks_GiB=9.00,0.33,0.33" in out
    assert dumped[0] in out


def test_decorator_is_transparent_when_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", lambda p: called.append(p))
    for cfg in (None, {}, {"enabled": False}):
        worker = _Worker(cfg)
        with pytest.raises(torch.OutOfMemoryError):
            worker.train(fail=True)
        assert worker.calls == 1
    assert called == []


def test_report_oom_never_raises(monkeypatch, capsys):
    def boom():
        raise RuntimeError("no snapshot here")

    monkeypatch.setattr(torch.cuda.memory, "_snapshot", boom)
    assert ms.report_oom("train", ms.MemorySnapshotConfig(enabled=True)) is None
    assert "dump failed" in capsys.readouterr().out


def test_report_oom_returns_none_when_the_dump_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", _fake_snapshot)

    def bad_dump(path):
        raise OSError("disk full")

    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", bad_dump)
    cfg = ms.MemorySnapshotConfig(enabled=True, directory=str(tmp_path))
    assert ms.report_oom("train", cfg) is None


def test_two_dumps_in_the_same_second_do_not_collide(monkeypatch, tmp_path):
    dumped: list[str] = []
    monkeypatch.setattr(torch.cuda.memory, "_snapshot", _fake_snapshot)
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", lambda p: dumped.append(p))
    cfg = ms.MemorySnapshotConfig(enabled=True, directory=str(tmp_path))
    ms.report_oom("train", cfg)
    ms.report_oom("train", cfg)
    assert len(set(dumped)) == 2


def test_config_defaults_live_on_the_model():
    cfg = ms.MemorySnapshotConfig.model_validate({})
    assert cfg.enabled is False and cfg.directory is None
    assert cfg.max_entries == ms.DEFAULT_MAX_ENTRIES and cfg.record_history is True
