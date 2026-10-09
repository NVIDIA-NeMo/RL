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

from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.weight_sync.remote_vllm_checkpoint_weight_synchronizer import (
    RemoteVllmCheckpointWeightSynchronizer,
)


class _Policy:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def export_hf_checkpoint(self, output_path: str) -> None:
        self.events.append(("export", output_path))
        path = Path(output_path)
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"weights")


class _Generation:
    def __init__(self, checkpoint_dir: Path, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.remote_config = SimpleNamespace(
            refit=SimpleNamespace(checkpoint_dir=str(checkpoint_dir))
        )

    def pause_generation_for_refit(self, *, clear_cache: bool) -> bool:
        self.events.append(("pause", clear_cache))
        return True

    def reload_weights(self, weights_path: str) -> None:
        assert Path(weights_path).is_dir()
        self.events.append(("reload", weights_path))

    def resume_generation_after_refit(self) -> bool:
        self.events.append(("resume", True))
        return True


def test_exports_versioned_checkpoint_then_reloads(tmp_path: Path) -> None:
    events: list[tuple[str, object]] = []
    sync = RemoteVllmCheckpointWeightSynchronizer(
        _Policy(events), _Generation(tmp_path, events)
    )
    sync.init_communicator()

    sync.sync_weights()

    final_path = tmp_path / "version_00000001"
    assert final_path.is_dir()
    assert not (tmp_path / ".version_00000001.tmp").exists()
    assert [event[0] for event in events] == [
        "export",
        "pause",
        "reload",
        "resume",
    ]
    assert events[2] == ("reload", str(final_path))
    assert sync.is_stale is False


def test_reload_failure_does_not_resume(tmp_path: Path) -> None:
    events: list[tuple[str, object]] = []
    generation = _Generation(tmp_path, events)

    def fail_reload(weights_path: str) -> None:
        events.append(("reload", weights_path))
        raise RuntimeError("partial reload")

    generation.reload_weights = fail_reload  # type: ignore[method-assign]
    sync = RemoteVllmCheckpointWeightSynchronizer(_Policy(events), generation)
    sync.init_communicator()

    with pytest.raises(RuntimeError, match="partial reload"):
        sync.sync_weights()

    assert "resume" not in [event[0] for event in events]
    assert sync.is_stale is True


def test_stale_staging_directory_advances_export_version(tmp_path: Path) -> None:
    (tmp_path / ".version_00000003.tmp").mkdir()
    events: list[tuple[str, object]] = []
    sync = RemoteVllmCheckpointWeightSynchronizer(
        _Policy(events), _Generation(tmp_path, events)
    )
    sync.init_communicator()

    sync.sync_weights()

    assert (tmp_path / "version_00000004").is_dir()
