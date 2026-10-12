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
from nemo_rl.weight_sync.hf_checkpoint_config import HfCheckpointRefitConfig

import pytest

from nemo_rl.weight_sync.factory import create_weight_synchronizer

from nemo_rl.weight_sync.hf_checkpoint_weight_synchronizer import (
    HfCheckpointWeightSynchronizer,
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
    sync = HfCheckpointWeightSynchronizer(
        _Policy(events),
        _Generation(tmp_path, events),
        HfCheckpointRefitConfig(checkpoint_dir=str(tmp_path)),
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
    sync = HfCheckpointWeightSynchronizer(
        _Policy(events),
        generation,
        HfCheckpointRefitConfig(checkpoint_dir=str(tmp_path)),
    )
    sync.init_communicator()

    with pytest.raises(RuntimeError, match="partial reload"):
        sync.sync_weights()

    assert "resume" not in [event[0] for event in events]
    assert sync.is_stale is True


def test_stale_staging_directory_advances_export_version(tmp_path: Path) -> None:
    (tmp_path / ".version_00000003.tmp").mkdir()
    events: list[tuple[str, object]] = []
    sync = HfCheckpointWeightSynchronizer(
        _Policy(events),
        _Generation(tmp_path, events),
        HfCheckpointRefitConfig(checkpoint_dir=str(tmp_path)),
    )
    sync.init_communicator()

    sync.sync_weights()

    assert (tmp_path / "version_00000004").is_dir()


def test_each_refit_publishes_a_new_immutable_version(tmp_path: Path) -> None:
    events: list[tuple[str, object]] = []
    sync = HfCheckpointWeightSynchronizer(
        _Policy(events),
        _Generation(tmp_path, events),
        HfCheckpointRefitConfig(checkpoint_dir=str(tmp_path)),
    )
    sync.init_communicator()
    sync.sync_weights()
    sync.sync_weights()
    assert (tmp_path / "version_00000001/model.safetensors").exists()
    assert (tmp_path / "version_00000002/model.safetensors").exists()
    assert [path for event, path in events if event == "reload"] == [
        str(tmp_path / "version_00000001"),
        str(tmp_path / "version_00000002"),
    ]


def test_relative_checkpoint_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute shared path"):
        HfCheckpointRefitConfig(checkpoint_dir="relative/exports")


def test_export_failure_never_reloads_or_resumes(tmp_path: Path) -> None:
    events: list[tuple[str, object]] = []
    policy = _Policy(events)

    def fail_export(output_path: str) -> None:
        raise RuntimeError("export failed")

    policy.export_hf_checkpoint = fail_export
    sync = HfCheckpointWeightSynchronizer(
        policy,
        _Generation(tmp_path, events),
        HfCheckpointRefitConfig(checkpoint_dir=str(tmp_path)),
    )
    sync.init_communicator()
    with pytest.raises(RuntimeError, match="export failed"):
        sync.sync_weights()
    assert events == []
    assert sync.is_stale


def test_factory_selects_hf_transport_without_collective_clusters(
    tmp_path: Path,
) -> None:
    policy = SimpleNamespace(cfg={"megatron_cfg": {"enabled": True}})
    generation = SimpleNamespace(
        cfg={
            "refit_transport": "hf_checkpoint",
            "refit_cfg": {"hf_checkpoint": {"checkpoint_dir": str(tmp_path)}},
            "vllm_cfg": {"async_engine": True},
        }
    )
    sync = create_weight_synchronizer(policy, generation, "vllm", False)
    assert isinstance(sync, HfCheckpointWeightSynchronizer)


@pytest.mark.parametrize(
    "colocated,megatron,async_engine,error",
    [
        (True, True, True, ValueError),
        (False, False, True, NotImplementedError),
        (False, True, False, ValueError),
    ],
)
def test_factory_rejects_unsupported_hf_topologies(
    tmp_path: Path,
    colocated: bool,
    megatron: bool,
    async_engine: bool,
    error: type[Exception],
) -> None:
    policy = SimpleNamespace(cfg={"megatron_cfg": {"enabled": megatron}})
    generation = SimpleNamespace(
        cfg={
            "refit_transport": "hf_checkpoint",
            "refit_cfg": {"hf_checkpoint": {"checkpoint_dir": str(tmp_path)}},
            "vllm_cfg": {"async_engine": async_engine},
        }
    )
    with pytest.raises(error):
        create_weight_synchronizer(policy, generation, "vllm", colocated)
