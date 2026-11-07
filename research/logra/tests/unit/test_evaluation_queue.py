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


"""Finalized-checkpoint identity and immutable external-evaluation requests."""

import json
from pathlib import Path

import pytest
from logra.evaluation_queue import enqueue_checkpoint


def prepare(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "evaluation"
    root.mkdir()
    (root / "protocol.json").write_text('{"program":"yi_original"}\n')
    checkpoint = tmp_path / "run/checkpoints/step_100"
    model = checkpoint / "policy/weights/model"
    model.mkdir(parents=True)
    (checkpoint / "training_info.json").write_text('{"step":100}\n')
    (checkpoint / "cua.pt").touch()
    (model / "shard.safetensors").touch()
    return root, checkpoint


def test_a_request_is_idempotent_across_resume(tmp_path: Path) -> None:
    root, checkpoint = prepare(tmp_path)
    path = enqueue_checkpoint(
        root, checkpoint, step=100, resume=None, requesting_run=tmp_path / "first"
    )
    original = path.read_bytes()
    assert (
        enqueue_checkpoint(
            root, checkpoint, step=100, resume=None, requesting_run=tmp_path / "resumed"
        )
        == path
    )
    assert path.read_bytes() == original
    assert len(list((root / "requests").iterdir())) == 1
    assert json.loads(original)["checkpoint"] == str(checkpoint.resolve())


def test_changed_protocol_cannot_relabel_an_existing_point(tmp_path: Path) -> None:
    root, checkpoint = prepare(tmp_path)
    path = enqueue_checkpoint(
        root, checkpoint, step=100, resume=None, requesting_run=tmp_path
    )
    original = path.read_bytes()
    (root / "protocol.json").write_text('{"program":"different"}\n')
    with pytest.raises(ValueError, match="Conflicting"):
        enqueue_checkpoint(
            root, checkpoint, step=100, resume=None, requesting_run=tmp_path
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "missing",
    ["training_info.json", "cua.pt", "policy/weights/model/shard.safetensors"],
)
def test_incomplete_checkpoint_is_not_queued(tmp_path: Path, missing: str) -> None:
    root, checkpoint = prepare(tmp_path)
    (checkpoint / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        enqueue_checkpoint(
            root, checkpoint, step=100, resume=None, requesting_run=tmp_path
        )
    assert not (root / "requests").exists()


def test_wrong_step_is_not_queued(tmp_path: Path) -> None:
    root, checkpoint = prepare(tmp_path)
    with pytest.raises(ValueError, match="finalized"):
        enqueue_checkpoint(
            root, checkpoint, step=50, resume=None, requesting_run=tmp_path
        )
    (checkpoint / "training_info.json").write_text('{"step":99}\n')
    with pytest.raises(ValueError, match="incomplete"):
        enqueue_checkpoint(
            root, checkpoint, step=100, resume=None, requesting_run=tmp_path
        )


def test_resumed_evaluation_uses_the_previous_run_checkpoint(tmp_path: Path) -> None:
    root, checkpoint = prepare(tmp_path)
    new_run = tmp_path / "continuation"
    new_run.mkdir()
    path = enqueue_checkpoint(
        root,
        new_run / "checkpoints/step_100",
        step=100,
        requesting_run=new_run,
        resume=checkpoint,
    )
    request = json.loads(path.read_text())
    assert request["checkpoint"] == str(checkpoint.resolve())
    assert request["requesting_run"] == str(new_run.resolve())
    assert not (new_run / "checkpoints/step_100").exists()


def test_later_missing_checkpoint_never_reuses_the_resume_policy(
    tmp_path: Path,
) -> None:
    root, checkpoint = prepare(tmp_path)
    with pytest.raises(FileNotFoundError):
        enqueue_checkpoint(
            root,
            tmp_path / "continuation/checkpoints/step_150",
            step=150,
            requesting_run=tmp_path / "continuation",
            resume=checkpoint,
        )
    assert not (root / "requests").exists()
