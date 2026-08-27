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
from __future__ import annotations

import threading

import pytest
import torch

from nemo_rl.data import train_data_artifacts as artifacts


def _tensor_payload() -> dict[str, torch.Tensor]:
    shared_rewards = torch.arange(3, dtype=torch.float32)
    return {
        "idx": torch.arange(3, dtype=torch.int64),
        "token_ids": torch.arange(24, dtype=torch.int64).reshape(3, 8)[:, ::2],
        "token_loss_mask": torch.tensor(
            [[True, False], [False, True], [True, True]],
            dtype=torch.bool,
        ),
        "advantages": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "rewards": shared_rewards,
        "filtered_rewards": shared_rewards,
    }


def test_train_data_artifact_round_trip(tmp_path):
    non_tensor_data = {
        "content": ["first", "second", "third"],
        "agent_ref": [{"name": "agent"}] * 3,
    }
    paths = artifacts.save_train_data_artifacts(
        base_dir=tmp_path,
        step=7,
        num_samples=3,
        non_tensor_data=non_tensor_data,
        tensors=_tensor_payload(),
    )

    assert paths.main.name == "train_data_step7.pt"
    assert paths.tensors.name == "train_data_step7.tensors.safetensors"
    assert paths.main.is_file()
    assert paths.tensors.is_file()
    assert not list(tmp_path.glob("*.tmp.*"))

    main_artifact = torch.load(paths.main, map_location="cpu", weights_only=False)
    assert main_artifact["format"] == artifacts.TRAIN_DATA_ARTIFACT_FORMAT
    assert main_artifact["tensor_sidecar"]["filename"] == paths.tensors.name
    assert (
        main_artifact["tensor_sidecar"]["transaction_id"]
        == main_artifact["transaction_id"]
    )

    loaded = artifacts.load_train_data_artifacts(paths.main)
    assert loaded["step"] == 7
    assert loaded["num_samples"] == 3
    assert loaded["data"]["content"] == non_tensor_data["content"]
    assert loaded["data"]["agent_ref"] == non_tensor_data["agent_ref"]
    for name, expected in _tensor_payload().items():
        actual = loaded["data"][name]
        assert actual.device.type == "cpu"
        assert actual.is_contiguous()
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected)


def test_train_data_artifact_detects_transaction_mismatch(tmp_path):
    paths = artifacts.save_train_data_artifacts(
        base_dir=tmp_path,
        step=1,
        num_samples=3,
        non_tensor_data={"content": ["a", "b", "c"]},
        tensors=_tensor_payload(),
    )
    main_artifact = torch.load(paths.main, map_location="cpu", weights_only=False)
    main_artifact["transaction_id"] = "different-transaction"
    torch.save(main_artifact, paths.main)

    with pytest.raises(ValueError, match="transaction ids differ"):
        artifacts.load_train_data_artifacts(paths.main)


@pytest.mark.parametrize(
    ("step", "num_samples", "error_type"),
    [
        (1.5, 3, TypeError),
        (True, 3, TypeError),
        (0, 3, ValueError),
        (1, 3.5, TypeError),
        (1, True, TypeError),
        (1, -1, ValueError),
    ],
)
def test_train_data_artifact_rejects_invalid_counts(
    tmp_path, step, num_samples, error_type
):
    with pytest.raises(error_type):
        artifacts.save_train_data_artifacts(
            base_dir=tmp_path,
            step=step,
            num_samples=num_samples,
            non_tensor_data={"content": ["a", "b", "c"]},
            tensors=_tensor_payload(),
        )


def test_train_data_artifact_failure_cleans_temporary_files(tmp_path, monkeypatch):
    def fail_main_save(*args, **kwargs):
        raise OSError("injected .pt failure")

    monkeypatch.setattr(artifacts.torch, "save", fail_main_save)
    with pytest.raises(OSError, match="injected .pt failure"):
        artifacts.save_train_data_artifacts(
            base_dir=tmp_path,
            step=2,
            num_samples=3,
            non_tensor_data={"content": ["a", "b", "c"]},
            tensors=_tensor_payload(),
        )

    assert not (tmp_path / "train_data_step2.pt").exists()
    assert not (tmp_path / "train_data_step2.tensors.safetensors").exists()
    assert not list(tmp_path.glob("*.tmp.*"))


def test_async_train_data_writer_can_be_polled(tmp_path):
    save_started = threading.Event()
    allow_save = threading.Event()

    def blocking_save(**kwargs):
        save_started.set()
        assert allow_save.wait(timeout=5)
        return artifacts.train_data_artifact_paths(tmp_path, kwargs["step"])

    writer = artifacts.AsyncTrainDataArtifactWriter(blocking_save)
    writer.start(
        step=4,
        num_samples=3,
        non_tensor_data={"content": ["a", "b", "c"]},
        tensors=_tensor_payload(),
    )

    assert save_started.wait(timeout=5)
    assert writer.finish(wait=False) is False
    allow_save.set()
    assert writer.finish(wait=True) is True


def test_async_train_data_writer_normalizes_on_writer_thread(tmp_path, monkeypatch):
    caller_thread_id = threading.get_ident()
    normalization_thread_ids = []
    normalize = artifacts.normalize_train_data_tensors

    def record_normalization_thread(tensors):
        normalization_thread_ids.append(threading.get_ident())
        return normalize(tensors)

    def save_in_log_dir(**kwargs):
        return artifacts.save_train_data_artifacts(base_dir=tmp_path, **kwargs)

    monkeypatch.setattr(
        artifacts,
        "normalize_train_data_tensors",
        record_normalization_thread,
    )
    writer = artifacts.AsyncTrainDataArtifactWriter(save_in_log_dir)
    writer.start(
        step=6,
        num_samples=3,
        non_tensor_data={"content": ["a", "b", "c"]},
        tensors=_tensor_payload(),
    )
    assert writer.finish(wait=True) is True

    assert normalization_thread_ids
    assert set(normalization_thread_ids).isdisjoint({caller_thread_id})


def test_async_train_data_writer_surfaces_save_error(tmp_path):
    def fail_save(**kwargs):
        raise OSError("injected sidecar failure")

    writer = artifacts.AsyncTrainDataArtifactWriter(fail_save)
    writer.start(
        step=5,
        num_samples=3,
        non_tensor_data={"content": ["a", "b", "c"]},
        tensors=_tensor_payload(),
    )

    with pytest.raises(RuntimeError, match="artifact save failed") as exc_info:
        writer.finish(wait=True)
    assert isinstance(exc_info.value.__cause__, OSError)
