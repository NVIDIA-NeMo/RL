# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest
import torch

from nemo_rl.models import vision_parity_dump


@pytest.mark.parametrize("disabled", ["", " ", "\t"])
def test_explicit_empty_dump_override_skips_tensor_copies_and_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, disabled: str
) -> None:
    monkeypatch.setenv("NEMO_RL_VISION_PARITY_DUMP_DIR", disabled)
    monkeypatch.setenv("NEMO_RL_TMPE_OUTLIER_DUMP_PATH", str(tmp_path / "outliers"))

    def unexpected_digest(_tensor: torch.Tensor) -> str:
        raise AssertionError("Disabled vision dumping must not copy or hash tensors")

    monkeypatch.setattr(vision_parity_dump, "_tensor_digest", unexpected_digest)
    assert (
        vision_parity_dump.dump_vision_features(
            "generation", torch.ones(1), torch.ones(1), [[1, 1]]
        )
        is None
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("explicit", [False, True])
def test_enabled_dump_uses_explicit_path_or_legacy_outlier_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit: bool
) -> None:
    monkeypatch.delenv("NEMO_RL_VISION_PARITY_DUMP_DIR", raising=False)
    monkeypatch.setenv("NEMO_RL_TMPE_OUTLIER_DUMP_PATH", str(tmp_path / "outliers"))
    expected = tmp_path / "vision_parity"
    if explicit:
        expected = tmp_path / "chosen-dump-dir"
        monkeypatch.setenv("NEMO_RL_VISION_PARITY_DUMP_DIR", str(expected))
    pixels = torch.ones(1, 3, 2, 2)
    features = torch.ones(1, 4)
    target = vision_parity_dump.dump_vision_features(
        "generation", pixels, features, [[2, 2]]
    )
    assert target is not None and target.parent == expected
    payload = torch.load(target, weights_only=True)
    torch.testing.assert_close(payload["pixels"], pixels)
    torch.testing.assert_close(payload["features"], features)


def test_dump_is_disabled_when_neither_environment_variable_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEMO_RL_VISION_PARITY_DUMP_DIR", raising=False)
    monkeypatch.delenv("NEMO_RL_TMPE_OUTLIER_DUMP_PATH", raising=False)
    assert vision_parity_dump._dump_root() is None
