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
"""Validate actor extras from source projects and installed distribution metadata."""

from email.message import Message
from pathlib import Path

import pytest

from nemo_rl.distributed import ray_actor_environment_registry as registry


def test_source_checkout_uses_working_tree_extras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "nemo_rl").mkdir()
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "nemo-rl"\n[project.optional-dependencies]\ncustom = []\n'
    )
    monkeypatch.setattr(registry, "git_root", str(tmp_path))
    monkeypatch.setattr(registry, "ACTOR_ENVIRONMENTS", {"example.Actor": ("custom",)})
    monkeypatch.setattr(
        registry,
        "metadata",
        lambda _: pytest.fail("Source checkouts must use their project"),
    )
    registry._reject_undeclared_extras()
    project.write_text(
        '[project]\nname = "nemo-rl"\n[project.optional-dependencies]\nother = []\n'
    )
    with pytest.raises(ValueError, match="pyproject.toml"):
        registry._reject_undeclared_extras()


def test_installed_wheel_uses_normalized_provides_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An unrelated pyproject in site-packages is not the distribution's source project.
    (tmp_path / "pyproject.toml").write_text(
        "[project.optional-dependencies]\nunrelated = []\n"
    )
    message = Message()
    message["Provides-Extra"] = "nemo-gym"
    monkeypatch.setattr(registry, "git_root", str(tmp_path))
    monkeypatch.setattr(
        registry, "metadata", lambda name: message if name == "nemo-rl" else None
    )
    monkeypatch.setattr(
        registry, "ACTOR_ENVIRONMENTS", {"example.Actor": ("nemo_gym",)}
    )
    registry._reject_undeclared_extras()
    monkeypatch.setattr(registry, "ACTOR_ENVIRONMENTS", {"example.Actor": ("typo",)})
    with pytest.raises(ValueError, match="Provides-Extra metadata"):
        registry._reject_undeclared_extras()


def test_installed_wheel_without_extras_rejects_requested_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(registry, "git_root", str(tmp_path))
    monkeypatch.setattr(registry, "metadata", lambda _: Message())
    monkeypatch.setattr(registry, "ACTOR_ENVIRONMENTS", {"example.Actor": ("vllm",)})
    with pytest.raises(ValueError, match="vllm"):
        registry._reject_undeclared_extras()
