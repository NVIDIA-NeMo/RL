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

from unittest.mock import MagicMock

import pytest

from tools import verify_tq_data_plane_checkpoint as verifier


def test_save_finalizes_by_renaming_parent_bundle(monkeypatch, tmp_path) -> None:
    final_bundle = tmp_path / "step_7"
    expected_staging_bundle = tmp_path / "tmp_step_7"
    save_calls = []

    def fake_save(checkpoint_dir, num_storage_units) -> None:
        save_calls.append((checkpoint_dir, num_storage_units))
        assert checkpoint_dir.parent.is_dir()
        checkpoint_dir.mkdir()
        (checkpoint_dir / "marker").write_text("saved")

    monkeypatch.setattr(verifier, "_save", fake_save)

    verifier._save_and_finalize_bundle(final_bundle, num_storage_units=3)

    assert save_calls == [(expected_staging_bundle / "data_plane", 3)]
    assert not expected_staging_bundle.exists()
    assert (final_bundle / "data_plane" / "marker").read_text() == "saved"


def test_save_refuses_to_replace_final_bundle(monkeypatch, tmp_path) -> None:
    final_bundle = tmp_path / "step_7"
    final_bundle.mkdir()
    save = MagicMock()
    monkeypatch.setattr(verifier, "_save", save)

    with pytest.raises(FileExistsError, match=str(final_bundle)):
        verifier._save_and_finalize_bundle(final_bundle, num_storage_units=1)

    save.assert_not_called()


def test_save_failure_removes_created_staging_bundle(monkeypatch, tmp_path) -> None:
    final_bundle = tmp_path / "step_7"
    staging_bundle = tmp_path / "tmp_step_7"

    def failing_save(checkpoint_dir, num_storage_units) -> None:
        del num_storage_units
        assert checkpoint_dir.parent == staging_bundle
        assert staging_bundle.is_dir()
        raise RuntimeError("injected TQ save failure")

    monkeypatch.setattr(verifier, "_save", failing_save)

    with pytest.raises(RuntimeError, match="injected TQ save failure"):
        verifier._save_and_finalize_bundle(final_bundle, num_storage_units=1)

    assert not staging_bundle.exists()
    assert not final_bundle.exists()
