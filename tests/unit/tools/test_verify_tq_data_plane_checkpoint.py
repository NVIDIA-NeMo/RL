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
from typing import Any
from unittest.mock import MagicMock

import pytest
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import backend_config
from tools import verify_tq_data_plane_checkpoint as verifier


class _FakeDataPlaneClient:
    def __init__(
        self,
        checkpoint_state: dict[str, Any],
        *,
        missing_samples: bool = False,
        premature_consumption: bool = False,
    ) -> None:
        self._checkpoint_state = checkpoint_state
        self._missing_samples = missing_samples
        self._premature_consumption = premature_consumption
        self._fields: TensorDict | None = None
        self._consumed: set[str] = set()

    def register_partition(self, **kwargs: Any) -> None:
        pass

    def put_samples(self, *, fields: TensorDict, **kwargs: Any) -> None:
        self._fields = fields.clone()

    def claim_meta(self, *, batch_size: int, **kwargs: Any) -> MagicMock:
        available = [
            sample_id
            for sample_id in verifier.SAMPLE_IDS
            if sample_id not in self._consumed
        ]
        sample_ids = available[:batch_size]
        self._consumed.update(sample_ids)
        return MagicMock(size=len(sample_ids), sample_ids=sample_ids)

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        del checkpoint_dir
        self._checkpoint_state.update(
            {
                "fields": self._fields,
                "consumed": set(self._consumed),
                "metadata": dict(metadata),
            }
        )

    def load_checkpoint(self, checkpoint_dir: Path) -> dict[str, Any]:
        del checkpoint_dir
        self._fields = self._checkpoint_state["fields"].clone()
        self._consumed = set(self._checkpoint_state["consumed"])
        return dict(self._checkpoint_state["metadata"])

    def get_samples(self, **kwargs: Any) -> TensorDict:
        if self._missing_samples:
            raise KeyError("injected missing sample")
        assert self._fields is not None
        return self._fields

    def check_consumption_status(self, *args: Any) -> bool:
        if self._premature_consumption:
            return True
        return len(self._consumed) == len(verifier.SAMPLE_IDS)

    def close(self) -> None:
        pass


def _checkpoint_state(*, schema_version: int | None = None) -> dict[str, Any]:
    return {
        "fields": verifier._expected_fields(),
        "consumed": {verifier.SAMPLE_IDS[0]},
        "metadata": verifier._checkpoint_metadata(
            [verifier.SAMPLE_IDS[0]],
            schema_version=(
                verifier.DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
                if schema_version is None
                else schema_version
            ),
        ),
    }


def test_data_plane_config_uses_nested_simple_backend_config() -> None:
    config = verifier._data_plane_config(num_storage_units=3)

    simple_config = backend_config(config)

    assert simple_config.num_storage_units == 3
    assert simple_config.storage_capacity == 1024


def test_save_load_round_trip_exercises_payload_and_cursor_restore(
    monkeypatch, tmp_path
) -> None:
    checkpoint_state: dict[str, Any] = {}
    clients = iter(
        [
            _FakeDataPlaneClient(checkpoint_state),
            _FakeDataPlaneClient(checkpoint_state),
        ]
    )
    monkeypatch.setattr(
        verifier,
        "build_data_plane_client",
        lambda *args, **kwargs: next(clients),
    )

    verifier._save(tmp_path / "data_plane", num_storage_units=2)
    verifier._load(tmp_path / "data_plane", num_storage_units=2)


@pytest.mark.parametrize(
    ("client_kwargs", "error_type", "match"),
    [
        ({"missing_samples": True}, KeyError, "injected missing sample"),
        (
            {"premature_consumption": True},
            AssertionError,
            "marked every row consumed",
        ),
    ],
)
def test_load_rejects_invalid_restored_state(
    monkeypatch,
    tmp_path,
    client_kwargs,
    error_type,
    match,
) -> None:
    client = _FakeDataPlaneClient(_checkpoint_state(), **client_kwargs)
    monkeypatch.setattr(
        verifier,
        "build_data_plane_client",
        lambda *args, **kwargs: client,
    )

    with pytest.raises(error_type, match=match):
        verifier._load(tmp_path / "data_plane", num_storage_units=2)


def test_load_rejects_schema_mismatch(monkeypatch, tmp_path) -> None:
    client = _FakeDataPlaneClient(_checkpoint_state(schema_version=-1))
    monkeypatch.setattr(
        verifier,
        "build_data_plane_client",
        lambda *args, **kwargs: client,
    )

    with pytest.raises(AssertionError, match="Unexpected data-plane checkpoint schema"):
        verifier._load(tmp_path / "data_plane", num_storage_units=2)


def test_run_child_uses_fresh_process(monkeypatch, tmp_path) -> None:
    run = MagicMock()
    monkeypatch.setattr(verifier.subprocess, "run", run)

    verifier._run_child("load", tmp_path / "step_1", num_storage_units=3)

    command = run.call_args.args[0]
    assert command[0] == verifier.sys.executable
    assert command[2:] == [
        "--phase",
        "load",
        "--checkpoint-dir",
        str(tmp_path / "step_1"),
        "--num-storage-units",
        "3",
    ]
    assert run.call_args.kwargs == {"check": True}


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
