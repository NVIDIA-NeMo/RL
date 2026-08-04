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

import json
from typing import Any, cast

import pytest

from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
    RolloutSnapshotManifest,
    bootstrap_fingerprint,
    commit_snapshot,
    ensure_bootstrap_anchor,
    prepare_snapshot_paths,
    prune_bootstrap_snapshots,
    resolve_latest_snapshot,
    validate_bootstrap_anchor,
)


class _DumpedConfig:
    def __init__(self, dumped: dict[str, Any]):
        self._dumped = dumped

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self._dumped


def _commit_snapshot(
    anchor,
    *,
    mutation_version: int,
    trainer_version: int = 0,
    fingerprint: str | None = "fingerprint-v1",
):
    tmp_path, final_path, _ = prepare_snapshot_paths(anchor)
    manifest = RolloutSnapshotManifest(
        schema_version=1,
        base_train_step=trainer_version,
        trainer_version=trainer_version,
        current_epoch=2,
        mutation_version=mutation_version,
        bootstrap_fingerprint=fingerprint,
    )
    (tmp_path / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict())
    )
    commit_snapshot(tmp_path, final_path, keep_latest_k=3)
    return final_path


def test_bootstrap_anchor_rejects_different_initial_state(tmp_path):
    anchor = ensure_bootstrap_anchor(tmp_path, fingerprint="fingerprint-v1")
    validate_bootstrap_anchor(anchor, fingerprint="fingerprint-v1")

    with pytest.raises(ValueError, match="does not match"):
        validate_bootstrap_anchor(anchor, fingerprint="fingerprint-v2")


def test_bootstrap_fingerprint_includes_new_semantic_sections_by_default():
    base = {
        "policy": {"model": "model-a"},
        "logger": {"log_dir": "/run/one"},
        "future_semantic_section": {"mode": "a"},
    }
    runtime_changed = {
        **base,
        "logger": {"log_dir": "/run/two"},
    }
    semantic_changed = {
        **base,
        "future_semantic_section": {"mode": "b"},
    }

    fingerprint = bootstrap_fingerprint(cast(Any, _DumpedConfig(base)))
    assert fingerprint == bootstrap_fingerprint(
        cast(Any, _DumpedConfig(runtime_changed))
    )
    assert fingerprint != bootstrap_fingerprint(
        cast(Any, _DumpedConfig(semantic_changed))
    )


def test_prune_bootstrap_snapshots_requires_durable_trainer_checkpoint(tmp_path):
    snapshot_root = tmp_path / "bootstrap" / "rollout_snapshots"
    snapshot_root.mkdir(parents=True)
    (snapshot_root / "snapshot_000001").mkdir()
    durable_anchor = tmp_path / "step_1"

    with pytest.raises(FileNotFoundError, match="durable trainer checkpoint"):
        prune_bootstrap_snapshots(
            tmp_path,
            durable_trainer_checkpoint=durable_anchor,
        )

    assert snapshot_root.is_dir()
    durable_anchor.mkdir()
    assert prune_bootstrap_snapshots(
        tmp_path,
        durable_trainer_checkpoint=durable_anchor,
    )
    assert not snapshot_root.exists()


def test_resolver_selects_latest_compatible_committed_snapshot(tmp_path):
    anchor = ensure_bootstrap_anchor(tmp_path, fingerprint="fingerprint-v1")
    first = _commit_snapshot(anchor, mutation_version=1)
    second = _commit_snapshot(anchor, mutation_version=2)

    resolved = resolve_latest_snapshot(
        anchor,
        expected_train_step=0,
        expected_trainer_version=0,
        expected_bootstrap_fingerprint="fingerprint-v1",
    )

    assert resolved is not None
    assert resolved.path == second
    assert resolved.manifest.mutation_version == 2
    assert first.is_dir()


def test_resolver_falls_back_from_corrupt_newest_snapshot(tmp_path):
    anchor = ensure_bootstrap_anchor(tmp_path, fingerprint="fingerprint-v1")
    first = _commit_snapshot(anchor, mutation_version=1)
    second = _commit_snapshot(anchor, mutation_version=2)
    (second / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text("not-json")

    resolved = resolve_latest_snapshot(
        anchor,
        expected_train_step=0,
        expected_trainer_version=0,
        expected_bootstrap_fingerprint="fingerprint-v1",
    )

    assert resolved is not None
    assert resolved.path == first


def test_resolver_fails_when_no_committed_snapshot_matches_anchor(tmp_path):
    anchor = ensure_bootstrap_anchor(tmp_path, fingerprint="fingerprint-v1")
    _commit_snapshot(anchor, mutation_version=1, fingerprint="different")

    with pytest.raises(ValueError, match="trainer-anchor mismatch"):
        resolve_latest_snapshot(
            anchor,
            expected_train_step=0,
            expected_trainer_version=0,
            expected_bootstrap_fingerprint="fingerprint-v1",
        )
