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
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, call

import pytest

from nemo_rl.algorithms.single_controller_utils import rollout_checkpoint
from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
    RolloutSnapshotManifest,
    bootstrap_fingerprint,
    commit_snapshot,
    ensure_bootstrap_anchor,
    prepare_snapshot_paths,
    prune_bootstrap_snapshots,
    reset_bootstrap_anchor,
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


def test_bootstrap_fingerprint_ignores_non_recovery_configuration() -> None:
    base = {
        "policy": {
            "model_name": "model-a",
            "optimizer": {"lr": 1.0e-6},
            "generation": {
                "backend": "vllm",
                "temperature": 1.0,
                "colocated": {"enabled": True, "resources": {"gpus": 8}},
                "vllm_cfg": {
                    "kv_cache_dtype": "auto",
                    "precision": "bfloat16",
                    "skip_tokenizer_init": False,
                },
            },
        },
        "data": {
            "train": [{"data_path": "/datasets/train.jsonl"}],
            "num_workers": 4,
        },
        "grpo": {
            "num_generations_per_prompt": 4,
            "max_num_steps": 10,
            "batch_multiplier": 1,
            "use_dynamic_sampling": False,
            "reward_shaping": {"enabled": False},
            "reward_scaling": {"enabled": False},
        },
        "loss_fn": {
            "reference_policy_kl_penalty": 0.01,
            "use_kl_in_reward": False,
        },
        "reward_penalties": {"penalize_unwanted_tokens": False},
        "token_capture": {
            "enabled": True,
            "staging_partition": "rollout_staging",
            "on_capture_failure": "continue",
        },
        "cluster": {"num_nodes": 2},
        "logger": {"log_dir": "/run/one"},
    }
    compatible_changed = {
        **base,
        "policy": {
            **base["policy"],
            "optimizer": {"lr": 5.0e-7},
            "generation": {
                **base["policy"]["generation"],
                "colocated": {"enabled": False, "resources": {"gpus": 16}},
                "vllm_cfg": {
                    "kv_cache_dtype": "fp8",
                    "precision": "float16",
                    "skip_tokenizer_init": True,
                },
            },
        },
        "data": {**base["data"], "num_workers": 16},
        "grpo": {
            **base["grpo"],
            "max_num_steps": 100,
            "batch_multiplier": 2,
            "use_dynamic_sampling": True,
            "reward_shaping": {"enabled": True},
            "reward_scaling": {"enabled": True},
        },
        "loss_fn": {
            "reference_policy_kl_penalty": 0.1,
            "use_kl_in_reward": True,
        },
        "reward_penalties": {"penalize_unwanted_tokens": True},
        "token_capture": {
            **base["token_capture"],
            "on_capture_failure": "abort",
        },
        "cluster": {"num_nodes": 8},
        "logger": {"log_dir": "/run/two"},
    }

    fingerprint = bootstrap_fingerprint(cast(Any, _DumpedConfig(base)))
    assert fingerprint == bootstrap_fingerprint(
        cast(Any, _DumpedConfig(compatible_changed))
    )


@pytest.mark.parametrize(
    ("section", "changed"),
    [
        ("policy", {"model_name": "model-b"}),
        ("data", {"train": [{"data_path": "/datasets/other.jsonl"}]}),
        ("grpo", {"num_generations_per_prompt": 8}),
        ("token_capture", {"mixed_weight_version_policy": "reject"}),
        (
            "async_rl",
            {"sampler": {"name": "windowed", "max_staleness_versions": 2}},
        ),
    ],
)
def test_bootstrap_fingerprint_rejects_rollout_semantic_changes(
    section: str,
    changed: dict[str, Any],
) -> None:
    base = {
        "policy": {
            "model_name": "model-a",
            "tokenizer": {"name": "tokenizer-a"},
            "generation": {"backend": "vllm", "temperature": 1.0},
        },
        "data": {"train": [{"data_path": "/datasets/train.jsonl"}]},
        "grpo": {"num_generations_per_prompt": 4},
        "token_capture": {
            "enabled": True,
            "mixed_weight_version_policy": "allow",
        },
        "async_rl": {
            "sampler": {"name": "windowed", "max_staleness_versions": 1}
        },
    }
    modified = {**base, section: {**base[section], **changed}}

    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) != (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(modified)))
    )


def test_bootstrap_fingerprint_rejects_generation_semantic_changes() -> None:
    base = {
        "policy": {
            "model_name": "model-a",
            "generation": {
                "backend": "vllm",
                "temperature": 1.0,
                "vllm_cfg": {"max_model_len": 4096},
            },
        }
    }
    sampling_changed = {
        **base,
        "policy": {
            **base["policy"],
            "generation": {
                **base["policy"]["generation"],
                "temperature": 0.5,
            },
        },
    }
    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) != (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(sampling_changed)))
    )

    context_changed = {
        **base,
        "policy": {
            **base["policy"],
            "generation": {
                **base["policy"]["generation"],
                "vllm_cfg": {"max_model_len": 8192},
            },
        },
    }
    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) != (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(context_changed)))
    )


def test_bootstrap_fingerprint_ignores_nested_gym_log_directory() -> None:
    base = {
        "policy": {"model": "model-a"},
        "env": {
            "should_use_nemo_gym": True,
            "nemo_gym": {
                "nemo_gym_log_dir": "/run/one/nemo_gym",
                "should_log_nemo_gym_responses": True,
                "policy_model": {"temperature": 1.0},
                "agent": {"concurrency": 16, "max_turns": 20},
            },
        },
    }
    runtime_changed = {
        **base,
        "env": {
            **base["env"],
            "nemo_gym": {
                **base["env"]["nemo_gym"],
                "nemo_gym_log_dir": "/run/two/nemo_gym",
                "should_log_nemo_gym_responses": False,
                "agent": {"concurrency": 64, "max_turns": 20},
            },
        },
    }
    semantic_changed = {
        **base,
        "env": {
            **base["env"],
            "nemo_gym": {
                **base["env"]["nemo_gym"],
                "policy_model": {"temperature": 0.5},
            },
        },
    }

    fingerprint = bootstrap_fingerprint(cast(Any, _DumpedConfig(base)))
    assert fingerprint == bootstrap_fingerprint(
        cast(Any, _DumpedConfig(runtime_changed))
    )
    assert fingerprint != bootstrap_fingerprint(
        cast(Any, _DumpedConfig(semantic_changed))
    )
    assert base["env"]["nemo_gym"]["nemo_gym_log_dir"] == "/run/one/nemo_gym"


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


def test_reset_bootstrap_anchor_discards_skipped_snapshot_lineage(
    tmp_path: Path,
) -> None:
    anchor = ensure_bootstrap_anchor(tmp_path, fingerprint="old-fingerprint")
    snapshot_root = anchor / "rollout_snapshots"
    (snapshot_root / "snapshot_000001").mkdir(parents=True)

    reset = reset_bootstrap_anchor(tmp_path, fingerprint="new-fingerprint")

    assert reset == anchor
    assert not snapshot_root.exists()
    validate_bootstrap_anchor(anchor, fingerprint="new-fingerprint")


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


def test_commit_snapshot_flushes_payload_before_publication(tmp_path, monkeypatch):
    anchor = tmp_path / "step_1"
    anchor.mkdir()
    tmp_snapshot, final_snapshot, _ = prepare_snapshot_paths(anchor)
    (tmp_snapshot / "payload").write_text("payload")
    fsync_tree = Mock()
    fsync_file = Mock()
    fsync_directory = Mock()
    monkeypatch.setattr(rollout_checkpoint, "_fsync_tree", fsync_tree)
    monkeypatch.setattr(rollout_checkpoint, "_fsync_file", fsync_file)
    monkeypatch.setattr(rollout_checkpoint, "_fsync_directory", fsync_directory)

    commit_snapshot(tmp_snapshot, final_snapshot, keep_latest_k=1)

    fsync_tree.assert_called_once_with(tmp_snapshot)
    assert fsync_file.call_args_list == [
        call(tmp_snapshot / "COMMITTED"),
        call(anchor / "rollout_snapshots" / "LATEST.tmp"),
    ]
    assert fsync_directory.call_args_list[:2] == [
        call(tmp_snapshot),
        call(anchor / "rollout_snapshots"),
    ]
    assert (final_snapshot / "COMMITTED").is_file()
    assert (
        anchor / "rollout_snapshots" / "LATEST"
    ).read_text().strip() == final_snapshot.name
