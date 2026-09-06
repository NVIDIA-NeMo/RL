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
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
    RolloutCheckpointConfig,
    TokenCaptureConfig,
)
from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    BOOTSTRAP_COMPATIBILITY_SCHEMA_VERSION,
    BOOTSTRAP_MANIFEST_FILENAME,
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
    ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
    BootstrapCompatibilityIdentity,
    RolloutSnapshotManifest,
    bootstrap_compatibility_identity,
    bootstrap_fingerprint,
    commit_snapshot,
    ensure_bootstrap_anchor,
    prepare_snapshot_paths,
    prune_bootstrap_snapshots,
    resolve_latest_snapshot,
    validate_bootstrap_anchor,
)
from nemo_rl.data import DataConfig
from nemo_rl.models.generation.vllm.config import VllmConfig, VllmSpecificArgs
from nemo_rl.models.policy import PolicyConfig


class _DumpedConfig:
    def __init__(self, dumped: dict[str, Any]):
        self._dumped = dumped

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self._dumped


def _test_bootstrap_identity(label: str = "v1") -> BootstrapCompatibilityIdentity:
    return BootstrapCompatibilityIdentity(
        schema_version=BOOTSTRAP_COMPATIBILITY_SCHEMA_VERSION,
        excluded_paths=(),
        config={"test_identity": label},
    )


def _test_bootstrap_fingerprint(label: str = "v1") -> str:
    return _test_bootstrap_identity(label).fingerprint()


def _ensure_test_bootstrap_anchor(tmp_path: Path, label: str = "v1") -> Path:
    return ensure_bootstrap_anchor(
        tmp_path,
        identity=_test_bootstrap_identity(label),
    )


def _commit_snapshot(
    anchor,
    *,
    mutation_version: int,
    trainer_version: int = 0,
    fingerprint: str | None = None,
):
    if fingerprint is None:
        fingerprint = _test_bootstrap_fingerprint()
    tmp_path, final_path, _ = prepare_snapshot_paths(anchor)
    manifest = RolloutSnapshotManifest(
        schema_version=ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        base_train_step=trainer_version,
        trainer_version=trainer_version,
        current_epoch=2,
        sampler_dispatch_index=trainer_version + 1,
        mutation_version=mutation_version,
        rolled_back_train_group_count=0,
        bootstrap_fingerprint=fingerprint,
    )
    (tmp_path / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict())
    )
    commit_snapshot(tmp_path, final_path, keep_latest_k=3)
    return final_path


def test_bootstrap_anchor_rejects_different_initial_state(tmp_path):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    assert _ensure_test_bootstrap_anchor(tmp_path) == anchor
    manifest = json.loads((anchor / BOOTSTRAP_MANIFEST_FILENAME).read_text())
    assert manifest["bootstrap_identity"] == _test_bootstrap_identity().to_dict()

    with pytest.raises(
        ValueError,
        match="Existing checkpoint state was not modified",
    ) as error:
        _ensure_test_bootstrap_anchor(tmp_path, "v2")

    assert "test_identity: 'v1' -> 'v2'" in str(error.value)
    assert "bootstrap_fingerprint" not in str(error.value)


def test_validate_bootstrap_anchor_is_read_only(tmp_path: Path) -> None:
    identity = _test_bootstrap_identity()
    anchor = ensure_bootstrap_anchor(tmp_path, identity=identity)
    snapshot = anchor / "rollout_snapshots" / "snapshot_000001"
    snapshot.mkdir(parents=True)
    payload = snapshot / "payload"
    payload.write_text("preserve me")

    before = {
        path.relative_to(anchor): path.read_bytes()
        for path in anchor.rglob("*")
        if path.is_file()
    }
    validate_bootstrap_anchor(anchor, identity=identity)
    after = {
        path.relative_to(anchor): path.read_bytes()
        for path in anchor.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_bootstrap_anchor_mismatch_names_redacted_config_paths(tmp_path: Path) -> None:
    original = _DumpedConfig(
        {
            "policy": {"generation": {"temperature": 1.0}},
            "env": {"service": {"api_key": "secret-one"}},
            "custom_algo": {"service_token": "custom-secret-one"},
        }
    )
    changed = _DumpedConfig(
        {
            "policy": {"generation": {"temperature": 0.7}},
            "env": {"service": {"api_key": "secret-two"}},
            "custom_algo": {"service_token": "custom-secret-two"},
        }
    )
    anchor = ensure_bootstrap_anchor(
        tmp_path,
        identity=bootstrap_compatibility_identity(cast(Any, original)),
    )

    with pytest.raises(ValueError) as error:
        validate_bootstrap_anchor(
            anchor,
            identity=bootstrap_compatibility_identity(cast(Any, changed)),
        )

    message = str(error.value)
    assert "policy.generation.temperature: 1.0 -> 0.7" in message
    assert "secret-one" not in message
    assert "secret-two" not in message
    manifest = (anchor / BOOTSTRAP_MANIFEST_FILENAME).read_text()
    assert "secret-one" not in manifest
    assert "custom-secret-one" not in manifest


def test_bootstrap_fingerprint_ignores_default_operational_paths() -> None:
    base = {
        "policy": {
            "model_name": "model-a",
            "optimizer": {"lr": 1.0e-6},
            "generation": {
                "backend": "vllm",
                "temperature": 1.0,
                "colocated": {"enabled": True, "resources": {"gpus": 8}},
                "port_range_low": 3000,
                "port_range_high": 4000,
            },
        },
        "data": {
            "train": [{"data_path": "/datasets/train.jsonl"}],
            "num_workers": 4,
        },
        "grpo": {"num_generations_per_prompt": 4},
        "token_capture": {
            "capture_dir": "/run/one/capture",
            "control_auth_token": "secret-one",
            "control_timeout_s": 60.0,
            "enabled": True,
            "num_reassembler_workers": 2,
            "on_capture_failure": "continue",
            "staging_partition": "rollout_staging",
        },
        "async_rl": {
            "sampler": {"name": "windowed", "max_staleness_versions": 1},
            "stall_watchdog": {"interval_s": 30},
        },
        "checkpointing": {"checkpoint_dir": "/run/one/checkpoints"},
        "rollout_checkpointing": {"snapshot_attempt_interval_s": 120},
        "cluster": {"num_nodes": 2},
        "logger": {"log_dir": "/run/one"},
    }
    operationally_changed = {
        **base,
        "policy": {
            **base["policy"],
            "optimizer": {"lr": 5.0e-7},
            "generation": {
                **base["policy"]["generation"],
                "colocated": {"enabled": False, "resources": {"gpus": 16}},
                "port_range_low": 5000,
                "port_range_high": 6000,
            },
        },
        "data": {**base["data"], "num_workers": 16},
        "token_capture": {
            **base["token_capture"],
            "capture_dir": "/run/two/capture",
            "control_auth_token": "secret-two",
            "control_timeout_s": 15.0,
            "num_reassembler_workers": 8,
        },
        "async_rl": {
            **base["async_rl"],
            "stall_watchdog": {"interval_s": 5},
        },
        "checkpointing": {"checkpoint_dir": "/run/two/checkpoints"},
        "rollout_checkpointing": {"snapshot_attempt_interval_s": 300},
        "cluster": {"num_nodes": 8},
        "logger": {"log_dir": "/run/two"},
    }

    fingerprint = bootstrap_fingerprint(cast(Any, _DumpedConfig(base)))
    assert fingerprint == bootstrap_fingerprint(
        cast(Any, _DumpedConfig(operationally_changed))
    )


def test_bootstrap_fingerprint_includes_unknown_config_by_default() -> None:
    base = {"custom_algo": {"semantic_setting": "one"}}
    changed = {"custom_algo": {"semantic_setting": "two"}}

    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) != (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(changed)))
    )


def test_bootstrap_fingerprint_honors_extra_excluded_dotpaths() -> None:
    base = {
        "custom_algo": {
            "semantic_setting": "same",
            "runtime": {"endpoint": "host-one", "port": 1234},
        },
        "rollout_checkpointing": {
            "extra_fingerprint_excluded_paths": ["custom_algo.runtime"]
        },
    }
    runtime_changed = {
        **base,
        "custom_algo": {
            **base["custom_algo"],
            "runtime": {"endpoint": "host-two", "port": 5678},
        },
    }
    semantic_changed = {
        **base,
        "custom_algo": {
            **base["custom_algo"],
            "semantic_setting": "different",
        },
    }

    fingerprint = bootstrap_fingerprint(cast(Any, _DumpedConfig(base)))
    assert fingerprint == bootstrap_fingerprint(
        cast(Any, _DumpedConfig(runtime_changed))
    )
    assert fingerprint != bootstrap_fingerprint(
        cast(Any, _DumpedConfig(semantic_changed))
    )
    identity = bootstrap_compatibility_identity(cast(Any, _DumpedConfig(base)))
    assert identity.config["custom_algo"] == {"semantic_setting": "same"}
    assert "custom_algo.runtime" in identity.excluded_paths


def test_bootstrap_fingerprint_extra_excluded_dotpaths_support_lists() -> None:
    base = {
        "custom_algo": {
            "workers": [
                {"name": "one", "log_dir": "/run/one"},
                {"name": "two", "log_dir": "/run/two"},
            ]
        },
        "rollout_checkpointing": {
            "extra_fingerprint_excluded_paths": ["custom_algo.workers.*.log_dir"]
        },
    }
    changed = {
        **base,
        "custom_algo": {
            "workers": [
                {"name": "one", "log_dir": "/other/one"},
                {"name": "two", "log_dir": "/other/two"},
            ]
        },
    }

    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) == (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(changed)))
    )


@pytest.mark.parametrize(
    "path",
    ["", " custom.path", "custom.path ", ".x", "x.", "x..y", "*", "**"],
)
def test_rollout_checkpoint_config_rejects_invalid_extra_excluded_path(
    path: str,
) -> None:
    with pytest.raises(ValueError, match="extra_fingerprint_excluded_paths"):
        RolloutCheckpointConfig(extra_fingerprint_excluded_paths=[path])


def test_builtin_fingerprint_exclusions_reference_declared_config_fields() -> None:
    """Keep typed portions of the built-in denylist from silently going stale."""

    def _fields(schema: Any) -> set[str]:
        model_fields = getattr(schema, "model_fields", None)
        if model_fields is not None:
            return set(model_fields)
        return set(schema.__annotations__)

    schemas_by_prefix = {
        (): _fields(MasterConfig),
        ("async_rl",): _fields(AsyncRLConfig),
        ("data",): _fields(DataConfig),
        ("policy",): _fields(PolicyConfig),
        ("policy", "generation"): _fields(VllmConfig),
        ("policy", "generation", "vllm_cfg"): _fields(VllmSpecificArgs),
        ("rollout_checkpointing",): _fields(RolloutCheckpointConfig),
        ("token_capture",): _fields(TokenCaptureConfig),
    }
    for excluded_path in rollout_checkpoint._BOOTSTRAP_FINGERPRINT_EXCLUDED_PATHS:
        segments = tuple(excluded_path.split("."))
        for prefix, fields in schemas_by_prefix.items():
            if segments[: len(prefix)] != prefix or len(segments) == len(prefix):
                continue
            next_segment = segments[len(prefix)]
            if not any(character in next_segment for character in "*?["):
                assert next_segment in fields, (
                    f"bootstrap fingerprint exclusion {excluded_path!r} refers to "
                    f"unknown config field {'.'.join((*prefix, next_segment))!r}"
                )


@pytest.mark.parametrize(
    ("section", "changed"),
    [
        ("policy", {"model_name": "model-b"}),
        ("data", {"train": [{"data_path": "/datasets/other.jsonl"}]}),
        ("grpo", {"num_generations_per_prompt": 8}),
        ("token_capture", {"mixed_weight_version_policy": "reject"}),
        ("token_capture", {"defer_routed_experts_to_policy": True}),
        ("token_capture", {"on_capture_failure": "abort"}),
        (
            "async_rl",
            {"sampler": {"name": "windowed", "max_staleness_versions": 2}},
        ),
        ("rollout_recovery", {"default_granularity": "prompt_group"}),
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
        "async_rl": {"sampler": {"name": "windowed", "max_staleness_versions": 1}},
        "rollout_recovery": {"default_granularity": "sibling"},
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

    blocked_tokens_changed = {
        **base,
        "policy": {
            **base["policy"],
            "generation": {
                **base["policy"]["generation"],
                "bad_words": ["forbidden"],
            },
        },
    }
    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) != (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(blocked_tokens_changed)))
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


def test_bootstrap_fingerprint_ignores_nemo_gym_service_routing() -> None:
    base = {
        "env": {
            "nemo_gym": {
                "genrm_model": {
                    "responses_api_models": {
                        "genrm_model": {
                            "base_url": "http://genrm-one/v1",
                            "model": "genrm-model-a",
                        }
                    }
                },
                "nl2bash_judge_model": {
                    "responses_api_models": {
                        "local_vllm_model": {
                            "base_url": "http://nl2bash-one/v1",
                            "model": "nl2bash-model-a",
                        }
                    }
                },
            }
        }
    }
    routing_changed = {
        "env": {
            "nemo_gym": {
                "genrm_model": {
                    "responses_api_models": {
                        "genrm_model": {
                            "base_url": "http://genrm-two/v1",
                            "model": "genrm-model-a",
                        }
                    }
                },
                "nl2bash_judge_model": {
                    "responses_api_models": {
                        "local_vllm_model": {
                            "base_url": "http://nl2bash-two/v1",
                            "model": "nl2bash-model-a",
                        }
                    }
                },
            }
        }
    }
    model_changed = {
        "env": {
            "nemo_gym": {
                **base["env"]["nemo_gym"],
                "genrm_model": {
                    "responses_api_models": {
                        "genrm_model": {
                            "base_url": "http://genrm-two/v1",
                            "model": "genrm-model-b",
                        }
                    }
                },
            }
        }
    }

    fingerprint = bootstrap_fingerprint(cast(Any, _DumpedConfig(base)))
    assert fingerprint == bootstrap_fingerprint(
        cast(Any, _DumpedConfig(routing_changed))
    )
    assert fingerprint != bootstrap_fingerprint(cast(Any, _DumpedConfig(model_changed)))
    identity = bootstrap_compatibility_identity(cast(Any, _DumpedConfig(base)))
    assert (
        "base_url"
        not in identity.config["env"]["nemo_gym"]["genrm_model"][
            "responses_api_models"
        ]["genrm_model"]
    )
    assert (
        "base_url"
        not in identity.config["env"]["nemo_gym"]["nl2bash_judge_model"][
            "responses_api_models"
        ]["local_vllm_model"]
    )


@pytest.mark.parametrize(
    "field",
    ["concurrency", "nemo_gym_log_dir", "num_processes", "verbose"],
)
def test_bootstrap_fingerprint_ignores_environment_runtime_fields(
    field: str,
) -> None:
    base = {
        "env": {
            "nemo_gym": {
                "service": {
                    "model": "model-a",
                    field: "runtime-one",
                }
            }
        }
    }
    runtime_changed = {
        "env": {
            "nemo_gym": {
                "service": {
                    "model": "model-a",
                    field: "runtime-two",
                }
            }
        }
    }

    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) == (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(runtime_changed)))
    )


@pytest.mark.parametrize(
    "field",
    ["hf_token", "judge_api_key", "policy_api_key", "wandb_api_key"],
)
def test_bootstrap_fingerprint_ignores_nested_environment_credentials(
    field: str,
) -> None:
    base = {
        "env": {
            "nemo_gym": {
                "service": {
                    "model": "model-a",
                    field: "credential-one",
                }
            }
        }
    }
    credential_changed = {
        "env": {
            "nemo_gym": {
                "service": {
                    "model": "model-a",
                    field: "credential-two",
                }
            }
        }
    }

    identity = bootstrap_compatibility_identity(cast(Any, _DumpedConfig(base)))
    assert field not in identity.config["env"]["nemo_gym"]["service"]
    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) == (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(credential_changed)))
    )


def test_bootstrap_fingerprint_keeps_environment_token_semantics() -> None:
    base = {
        "env": {
            "nemo_gym": {
                "service": {
                    "model": "model-a",
                    "max_tokens": 1024,
                    "tokenizer": "tokenizer-a",
                }
            }
        }
    }
    max_tokens_changed = {
        "env": {
            "nemo_gym": {
                "service": {
                    **base["env"]["nemo_gym"]["service"],
                    "max_tokens": 2048,
                }
            }
        }
    }

    identity = bootstrap_compatibility_identity(cast(Any, _DumpedConfig(base)))
    assert identity.config["env"]["nemo_gym"]["service"] == {
        "max_tokens": 1024,
        "model": "model-a",
        "tokenizer": "tokenizer-a",
    }
    assert bootstrap_fingerprint(cast(Any, _DumpedConfig(base))) != (
        bootstrap_fingerprint(cast(Any, _DumpedConfig(max_tokens_changed)))
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
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    first = _commit_snapshot(anchor, mutation_version=1)
    second = _commit_snapshot(anchor, mutation_version=2)

    resolved = resolve_latest_snapshot(
        anchor,
        expected_train_step=0,
        expected_trainer_version=0,
        expected_bootstrap_fingerprint=_test_bootstrap_fingerprint(),
    )

    assert resolved is not None
    assert resolved.path == second
    assert resolved.manifest.mutation_version == 2
    assert first.is_dir()


def test_resolver_falls_back_from_corrupt_newest_snapshot(tmp_path):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    first = _commit_snapshot(anchor, mutation_version=1)
    second = _commit_snapshot(anchor, mutation_version=2)
    (second / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text("not-json")

    resolved = resolve_latest_snapshot(
        anchor,
        expected_train_step=0,
        expected_trainer_version=0,
        expected_bootstrap_fingerprint=_test_bootstrap_fingerprint(),
    )

    assert resolved is not None
    assert resolved.path == first


def test_resolver_ignores_unpublished_temporary_snapshot(tmp_path):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    published = _commit_snapshot(anchor, mutation_version=1)
    incomplete = anchor / "rollout_snapshots" / "tmp_snapshot_000002"
    incomplete.mkdir()
    (incomplete / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text("{}")

    resolved = resolve_latest_snapshot(
        anchor,
        expected_train_step=0,
        expected_trainer_version=0,
        expected_bootstrap_fingerprint=_test_bootstrap_fingerprint(),
    )

    assert resolved is not None
    assert resolved.path == published


def test_resolver_fails_when_no_committed_snapshot_matches_anchor(tmp_path):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    _commit_snapshot(
        anchor,
        mutation_version=1,
        fingerprint=_test_bootstrap_fingerprint("different"),
    )

    with pytest.raises(ValueError, match="trainer-anchor mismatch"):
        resolve_latest_snapshot(
            anchor,
            expected_train_step=0,
            expected_trainer_version=0,
            expected_bootstrap_fingerprint=_test_bootstrap_fingerprint(),
        )


def test_commit_snapshot_flushes_payload_before_publication(tmp_path, monkeypatch):
    anchor = tmp_path / "step_1"
    anchor.mkdir()
    tmp_snapshot, final_snapshot, _ = prepare_snapshot_paths(anchor)
    (tmp_snapshot / "payload").write_text("payload")
    fsync_tree = Mock()
    fsync_directory = Mock()
    monkeypatch.setattr(rollout_checkpoint, "_fsync_tree", fsync_tree)
    monkeypatch.setattr(rollout_checkpoint, "_fsync_directory", fsync_directory)

    commit_snapshot(tmp_snapshot, final_snapshot, keep_latest_k=1)

    fsync_tree.assert_called_once_with(tmp_snapshot)
    assert fsync_directory.call_args_list == [call(anchor / "rollout_snapshots")]
    assert final_snapshot.is_dir()
    assert not tmp_snapshot.exists()


def test_commit_snapshot_prunes_oldest_committed_snapshot(tmp_path):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    first = _commit_snapshot(anchor, mutation_version=1)
    second = _commit_snapshot(anchor, mutation_version=2)
    third_tmp, third, _ = prepare_snapshot_paths(anchor)
    manifest = RolloutSnapshotManifest(
        schema_version=ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        base_train_step=0,
        trainer_version=0,
        current_epoch=2,
        sampler_dispatch_index=2,
        mutation_version=3,
        rolled_back_train_group_count=0,
        bootstrap_fingerprint=_test_bootstrap_fingerprint(),
    )
    (third_tmp / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict())
    )

    commit_snapshot(third_tmp, third, keep_latest_k=2)

    assert not first.exists()
    assert second.is_dir()
    assert third.is_dir()


def test_commit_snapshot_removes_stale_snapshot_from_live_namespace_before_delete(
    tmp_path,
    monkeypatch,
):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    first = _commit_snapshot(anchor, mutation_version=1)
    _commit_snapshot(anchor, mutation_version=2)
    third_tmp, third, _ = prepare_snapshot_paths(anchor)
    manifest = RolloutSnapshotManifest(
        schema_version=ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        base_train_step=0,
        trainer_version=0,
        current_epoch=2,
        sampler_dispatch_index=2,
        mutation_version=3,
        rolled_back_train_group_count=0,
        bootstrap_fingerprint=_test_bootstrap_fingerprint(),
    )
    (third_tmp / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict())
    )
    real_rmtree = rollout_checkpoint.shutil.rmtree

    def fail_trash_delete(path: Path) -> None:
        path = Path(path)
        if path.name.startswith("trash_snapshot_"):
            raise OSError("simulated delete failure")
        real_rmtree(path)

    monkeypatch.setattr(rollout_checkpoint.shutil, "rmtree", fail_trash_delete)

    with pytest.raises(OSError, match="simulated delete failure"):
        commit_snapshot(third_tmp, third, keep_latest_k=2)

    assert not first.exists()
    assert (first.parent / f"trash_{first.name}").is_dir()
    assert third.is_dir()
    resolved = resolve_latest_snapshot(
        anchor,
        expected_train_step=0,
        expected_trainer_version=0,
        expected_bootstrap_fingerprint=_test_bootstrap_fingerprint(),
    )
    assert resolved is not None
    assert resolved.path == third


def test_prepare_snapshot_paths_sweeps_interrupted_snapshot_garbage(tmp_path):
    anchor = _ensure_test_bootstrap_anchor(tmp_path)
    root = anchor / rollout_checkpoint.ROLLOUT_SNAPSHOTS_DIRNAME
    stale_tmp = root / "tmp_snapshot_000001"
    stale_trash = root / "trash_snapshot_000002"
    stale_tmp.mkdir(parents=True)
    stale_trash.mkdir()
    (stale_tmp / "stale").write_text("stale")
    (stale_trash / "stale").write_text("stale")

    tmp_path, final_path, sequence = prepare_snapshot_paths(anchor)

    assert not (root / "trash_snapshot_000002").exists()
    assert tmp_path == root / "tmp_snapshot_000001"
    assert not (tmp_path / "stale").exists()
    assert final_path == root / "snapshot_000001"
    assert sequence == 1


def test_manifest_rejects_bool_for_integer_field():
    raw = {
        "schema_version": ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        "base_train_step": 0,
        "trainer_version": 0,
        "current_epoch": 0,
        "sampler_dispatch_index": -1,
        "mutation_version": 0,
        "rolled_back_train_group_count": 0,
        "bootstrap_fingerprint": "fingerprint-v1",
    }
    raw["mutation_version"] = True

    with pytest.raises(ValueError, match="mutation_version.*integer"):
        RolloutSnapshotManifest.from_mapping(raw)


def test_manifest_rejects_dispatch_index_below_initial_state():
    raw = {
        "schema_version": ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        "base_train_step": 0,
        "trainer_version": 0,
        "current_epoch": 0,
        "sampler_dispatch_index": -2,
        "mutation_version": 0,
        "rolled_back_train_group_count": 0,
        "bootstrap_fingerprint": "fingerprint-v1",
    }

    with pytest.raises(ValueError, match="sampler_dispatch_index.*at least -1"):
        RolloutSnapshotManifest.from_mapping(raw)
