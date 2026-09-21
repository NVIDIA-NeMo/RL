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
"""Metadata-only finalizer actor boundary tests."""

from __future__ import annotations

import builtins
from dataclasses import fields, replace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import ray
import torch

import nemo_rl.experience.rollout_reassembler_actor as actor_module
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROLLOUT_METRICS
from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
from nemo_rl.experience.metric_utils import RolloutTelemetry
from nemo_rl.experience.rollout_reassembler import FinalizedGroup
from nemo_rl.experience.rollout_reassembler_actor import (
    _FORBIDDEN_RPC_KEYS,
    ReassemblyRequest,
    RolloutReassemblerActor,
    RolloutReassemblerActorConfig,
    assert_metadata_only,
    create_rollout_reassembler_actors,
)


def _request() -> ReassemblyRequest:
    return ReassemblyRequest(
        group_id="group",
        prompt_idx=17,
        rollout_ids=("group_g0",),
        canonical_sample_ids=("group_g0",),
        receipts=(
            {
                "rollout_id": "group_g0",
                "manifest": [
                    {
                        "call_id": "call",
                        "staging_key": "group_g0/call",
                        "delta_len": 2,
                    }
                ],
            },
        ),
        rewards=(1.0,),
        mask_sample=(False,),
        fallback_weight_version=4,
    )


def test_finalizer_request_and_result_are_metadata_only() -> None:
    assert_metadata_only(_request())
    result = FinalizedGroup(
        meta=KVBatchMeta(
            partition_id="canonical",
            task_name="train",
            sample_ids=["group_g0"],
            fields=["input_ids"],
            sequence_lengths=[3],
            tags=[{"weight_version": 4}],
        ),
        group_min_wv=4,
        group_max_wv=4,
        staging_keys=["group_g0/call"],
        metrics={"finalize/total_ms": 1.0},
    )
    assert_metadata_only(result)


def test_finalize_forwards_loss_multiplier_to_reassembler() -> None:
    actor_cls = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._finalizer = MagicMock()
    result = FinalizedGroup(
        meta=None,
        group_min_wv=4,
        group_max_wv=4,
        staging_keys=[],
        dropped=True,
        drop_reason="test",
    )
    actor._finalizer.finalize_group.return_value = result
    request = replace(_request(), loss_multiplier=0.25, rollout_environment="swe")

    assert actor.finalize(request) is result
    actor._finalizer.finalize_group.assert_called_once_with(
        "group",
        ["group_g0"],
        [request.receipts[0]],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=4,
        prompt_idx=17,
        loss_multiplier=0.25,
        rollout_environment="swe",
        canonical_sample_ids=["group_g0"],
    )


def test_finalizer_forwards_mooncake_checkpoint_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_cls = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    command = {"operation": "INFO"}
    response = {"participant_id": "finalizer"}
    dispatch = MagicMock(return_value=response)
    monkeypatch.setattr(actor_module, "run_checkpoint_command", dispatch)

    assert actor.mooncake_checkpoint(command) is response
    dispatch.assert_called_once_with(command)


@pytest.mark.parametrize(
    "payload",
    [
        torch.ones(2),
        {"input_ids": [1, 2]},
        {"routed_experts": [[[[1, 2]]]]},
    ],
)
def test_metadata_guard_rejects_tensor_and_heavy_row_payloads(payload) -> None:
    with pytest.raises(TypeError):
        assert_metadata_only(payload)


def test_rpc_dataclass_fields_are_classified() -> None:
    """A new field on either RPC dataclass must be a deliberate choice.

    assert_metadata_only cannot tell a heavy list[int] of token ids from a short
    list of metadata, so _FORBIDDEN_RPC_KEYS is maintained by hand. Pinning the
    inventory makes a new field fail here until someone decides whether it is
    light enough to cross the wire.
    """
    assert {f.name for f in fields(ReassemblyRequest)} == {
        "group_id",
        "rollout_ids",
        "canonical_sample_ids",
        "receipts",
        "rewards",
        "fallback_weight_version",
        "prompt_idx",
        "mask_sample",
        "loss_multiplier",
        "rollout_environment",
        "telemetry",
    }
    assert {f.name for f in fields(FinalizedGroup)} == {
        "meta",
        "group_min_wv",
        "group_max_wv",
        "staging_keys",
        "canonical_output_tokens",
        "metrics",
        "dropped",
        "drop_reason",
        "valid_row_count",
        "total_row_count",
    }


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("max_seq_len", [3, 4])
def test_finalizer_retains_only_complete_sibling_telemetry(
    complete, max_seq_len, caplog
):
    actor_cls = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._max_seq_len = max_seq_len
    actor._finalizer = MagicMock()
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["group_g0"],
        sequence_lengths=[3],
    )
    result = FinalizedGroup(meta=meta, group_min_wv=4, group_max_wv=4, staging_keys=[])
    actor._finalizer.finalize_group.return_value = result
    snapshot = RolloutTelemetry(
        "swe",
        {"environment/swe/turns_per_sample": 3.0},
        {"environment/swe/sample_count": 1.0},
    )
    request = replace(
        _request(),
        rollout_environment="swe",
        telemetry=(snapshot if complete else None,),
    )
    assert actor.finalize(request) is result
    if complete:
        assert meta.extra_info[ROLLOUT_METRICS][0][
            "environment/swe/turns_per_sample/histogram"
        ] == [3.0]
        assert_metadata_only(result)
        assert meta.extra_info[ROLLOUT_METRICS][0]["truncation_rate"] == int(
            max_seq_len == 3
        )
        assert meta.extra_info[ROLLOUT_METRICS][0][
            "environment/swe/truncated/histogram"
        ] == [int(max_seq_len == 3)]
    else:
        assert ROLLOUT_METRICS not in meta.extra_info
        assert "lacks complete rollout telemetry" in caplog.text


def test_finalizer_rejects_misaligned_telemetry_before_publication():
    actor_cls = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._finalizer = MagicMock()
    with pytest.raises(ValueError, match="align with logical siblings"):
        actor.finalize(replace(_request(), telemetry=(None, None)))
    with pytest.raises(ValueError, match="environment mismatch"):
        actor.finalize(
            replace(_request(), telemetry=(RolloutTelemetry("wrong", {}, {}),))
        )
    actor._finalizer.finalize_group.assert_not_called()


@pytest.mark.parametrize("key", sorted(_FORBIDDEN_RPC_KEYS))
def test_every_forbidden_key_is_rejected(key) -> None:
    """Removing an entry from the denylist should fail loudly."""
    with pytest.raises(TypeError, match="forbidden heavy field"):
        assert_metadata_only({key: [1, 2, 3]})


@pytest.mark.parametrize(
    ("startup_fails", "cleanup_fails"),
    [(False, False), (True, False), (True, True)],
    ids=["ready", "startup-failure", "cleanup-failure"],
)
def test_factory_selects_gym_environment_and_waits_for_dependencies(
    startup_fails: bool,
    cleanup_fails: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actor_fqn = "nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor"
    assert ACTOR_ENVIRONMENTS[actor_fqn] == ["nemo_gym"]
    config = RolloutReassemblerActorConfig(
        partition_id="canonical",
        staging_partition="staging",
        pad_token_id=0,
        router_replay_enabled=False,
        defer_routed_experts_to_policy=False,
        max_seq_len=4096,
    )
    dp_config = {"enabled": True, "impl": "transfer_queue", "backend": "simple"}
    actors = [MagicMock(), MagicMock()]
    runtime_env = {"py_executable": "/gym-venv/bin/python"}
    with (
        patch.object(
            actor_module, "make_actor_runtime_env", return_value=runtime_env
        ) as make_env,
        patch.object(RolloutReassemblerActor, "options") as options,
        patch.object(ray, "get") as get,
        patch.object(ray, "kill") as kill,
    ):
        options.return_value.remote.side_effect = actors
        if startup_fails:
            startup_error = ray.exceptions.RayError("missing nemo_gym")
            get.side_effect = startup_error
            if cleanup_fails:
                kill.side_effect = [RuntimeError("kill failed"), None]
            with pytest.raises(ray.exceptions.RayError) as exc_info:
                create_rollout_reassembler_actors(dp_config, config, num_workers=2)
            assert exc_info.value is startup_error
            assert kill.call_args_list == [call(actor) for actor in actors]
            if cleanup_fails:
                assert (
                    "finalizer actor termination failed: kill failed"
                    in capsys.readouterr().out
                )
        else:
            assert (
                create_rollout_reassembler_actors(dp_config, config, num_workers=2)
                == actors
            )
            kill.assert_not_called()

        make_env.assert_called_once_with(actor_fqn)
        assert options.call_args_list == [call(runtime_env=runtime_env)] * 2
        assert (
            options.return_value.remote.call_args_list == [call(dp_config, config)] * 2
        )
        for actor in actors:
            actor.check_dependencies.remote.assert_called_once_with()
        get.assert_called_once_with(
            [actor.check_dependencies.remote.return_value for actor in actors]
        )


def test_dependency_check_propagates_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    import_error = ModuleNotFoundError("No module named 'nemo_gym'", name="nemo_gym")

    def fail_rebuild_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "nemo_gym.token_id_capture.staging.rebuild":
            raise import_error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_rebuild_import)
    actor = object.__new__(RolloutReassemblerActor.__ray_metadata__.modified_class)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        actor.check_dependencies()
    assert exc_info.value is import_error
