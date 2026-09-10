# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from nemo_rl.weight_sync.model_express import ModelExpressWeightSynchronizer
from nemo_rl.weight_sync.factory import create_weight_synchronizer


def _sync():
    policy = MagicMock()
    policy.initialize_model_express.return_value = ["publisher:global-rank:0"]
    generation = MagicMock()
    generation.cfg = {"model_name": "test/model"}
    control = MagicMock()
    control.create_weight_version.return_value = SimpleNamespace(
        version_id="version-1",
        ref=SimpleNamespace(version_id="version-1"),
    )
    control.get_weight_version.return_value = SimpleNamespace(
        state=SimpleNamespace(value="READY")
    )
    return (
        ModelExpressWeightSynchronizer(
            policy=policy,
            generation=generation,
            control_client=control,
            payload_format=SimpleNamespace(value="FULL_TENSOR"),
        ),
        policy,
        generation,
        control,
    )


def test_sync_runs_complete_version_lifecycle():
    sync, policy, generation, control = _sync()

    sync.init_communicator()
    sync.sync_weights(timer=MagicMock(time=lambda _name: nullcontext()))

    policy.initialize_model_express.assert_called_once_with(server_url=None)
    generation.initialize_model_express.assert_called_once_with(server_url=None)
    control.create_weight_version.assert_called_once()
    policy.publish_model_express_version.assert_called_once()
    generation.update_weights_from_model_express.assert_called_once()
    control.delete_weight_version.assert_called_once_with("version-1")
    policy.release_model_express_version.assert_called_once()
    assert sync.is_stale is False


def test_sync_treats_duplicate_slots_as_redundant_publishers():
    sync, policy, _generation, control = _sync()
    policy.initialize_model_express.return_value = [
        "megatron:partition:a",
        "megatron:partition:a",
    ]

    sync.init_communicator()
    sync.sync_weights()

    assert control.create_weight_version.call_args.kwargs["expected_source_slots"] == [
        "megatron:partition:a"
    ]


def test_sync_retires_and_releases_version_when_update_fails():
    sync, policy, generation, control = _sync()
    sync.init_communicator()
    generation.update_weights_from_model_express.side_effect = RuntimeError(
        "apply failed"
    )

    with pytest.raises(RuntimeError, match="apply failed"):
        sync.sync_weights()

    assert control.mock_calls[-1] == call.delete_weight_version("version-1")
    policy.release_model_express_version.assert_called_once()
    assert sync.is_stale is True


def test_retry_after_failed_transfer_uses_fresh_idempotency_key():
    sync, _policy, generation, control = _sync()
    sync.init_communicator()
    generation.update_weights_from_model_express.side_effect = [
        RuntimeError("apply failed"),
        None,
    ]

    with pytest.raises(RuntimeError, match="apply failed"):
        sync.sync_weights()
    sync.sync_weights()

    first, second = control.create_weight_version.call_args_list
    assert "version_number" not in first.kwargs
    assert "version_number" not in second.kwargs
    assert first.kwargs["idempotency_key"].split(":")[1] == "1"
    assert second.kwargs["idempotency_key"].split(":")[1] == "1"
    assert first.kwargs["idempotency_key"] != second.kwargs["idempotency_key"]
    assert sync.is_stale is False


def test_sync_requires_ready_version_before_generator_update():
    sync, policy, generation, control = _sync()
    sync.init_communicator()
    control.get_weight_version.return_value = SimpleNamespace(
        state=SimpleNamespace(value="STAGING")
    )

    with pytest.raises(RuntimeError, match="did not become READY"):
        sync.sync_weights()

    generation.update_weights_from_model_express.assert_not_called()
    control.delete_weight_version.assert_called_once_with("version-1")
    policy.release_model_express_version.assert_called_once()


@pytest.mark.parametrize(
    ("generation_backend", "colocated"),
    [
        ("vllm", False),
        ("sglang", True),
        ("megatron", False),
        ("dynamo", True),
    ],
)
def test_factory_routes_model_express_without_capability_gates(
    generation_backend: str, colocated: bool
):
    policy = MagicMock()
    policy.cfg = {"megatron_cfg": {"enabled": False}}
    generation = MagicMock()
    generation.cfg = {
        "model_name": "test/model",
        "refit_transport": "model_express",
        "refit_cfg": SimpleNamespace(
            model_express=SimpleNamespace(server_url="mx-server:50051")
        ),
    }

    with patch(
        "nemo_rl.weight_sync.model_express.ModelExpressWeightSynchronizer"
    ) as constructor:
        result = create_weight_synchronizer(
            policy=policy,
            generation=generation,
            generation_backend=generation_backend,
            colocated=colocated,
        )

    assert result is constructor.return_value
    constructor.assert_called_once_with(
        policy=policy,
        generation=generation,
        server_url="mx-server:50051",
    )
