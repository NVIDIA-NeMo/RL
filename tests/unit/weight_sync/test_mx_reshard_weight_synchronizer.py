# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from nemo_rl.weight_sync.mx_reshard_weight_synchronizer import (
    MxReshardWeightSynchronizer,
    check_mx_reshard_refit_support,
)


def _cluster(size: int) -> MagicMock:
    cluster = MagicMock()
    cluster.world_size.return_value = size
    return cluster


def _generation() -> MagicMock:
    generation = MagicMock()
    generation.cfg = {"refit_transport": "mx_reshard"}
    return generation


def test_factory_selects_mx_reshard() -> None:
    from nemo_rl.weight_sync.factory import create_weight_synchronizer

    sync = create_weight_synchronizer(
        policy=MagicMock(),
        generation=_generation(),
        generation_backend="vllm",
        colocated=False,
        train_cluster=_cluster(8),
        inference_cluster=_cluster(4),
    )
    assert isinstance(sync, MxReshardWeightSynchronizer)


def test_config_normalizes_mx_scope() -> None:
    from nemo_rl.models.generation.vllm.config import normalize_vllm_refit_config

    config = {
        "refit_transport": "mx_reshard",
        "refit_cfg": {
            "mx_reshard": {
                "server_url": "mx.example:8001",
                "timeout_s": 30,
                "listen_port_base": 19000,
            }
        },
    }
    normalized = normalize_vllm_refit_config(config)  # type: ignore[arg-type]
    assert normalized is not None
    assert normalized.mx_reshard.server_url == "mx.example:8001"
    assert normalized.mx_reshard.timeout_s == 30
    assert normalized.mx_reshard.publisher_listen_port_base == 19000
    assert normalized.mx_reshard.receiver_listen_port_base == 29000


def test_config_normalizes_distinct_role_specific_port_bases() -> None:
    from nemo_rl.models.generation.vllm.config import normalize_vllm_refit_config

    config = {
        "refit_transport": "mx_reshard",
        "refit_cfg": {
            "mx_reshard": {
                "publisher_listen_port_base": 19000,
                "receiver_listen_port_base": 29000,
            }
        },
    }
    normalized = normalize_vllm_refit_config(config)  # type: ignore[arg-type]
    assert normalized is not None
    assert normalized.mx_reshard.publisher_listen_port_base == 19000
    assert normalized.mx_reshard.receiver_listen_port_base == 29000


def test_support_validation_rejects_colocation_and_dtensor() -> None:
    master = SimpleNamespace(
        policy={
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": True},
            },
            "megatron_cfg": {"enabled": True},
            "dtensor_cfg": {"enabled": True},
        }
    )
    with pytest.raises(ValueError, match="mx_reshard refit configuration"):
        check_mx_reshard_refit_support(master)


def _supported_gqa_master(*, heads: int = 64, query_groups: int = 2):
    return SimpleNamespace(
        policy={
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": False},
                "vllm_cfg": {"kv_cache_dtype": "auto"},
            },
            "megatron_cfg": {
                "enabled": True,
                "tensor_model_parallel_size": 8,
                "expert_tensor_parallel_size": 1,
                "num_attention_heads": heads,
                "num_query_groups": query_groups,
            },
            "dtensor_cfg": {"enabled": False},
        }
    )


def test_support_validation_accepts_kv_heads_below_tp() -> None:
    check_mx_reshard_refit_support(_supported_gqa_master())


def test_support_validation_rejects_query_heads_that_do_not_form_groups() -> None:
    with pytest.raises(ValueError, match="divisible by num_query_groups"):
        check_mx_reshard_refit_support(_supported_gqa_master(heads=63, query_groups=2))


def test_result_validation_accepts_flattened_and_nested_ray_shapes() -> None:
    MxReshardWeightSynchronizer._require_all(True, "scalar")
    MxReshardWeightSynchronizer._require_all([True, (True, [True])], "nested")
    with pytest.raises(RuntimeError, match="nested failure"):
        MxReshardWeightSynchronizer._require_all(
            [True, [True, False]], "nested failure"
        )


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_publish_quorum_completes_before_pull(mock_ray: MagicMock) -> None:
    events: list[str] = []
    policy = MagicMock()
    generation = _generation()
    publish_refs = [object(), object()]
    pull_refs = [object(), object()]
    policy.publish_mx_reshard_weights.side_effect = lambda **_: (
        events.append("publish-called") or publish_refs
    )
    generation.update_weights_from_mx_reshard.side_effect = lambda **_: (
        events.append("pull-called") or pull_refs
    )

    def get(refs):
        if refs is publish_refs:
            events.append("publish-complete")
            return [True, True]
        if refs is pull_refs:
            events.append("pull-complete")
            return [True, True]
        raise AssertionError(refs)

    mock_ray.get.side_effect = get
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(2), _cluster(2))
    sync.sync_weights()

    assert events == [
        "publish-called",
        "publish-complete",
        "pull-called",
        "pull-complete",
    ]
    policy.publish_mx_reshard_weights.assert_called_once_with(version=1)
    generation.update_weights_from_mx_reshard.assert_called_once_with(version=1)


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_publish_failure_prevents_pull(mock_ray: MagicMock) -> None:
    policy = MagicMock()
    generation = _generation()
    policy.publish_mx_reshard_weights.return_value = ["publish-ref"]
    mock_ray.get.return_value = [True, False]
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(2), _cluster(2))

    with pytest.raises(RuntimeError, match="publish failed"):
        sync.sync_weights()

    generation.update_weights_from_mx_reshard.assert_not_called()
    assert sync.is_stale


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_receiver_failure_does_not_commit_version(mock_ray: MagicMock) -> None:
    policy = MagicMock()
    generation = _generation()
    policy.publish_mx_reshard_weights.return_value = ["publish-ref"]
    generation.update_weights_from_mx_reshard.return_value = ["pull-ref"]
    mock_ray.get.side_effect = [[True], [False]]
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(1), _cluster(1))

    with pytest.raises(RuntimeError, match="receive failed"):
        sync.sync_weights()

    assert sync.is_stale
    assert sync._version == 0
    assert generation.mock_calls[-1] == call.update_weights_from_mx_reshard(version=1)


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_init_uses_physical_train_world_size(mock_ray: MagicMock) -> None:
    policy = MagicMock()
    generation = _generation()
    policy.init_mx_reshard_publisher.return_value = ["trainer-init"]
    generation.init_mx_reshard_receiver.return_value = ["receiver-init"]
    mock_ray.get.side_effect = [[True] * 16, [True] * 8]
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(16), _cluster(8))

    sync.init_communicator()

    policy.init_mx_reshard_publisher.assert_called_once_with(train_world_size=16)
    generation.init_mx_reshard_receiver.assert_called_once_with(
        train_world_size=16,
        inference_world_size=8,
    )


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_init_rejects_overlapping_port_ranges_before_worker_rpcs(
    mock_ray: MagicMock,
) -> None:
    policy = MagicMock()
    generation = _generation()
    generation.cfg["refit_cfg"] = {
        "mx_reshard": {
            "publisher_listen_port_base": 19000,
            "receiver_listen_port_base": 19001,
        }
    }
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(2), _cluster(2))

    with pytest.raises(ValueError, match="listen port ranges overlap"):
        sync.init_communicator()

    policy.init_mx_reshard_publisher.assert_not_called()
    generation.init_mx_reshard_receiver.assert_not_called()
    mock_ray.get.assert_not_called()


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_shutdown_invokes_both_cleanup_quorums(mock_ray: MagicMock) -> None:
    events: list[str] = []
    policy = MagicMock()
    generation = _generation()
    policy.shutdown_mx_reshard_publisher.side_effect = lambda: (
        events.append("publisher-called") or ["trainer-shutdown"]
    )
    generation.shutdown_mx_reshard_receiver.side_effect = lambda: (
        events.append("receiver-called") or ["receiver-shutdown"]
    )
    mock_ray.get.side_effect = [[True, True], [[True], True]]
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(2), _cluster(2))

    sync.shutdown()
    sync.shutdown()

    policy.shutdown_mx_reshard_publisher.assert_called_once_with()
    generation.shutdown_mx_reshard_receiver.assert_called_once_with()
    assert events == ["receiver-called", "publisher-called"]


@patch("nemo_rl.weight_sync.mx_reshard_weight_synchronizer.ray")
def test_shutdown_attempts_publisher_cleanup_after_receiver_failure(
    mock_ray: MagicMock,
) -> None:
    policy = MagicMock()
    generation = _generation()
    policy.shutdown_mx_reshard_publisher.return_value = ["trainer-shutdown"]
    generation.shutdown_mx_reshard_receiver.return_value = ["receiver-shutdown"]
    mock_ray.get.side_effect = [RuntimeError("receiver cleanup failed"), [True]]
    sync = MxReshardWeightSynchronizer(policy, generation, _cluster(1), _cluster(1))

    with pytest.raises(RuntimeError, match="receiver cleanup failed"):
        sync.shutdown()

    policy.shutdown_mx_reshard_publisher.assert_called_once_with()
