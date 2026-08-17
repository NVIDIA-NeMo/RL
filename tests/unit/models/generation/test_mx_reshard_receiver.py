# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.distributed.mx_vllm_reshard_receiver import (
    MxVllmReshardReceiver,
)


def _build_receiver():
    manager = MagicMock()
    receiver_client = MagicMock()
    raw_receiver = MagicMock(_manager=manager, _mx_client=receiver_client)
    client = MagicMock()
    rendezvous = MagicMock()
    with (
        patch(
            "modelexpress.engines.vllm.refit.receiver.VllmReshardReceiver",
            return_value=raw_receiver,
        ) as receiver_cls,
        patch("modelexpress.client.MxClient", return_value=client) as client_cls,
        patch(
            "modelexpress.refit.reshard.rendezvous.MxReshardRendezvous",
            return_value=rendezvous,
        ) as rendezvous_cls,
    ):
        wrapper = MxVllmReshardReceiver(
            model="model",
            vllm_config="vllm-config",
            model_config="model-config",
            model_name="Qwen/Qwen3-30B",
            server_url="mx.example:8001",
            agent_name="nemo-rl-vllm-3",
            local_rank=1,
            global_rank=3,
            num_trainer_sources=16,
            device="cuda:1",
            listen_port=19003,
            timeout=77.0,
        )
    return (
        wrapper,
        raw_receiver,
        manager,
        receiver_client,
        client,
        rendezvous,
        receiver_cls,
        client_cls,
        rendezvous_cls,
    )


def test_exact_modelexpress_635_constructor_arguments():
    (
        _wrapper,
        _raw,
        _manager,
        _receiver_client,
        client,
        _rendezvous,
        receiver_cls,
        client_cls,
        rendezvous_cls,
    ) = _build_receiver()

    client_cls.assert_called_once_with(server_url="mx.example:8001")
    rendezvous_cls.assert_called_once_with(
        client,
        role="inference",
        rank=3,
        model_name="Qwen/Qwen3-30B",
    )
    receiver_cls.assert_called_once_with(
        model="model",
        vllm_config="vllm-config",
        model_config="model-config",
        model_name="Qwen/Qwen3-30B",
        mx_server="mx.example:8001",
        agent_name="nemo-rl-vllm-3",
        local_rank=1,
        global_rank=3,
        num_trainer_sources=16,
        device="cuda:1",
        listen_port=19003,
        timeout=77.0,
    )


def test_version_mismatch_fails_before_receiver_install():
    wrapper, raw, _, _, _, rendezvous, *_ = _build_receiver()
    rendezvous.discover_trainers.return_value = [
        SimpleNamespace(publisher_step=8) for _ in range(16)
    ]

    with pytest.raises(RuntimeError, match="requested 9"):
        wrapper.update_weights(9)

    raw.update_weights.assert_not_called()


def test_matching_version_is_forwarded_with_timeout():
    wrapper, raw, _, _, _, rendezvous, *_ = _build_receiver()
    rendezvous.discover_trainers.return_value = [
        SimpleNamespace(publisher_step=9) for _ in range(16)
    ]
    raw.update_weights.return_value = {"step": 9}

    assert wrapper.update_weights(9) == {"step": 9}
    rendezvous.discover_trainers.assert_called_once_with(16, timeout=77.0)
    raw.update_weights.assert_called_once_with(9, timeout=77.0)


def test_phase_telemetry_cannot_fail_a_successful_refit(capsys):
    """Reporting must never break the operation it measures.

    The phase split reads fields off the rendezvous payloads, and an earlier
    version accessed ``payload.tensors`` directly. Any payload shape without that
    field then raised AttributeError *after* the weights had already installed,
    turning a successful refit into a failed one.
    """
    wrapper, raw, _, _, _, rendezvous, *_ = _build_receiver()
    # Payloads carry the version stamp but no shard table, which is exactly the
    # shape that used to abort the refit.
    rendezvous.discover_trainers.return_value = [
        SimpleNamespace(publisher_step=9) for _ in range(16)
    ]
    raw.update_weights.return_value = {"step": 9}

    assert wrapper.update_weights(9) == {"step": 9}

    record = json.loads(capsys.readouterr().out.split("MX_RECV_PHASE ", 1)[1])
    assert record["step"] == 9
    assert record["rank"] == 3
    assert record["trainer_sources"] == 16
    # Absent shard tables count as zero rather than aborting.
    assert record["tensors_seen"] == 0
    assert record["discover_s"] >= 0.0 and record["mx_update_s"] >= 0.0


def test_phase_telemetry_counts_shard_table_entries(capsys):
    """``tensors_seen`` is what showed the quorum cost scales with source count
    rather than bytes moved, so it has to actually count."""
    wrapper, raw, _, _, _, rendezvous, *_ = _build_receiver()
    rendezvous.discover_trainers.return_value = [
        SimpleNamespace(publisher_step=4, tensors=tuple(range(100))) for _ in range(16)
    ]
    raw.update_weights.return_value = {"step": 4}

    wrapper.update_weights(4)

    record = json.loads(capsys.readouterr().out.split("MX_RECV_PHASE ", 1)[1])
    assert record["tensors_seen"] == 1600


def test_shutdown_releases_manager_and_both_clients():
    wrapper, _, manager, receiver_client, client, _, *_ = _build_receiver()

    wrapper.shutdown()
    wrapper.shutdown()

    manager.shutdown.assert_called_once()
    receiver_client.close.assert_called_once()
    client.close.assert_called_once()
