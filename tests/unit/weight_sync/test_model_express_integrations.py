# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.weight_sync.model_express.generator import (
    ModelExpressGeneratorIntegration,
)
from nemo_rl.weight_sync.model_express.trainer import (
    ModelExpressTrainerIntegration,
)


def test_generator_update_releases_staged_handle_after_apply():
    client = MagicMock()
    staged = client.stage_weight.return_value
    integration = ModelExpressGeneratorIntegration(client)
    version = SimpleNamespace(version_id="version-1")

    assert integration.update(version) is True

    client.stage_weight.assert_called_once_with(version=version)
    staged.wait.assert_called_once_with()
    client.apply_weight.assert_called_once_with(staged)
    staged.release.assert_called_once_with()


def test_generator_update_releases_staged_handle_after_failure():
    client = MagicMock()
    staged = client.stage_weight.return_value
    client.apply_weight.side_effect = RuntimeError("apply failed")
    integration = ModelExpressGeneratorIntegration(client)

    with pytest.raises(RuntimeError, match="apply failed"):
        integration.update(SimpleNamespace(version_id="version-1"))

    staged.release.assert_called_once_with()


def test_generator_initialize_constructs_public_client():
    client = MagicMock()
    model = object()
    vllm_config = object()
    model_config = SimpleNamespace(model="test/model")

    with patch(
        "modelexpress_rl.ModelExpressGeneratorClient.initialize",
        return_value=client,
    ) as initialize:
        integration = ModelExpressGeneratorIntegration.initialize(
            model=model,
            vllm_config=vllm_config,
            model_config=model_config,
            server_url="mx-server:50051",
        )

    assert integration._client is client
    initialize.assert_called_once_with(
        model=model,
        vllm_config=vllm_config,
        model_config=model_config,
        model_name="test/model",
        server_url="mx-server:50051",
    )


def test_trainer_integration_owns_publication_and_cleanup():
    client = MagicMock()
    staged = client.stage_shard.return_value
    specs = [SimpleNamespace(name="weight")]
    integration = ModelExpressTrainerIntegration(client)
    integration._tensor_specs = specs
    version = SimpleNamespace(version_id="version-1")

    integration.publish(version)
    integration.release(version)
    integration.close()

    client.stage_shard.assert_called_once_with(version=version, tensors=specs)
    staged.publish.assert_called_once_with()
    client.release_version.assert_called_once_with(version=version)
    client.close.assert_called_once_with()


def test_trainer_from_config_constructs_only_public_client():
    client = MagicMock()
    config = {
        "model_name": "test/model",
        "generation": {
            "refit_transport": "model_express",
            "refit_cfg": SimpleNamespace(
                model_express=SimpleNamespace(server_url="mx-server:50051")
            ),
        },
    }

    with patch(
        "modelexpress_rl.ModelExpressTrainerClient.initialize",
        return_value=client,
    ) as initialize:
        integration = ModelExpressTrainerIntegration.from_config(
            config=config,
            local_rank=2,
        )

    assert integration is not None
    assert integration._client is client
    initialize.assert_called_once_with(
        model_name="test/model",
        device_id=2,
        server_url="mx-server:50051",
    )


def test_trainer_initialize_registers_tensors_then_resolves_source_slot():
    client = MagicMock(source_slot_id="publisher:global-rank:0")
    specs = [SimpleNamespace(name="weight", tensor=object())]
    integration = ModelExpressTrainerIntegration(client)

    with patch(
        "nemo_rl.weight_sync.model_express.trainer.build_megatron_tensor_specs",
        return_value=specs,
    ):
        source_slot_id = integration.initialize(
            model_name="test/model",
            conversion_tasks=[],
            transformer_config=object(),
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            server_url="mx-server:50051",
        )

    assert source_slot_id == "publisher:global-rank:0"
    client.register_tensors.assert_called_once_with({"weight": specs[0].tensor})
