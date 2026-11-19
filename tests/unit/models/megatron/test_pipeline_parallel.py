# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.mcore


@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_world_size",
    return_value=1,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.torch.distributed.broadcast_object_list"
)
def test_loss_metric_broadcast_is_noop_for_pp1(
    mock_broadcast: MagicMock, mock_pp_size: MagicMock
) -> None:
    from nemo_rl.models.megatron.pipeline_parallel import (
        broadcast_loss_metrics_from_last_stage,
    )

    metrics = [{"loss": torch.tensor(0.5)}]

    result = broadcast_loss_metrics_from_last_stage(metrics)

    assert result is metrics
    mock_broadcast.assert_not_called()


@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_world_size",
    return_value=2,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_last_rank",
    return_value=1,
)
@patch("nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_group")
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.is_pipeline_last_stage",
    return_value=True,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.torch.distributed.broadcast_object_list"
)
def test_loss_metric_object_broadcast_materializes_scalar_tensors(
    mock_broadcast: MagicMock,
    mock_is_last_stage: MagicMock,
    mock_pp_group: MagicMock,
    mock_last_rank: MagicMock,
    mock_pp_size: MagicMock,
) -> None:
    from nemo_rl.models.megatron.pipeline_parallel import (
        broadcast_loss_metrics_from_last_stage,
    )

    metrics = [
        {
            "loss": torch.tensor(0.5),
            "num_valid_samples": torch.tensor(3.0),
            "lr": 1.0e-5,
        },
        {
            "loss": torch.tensor(0.3),
            "num_valid_samples": torch.tensor(5.0),
            "lr": 1.0e-5,
        },
    ]

    result = broadcast_loss_metrics_from_last_stage(metrics)

    assert result == [
        {"loss": 0.5, "num_valid_samples": 3.0, "lr": 1.0e-5},
        {"loss": pytest.approx(0.3), "num_valid_samples": 5.0, "lr": 1.0e-5},
    ]
    broadcast_payload = mock_broadcast.call_args.args[0][0]
    assert all(
        not isinstance(value, torch.Tensor)
        for metric in broadcast_payload
        for value in metric.values()
    )


@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_world_size",
    return_value=2,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_last_rank",
    return_value=1,
)
@patch("nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_group")
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.is_pipeline_last_stage",
    return_value=True,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.torch.distributed.broadcast_object_list"
)
def test_loss_metric_object_broadcast_preserves_flat_mtp_dict(
    mock_broadcast: MagicMock,
    mock_is_last_stage: MagicMock,
    mock_pp_group: MagicMock,
    mock_last_rank: MagicMock,
    mock_pp_size: MagicMock,
) -> None:
    from nemo_rl.models.megatron.pipeline_parallel import (
        broadcast_loss_metrics_from_last_stage,
    )

    metrics = {
        "mtp_loss_0": torch.tensor(0.25),
        "mtp_loss_1": torch.tensor(0.125),
    }

    result = broadcast_loss_metrics_from_last_stage(metrics)

    assert result == {"mtp_loss_0": 0.25, "mtp_loss_1": 0.125}
    assert result is mock_broadcast.call_args.args[0][0]
    assert all(not isinstance(value, torch.Tensor) for value in result.values())


@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_world_size",
    return_value=2,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_last_rank",
    return_value=1,
)
@patch("nemo_rl.models.megatron.pipeline_parallel.get_pipeline_model_parallel_group")
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.is_pipeline_last_stage",
    return_value=False,
)
@patch(
    "nemo_rl.models.megatron.pipeline_parallel.torch.distributed.broadcast_object_list"
)
def test_loss_metric_object_broadcast_receives_flat_mtp_dict(
    mock_broadcast: MagicMock,
    mock_is_last_stage: MagicMock,
    mock_pp_group: MagicMock,
    mock_last_rank: MagicMock,
    mock_pp_size: MagicMock,
) -> None:
    from nemo_rl.models.megatron.pipeline_parallel import (
        broadcast_loss_metrics_from_last_stage,
    )

    def receive_metrics(payload: list[object], **_: object) -> None:
        payload[0] = {"mtp_loss_0": 0.25}

    mock_broadcast.side_effect = receive_metrics

    result = broadcast_loss_metrics_from_last_stage()

    assert result == {"mtp_loss_0": 0.25}
