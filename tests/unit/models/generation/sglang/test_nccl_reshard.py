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

"""Dense SGLang destination layout and serving-lifecycle correctness."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker
from nemo_rl.models.generation.sglang.utils.nccl_reshard import (
    build_sglang_refit_param_map,
)
from nemo_rl.models.generation.sglang.utils.train_utils import MultiprocessingSerializer
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    check_nccl_reshard_refit_support,
)
from nemo_rl.weight_sync.sglang_weight_synchronizer import (
    SGLangNcclReshardWeightSynchronizer,
)


def _model(*, tp_size: int, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    model = torch.nn.Module()
    model.add_module("model", torch.nn.Module())
    model.model.add_module("layers", torch.nn.ModuleList([torch.nn.Module()]))
    mlp = torch.nn.Module()
    model.model.layers[0].add_module("mlp", mlp)
    mlp.add_module("gate_up_proj", torch.nn.Linear(4, 16 // tp_size, bias=False))
    mlp.add_module("down_proj", torch.nn.Linear(8 // tp_size, 4, bias=False))
    return model.to(dtype=dtype)


def _metadata() -> dict:
    prefix = "model.layers.0.mlp"
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": f"{prefix}.{proj}.weight",
                    "global_shape": shape,
                    "dtype": "torch.bfloat16",
                }
                for proj, shape in (
                    ("gate_proj", [8, 4]),
                    ("up_proj", [8, 4]),
                    ("down_proj", [4, 8]),
                )
            ]
        },
    }


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_bulk_destinations_match_independent_hf_shards(tp_size: int) -> None:
    full_gate = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    full_up = full_gate + 100
    full_down = full_gate.T.contiguous() + 200
    for rank in range(tp_size):
        model = _model(tp_size=tp_size)
        mapping = build_sglang_refit_param_map(model, _metadata(), tp_size=tp_size)
        for projection, full, dim in (
            ("gate_proj", full_gate, 0),
            ("up_proj", full_up, 0),
            ("down_proj", full_down, 1),
        ):
            shard = full.chunk(tp_size, dim=dim)[rank]
            mapping.specs[f"model.layers.0.mlp.{projection}.weight"].base.copy_(shard)
        mlp = model.model.layers[0].mlp
        torch.testing.assert_close(
            mlp.gate_up_proj.weight,
            torch.cat((full_gate.chunk(tp_size)[rank], full_up.chunk(tp_size)[rank])),
        )
        torch.testing.assert_close(
            mlp.down_proj.weight, full_down.chunk(tp_size, dim=1)[rank]
        )


def test_wrong_destination_dtype_rejected_before_transfer() -> None:
    with pytest.raises(ValueError, match="store BF16"):
        build_sglang_refit_param_map(
            _model(tp_size=2, dtype=torch.float32), _metadata(), tp_size=2
        )


def test_wrong_tp_layout_rejected_before_transfer() -> None:
    with pytest.raises(ValueError, match="expected contiguous"):
        build_sglang_refit_param_map(_model(tp_size=1), _metadata(), tp_size=2)


def _config() -> dict:
    return {
        "precision": "bfloat16",
        "offload_policy_before_refit": False,
        "megatron_cfg": {"enabled": True},
        "automodel_cfg": {"enabled": False},
        "generation": {
            "backend": "sglang",
            "refit_transport": "nccl_reshard",
            "colocated": {"enabled": False},
            "sglang_cfg": {
                "tp_size": 2,
                "pp_size": 1,
                "ep_size": 1,
                "dp_size": 1,
                "quantization": {"scheme": "bf16"},
                "sglang_fault_tolerance_config": {"use_fault_tolerance": False},
                "sglang_server_config": {
                    "num_gpus": 2,
                    "pause_generation_mode": "abort",
                    "needs_offload": False,
                },
            },
        },
    }


def _synchronizer() -> tuple[SGLangNcclReshardWeightSynchronizer, MagicMock]:
    config = _config()
    policy = MagicMock(cfg=config)
    generation = MagicMock(
        cfg=config["generation"],
        num_gpus_per_engine=2,
        num_gpus_per_node=4,
        pause_generation_mode="abort",
    )
    inference_cluster = MagicMock()
    inference_cluster.world_size.return_value = 2
    sync = create_weight_synchronizer(
        policy,
        generation,
        generation_backend="sglang",
        colocated=False,
        train_cluster=MagicMock(),
        inference_cluster=inference_cluster,
        refit_timeout_s=30,
    )
    assert isinstance(sync, SGLangNcclReshardWeightSynchronizer)
    return sync, generation


def test_setup_accepts_dense_bf16_sglang() -> None:
    check_nccl_reshard_refit_support(SimpleNamespace(policy=_config()))


@pytest.mark.parametrize("key,value", [("pp_size", 2), ("ep_size", 2), ("dp_size", 2)])
def test_setup_rejects_unsupported_parallelism(key: str, value: int) -> None:
    config = _config()
    config["generation"]["sglang_cfg"][key] = value
    with pytest.raises(ValueError, match="PP=EP=DP=1"):
        check_nccl_reshard_refit_support(SimpleNamespace(policy=config))


def test_sglang_lifecycle_reuses_shared_transport_and_connects_once() -> None:
    sync, generation = _synchronizer()
    events = []
    generation.begin_weight_update.side_effect = lambda: events.append("begin")
    generation.end_weight_update.side_effect = lambda: events.append("end")
    generation.continue_generation.side_effect = lambda: events.append("resume")
    with patch.object(sync, "_transport") as transport:
        transport.init_communicator.side_effect = lambda: events.append("connect")
        transport.sync_weights.side_effect = lambda **kwargs: events.append("transfer")
        sync.sync_weights()
        sync.sync_weights()
        transport.init_communicator.assert_called_once()
    assert events == [
        "begin",
        "connect",
        "transfer",
        "end",
        "resume",
        "begin",
        "transfer",
        "end",
        "resume",
    ]
    assert not sync.is_stale


def test_failed_transfer_keeps_serving_paused_and_latches_failure() -> None:
    sync, generation = _synchronizer()
    with patch.object(sync, "_transport") as transport:
        transport.sync_weights.side_effect = RuntimeError("injected transfer failure")
        with pytest.raises(RuntimeError, match="injected transfer failure"):
            sync.sync_weights()
        with pytest.raises(RuntimeError, match="restart the run"):
            sync.sync_weights()
        transport.sync_weights.assert_called_once()
    generation.continue_generation.assert_not_called()
    generation.end_weight_update.assert_not_called()
    assert sync.is_stale


def test_control_payload_can_be_consumed_by_independent_tp_processes() -> None:
    worker_class = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = worker_class.__new__(worker_class)
    worker.sglang_cfg = {"sglang_cfg": {"tp_size": 2}}
    worker._make_request = MagicMock(return_value={"success": True})
    payload = {"refit_timeout_s": 30}
    assert worker.nccl_reshard_command("refit", payload)
    body = worker._make_request.call_args.args[1]
    serialized = body["serialized_named_tensors"]
    assert len(serialized) == 2
    for encoded in serialized:
        [(operation, control)] = MultiprocessingSerializer.deserialize(encoded)
        assert operation == "refit"
        assert json.loads(bytes(control.tolist())) == payload
        receiver = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import base64, json, multiprocessing, pickle, sys; "
                "multiprocessing.current_process().authkey = b'independent-sglang'; "
                "[(operation, control)] = pickle.loads(base64.b64decode(sys.stdin.read())); "
                "assert operation == 'refit'; "
                "assert json.loads(bytes(control.tolist())) == {'refit_timeout_s': 30}",
            ],
            input=encoded,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert receiver.returncode == 0, receiver.stderr


def test_shared_mover_uses_sglang_mesh_without_a_ray_worker_group() -> None:
    sync, generation = _synchronizer()
    mover = sync._transport
    mover._train_cluster.world_size.return_value = 1
    mover._train_cluster.num_gpus_per_node = 1
    mover._train_cluster.get_master_address_and_port.return_value = ("127.0.0.1", 1500)
    mover._train_cluster.get_available_address_and_port.return_value = (
        "127.0.0.1",
        1501,
    )
    mover._policy.init_collective.return_value = []
    mover._policy.init_nccl_reshard_comm_group.return_value = []
    mover._policy.prepare_nccl_reshard_refit_info.return_value = _metadata()
    generation.rebuild_collective.return_value = []
    generation.rebuild_nccl_reshard_comm_group.return_value = []
    del generation.worker_group
    with patch("nemo_rl.weight_sync.nccl_reshard_weight_synchronizer.ray.get"):
        mover.init_communicator()
    mover._policy.prepare_nccl_reshard_refit_info.assert_called_once_with(
        {"tp_size": 1, "ep_size": 1, "etp_size": 1, "pp_size": 1},
        {"tp_size": 2, "ep_size": 1, "etp_size": 2, "pp_size": 1},
        1,
        2,
    )
    membership = generation.set_refit_membership.call_args.args[0]
    assert membership.world_size == 3
    assert membership.workers_per_shard == 2
    assert membership.shard_prefixes == {0: 0}
